import os
import logging
from torch.backends import cudnn
from utils.logger import setup_logger
from datasets import make_dataset
from model import make_model
from solver import make_optimizer, WarmupMultiStepLR
from loss import make_loss
from processor import do_train
from datasets.bases import TrainImageDataset
from datasets.bases import ValImageDataset
from torch.utils.data import DataLoader
import random
import torch
import time
import torch.nn as nn
import numpy as np
import numpy.ma as ma
import os
import argparse
from config import cfg
import pickle
from datasets.preprocessing import RandomErasing
from datasets.sampler import RandomIdentitySampler
from datasets.make_dataloader import train_collate_fn, val_collate_fn
from functions import keyfromval, strint, ranges, search
from utils.meter import AverageMeter
from utils.metrics import R1_mAP,R1_mAP_eval,R1_mAP_Pseudo,R1_mAP_query_mining



def main():
    
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True
    #parser = argparse.ArgumentParser(description="ReID Baseline Training")
    #parser.add_argument(
        #"--config_file", default="", help="path to config file", type=str)
    
    #parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)

    #args = parser.parse_args()
    config_file = 'configs/baseline_veri_r101_a.yml'
    if config_file != "":
        cfg.merge_from_file(config_file)
    #cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(config_file)

    if config_file != "":
        logger.info("Loaded configuration file {}".format(config_file))
        with open(config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    
    path = 'D:/Python_SMU/Veri/verigms/gms/'
    pkl = {}
    entries = os.listdir(path)
    for name in entries:
        f = open((path+name), 'rb')
        if name=='featureMatrix.pkl':
            s = name[0:13]
        else:
            s = name[0:3]
        pkl[s] = pickle.load(f)
        f.close

    with open('cids.pkl', 'rb') as handle:
        b = pickle.load(handle)

    with open('index.pkl', 'rb') as handle:
        c = pickle.load(handle)

    train_transforms, val_transforms, dataset, train_set, val_set = make_dataset(cfg, pkl_file='index.pkl')

    num_workers = cfg.DATALOADER.NUM_WORKERS
    num_classes = dataset.num_train_pids
    #pkl_f = 'index.pkl'
    pid = 0
    pidx = {}
    for img_path, pid,_,_ in dataset.train:
        path = img_path.split('\\')[-1]
        folder = path[1:4]
        pidx[folder] = pid
        pid+= 1

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, pin_memory=True, collate_fn=train_collate_fn
        )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            pin_memory=True,collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    print("train loader loaded successfully")

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        pin_memory=True,collate_fn=train_collate_fn
    )
    print("val loader loaded successfully")

    if cfg.MODEL.PRETRAIN_CHOICE == 'finetune':
        model = make_model(cfg, num_class=576)
        model.load_param_finetune(cfg.MODEL.PRETRAIN_PATH)
        print('Loading pretrained model for finetuning......')
    else:
        model = make_model(cfg, num_class=num_classes)

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                  cfg.SOLVER.WARMUP_FACTOR,
                                  cfg.SOLVER.WARMUP_EPOCHS, cfg.SOLVER.WARMUP_METHOD)
                            
    print("model,optimizer, loss, scheduler loaded successfully")

    height, width = cfg.INPUT.SIZE_TRAIN
    
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info('start training')

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(len(dataset.query), max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    model.base._freeze_stages()
    logger.info('Freezing the stages number:{}'.format(cfg.MODEL.FROZEN))
    
    data_index = search(pkl)
    print("Ready for training")

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step()
        model.train()
        for n_iter, (img, label, index, pid, cid) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            #img = img.to(device)
            #target = vid.to(device)
            trainX, trainY = torch.zeros((train_loader.batch_size*3,3,height, width), dtype=torch.float32), torch.zeros((train_loader.batch_size*3), dtype = torch.int64)
            
            for i in range(train_loader.batch_size):
                labelx = label[i]
                indexx = index[i]
                cidx = pid[i]
                if indexx >len(pkl[labelx])-1:
                    indexx = len(pkl[labelx])-1
                
                a = pkl[labelx][indexx]
                minpos = np.argmin(ma.masked_where(a==0, a)) 
                pos_dic = train_set[data_index[cidx][1]+minpos]
                #print(pos_dic[1])
                neg_label = int(labelx)

                while True:
                    neg_label = random.choice(range(1, 770))
                    if neg_label is not int(labelx) and os.path.isdir(os.path.join('D:/datasets/veri-split/train', strint(neg_label))) is True:
                        break
                
                negative_label = strint(neg_label)
                neg_cid = pidx[negative_label]
                neg_index = random.choice(range(0, len(pkl[negative_label])))

                neg_dic = train_set[data_index[neg_cid][1]+neg_index]
                trainX[i] = img[i]
                trainX[i+train_loader.batch_size] = pos_dic[0]
                trainX[i+(train_loader.batch_size*2)] = neg_dic[0]
                trainY[i] = cidx
                trainY[i+train_loader.batch_size] = pos_dic[3]
                trainY[i+(train_loader.batch_size*2)] = neg_dic[3]

            #print(trainY)
            trainX = trainX.cuda()
            trainY = trainY.cuda()

            score, feat = model(trainX, trainY)
            loss = loss_func(score, feat, trainY)
            loss.backward()
            optimizer.step()
            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                optimizer_center.step()

            acc = (score.max(1)[1] == trainY).float().mean()
            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            model.eval()
            for n_iter, (img, vid, camid, _,_) in enumerate(val_loader):
                with torch.no_grad():
                    img = img.to(device)
                    feat = model(img)
                    evaluator.update((feat, vid, camid))

            cmc, mAP, _, _, _, _, _ = evaluator.compute()
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

def do_inference_query_mining(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("reid_baseline.test")
    logger.info("Enter inferencing")
    evaluator = R1_mAP_query_mining(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,
                       reranking=cfg.TEST.RE_RANKING,reranking_track=cfg.TEST.RE_RANKING_TRACK)
    evaluator.reset()
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    for n_iter, (img, pid, camid, trackid, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)

            if cfg.TEST.FLIP_FEATS == 'on':
                feat = torch.FloatTensor(img.size(0), 2048).zero_().cuda()
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
                        img = img.index_select(3, inv_idx)
                    f = model(img)
                    feat = feat + f
            else:
                feat = model(img)

            evaluator.update((feat, pid, camid, trackid, imgpath))
            img_path_list.extend(imgpath)

    distmat, img_name_q, img_name_g, qfeats, gfeats = evaluator.compute(cfg.OUTPUT_DIR)

    print('The shape of distmat is: {}'.format(distmat.shape))
    np.save(os.path.join(cfg.OUTPUT_DIR, cfg.TEST.DIST_MAT) , distmat)

    return distmat

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("reid_baseline.test")
    logger.info("Enter inferencing")
    if cfg.TEST.EVAL:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,
                       reranking=cfg.TEST.RE_RANKING,reranking_track=cfg.TEST.RE_RANKING_TRACK)
    evaluator.reset()
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    for n_iter, (img, pid, camid, trackid, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)

            if cfg.TEST.FLIP_FEATS == 'on':
                feat = torch.FloatTensor(img.size(0), 2048).zero_().cuda()
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
                        img = img.index_select(3, inv_idx)
                    f = model(img)
                    feat = feat + f
            else:
                feat = model(img)
            if cfg.TEST.EVAL:
                evaluator.update((feat, pid, camid))
            else:
                evaluator.update((feat, pid, camid, trackid, imgpath))
            img_path_list.extend(imgpath)
    if cfg.TEST.EVAL:
        cmc, mAP, _, _, _, _, _ = evaluator.compute()
        logger.info("Validation Results ")
        logger.info("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    else:
        distmat, img_name_q, img_name_g, qfeats, gfeats = evaluator.compute(cfg.OUTPUT_DIR)
        np.save(os.path.join(cfg.OUTPUT_DIR, cfg.TEST.DIST_MAT) , distmat)
        print('over')

def do_inference_Pseudo_track_rerank(cfg,
                 model,
                val_loader,
                num_query
                 ):
    device = "cuda"

    evaluator = R1_mAP_Pseudo(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    for n_iter, (img, pid, camid, trackid, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)

            if cfg.TEST.FLIP_FEATS == 'on':
                feat = torch.FloatTensor(img.size(0), 2048).zero_().cuda()
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
                        img = img.index_select(3, inv_idx)
                    f = model(img)
                    feat = feat + f
            else:
                feat = model(img)

            evaluator.update((feat, pid, camid, trackid, imgpath))
            img_path_list.extend(imgpath)

    distmat, img_name_q, img_name_g, qfeats, gfeats = evaluator.compute(cfg.OUTPUT_DIR)

    return distmat, img_name_q, img_name_g

if __name__ == '__main__':
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    random.seed(1234)
    main()