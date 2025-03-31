import os
import torch
from pycocotools import coco
import queue
import threading
from model_image import build_model, weights_init
from tools import custom_print
from data_processed import train_data_producer
from train import train
import time
import argparse
torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    # train_val_config
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_path', default='./weights/swin_base_patch4_window7_224_22k.pth',help="vgg path")
    parser.add_argument('--npy_path',default='./utils/new_cat2imgid_dict4000.npy', help="npy path")
    parser.add_argument('--output_dir', default='./results/', help='directory for result')
    parser.add_argument('--gpu_id', default='cuda:0', help='id of gpu')
    parser.add_argument('--img_size', default=224, help='image size')
    parser.add_argument('--lr', default=2e-5, help='learning rate')
    parser.add_argument('--lr_de', default=20000, help='learning rate decay')
    parser.add_argument('--epochs', default=40000, help='epochs')
    parser.add_argument('--bs', default=8, help='batch size')
    parser.add_argument('--gs', default=5, help='group size')
    parser.add_argument('--log_interval', default=100, help='log interval')
    parser.add_argument('--val_interval', default=1000, help='val interval')
    args = parser.parse_args()
    
    annotation_file = './annotations/instances_train2017.json'
    coco_item = coco.COCO(annotation_file=annotation_file)

    train_datapath = './train2017/'

    val_datapath = ['./dataset/CoCA',
                    './dataset/CoSal2015',
                    './dataset/CoSOD3k']

    pretrain_path = args.pretrain_path
    npy = args.npy_path

    # project config
    project_name = 'Model'
    device = torch.device('cuda:0')
    img_size = args.img_size
    lr = args.lr
    lr_de = args.lr_de
    epochs = args.epochs
    batch_size = args.bs
    group_size = args.gs
    log_interval = args.log_interval
    val_interval = args.val_interval

    # create log dir
    log_root = './logs'
    if not os.path.exists(log_root):
        os.makedirs(log_root)

    # create log txt
    log_txt_file = os.path.join(log_root, project_name + '_log.txt')
    custom_print(project_name, log_txt_file, 'w')

    # create model save dir
    models_root = './models'

    if not os.path.exists(models_root):
        os.makedirs(models_root)

    models_train_last = os.path.join(models_root, project_name + '_last.pth')
    models_train_3kbest = os.path.join(models_root, project_name + '3k_best.pth')
    models_train_2015best = os.path.join(models_root, project_name + '2015_best.pth')
    models_train_cobest = os.path.join(models_root, project_name + 'co_best.pth')
    net = build_model(device).to(device)
    net.train()
    net.apply(weights_init)
    net.load_pre(pretrain_path)

    q = queue.Queue(maxsize=40)

    p1 = threading.Thread(target=train_data_producer, args=(coco_item, train_datapath, npy, q, batch_size, group_size, img_size))
    p2 = threading.Thread(target=train_data_producer, args=(coco_item, train_datapath, npy, q, batch_size, group_size, img_size))
    p3 = threading.Thread(target=train_data_producer, args=(coco_item, train_datapath, npy, q, batch_size, group_size, img_size))
    p1.start()
    p2.start()
    p3.start()
    time.sleep(2)

    train(net, device, q, log_txt_file, val_datapath, models_train_3kbest,models_train_2015best,models_train_cobest, models_train_last, lr, lr_de, epochs, log_interval, val_interval)
