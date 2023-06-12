import argparse
import os
import time
import torch
import torch.nn as nn
import logging
# from utils.getTime import get_time_str
# from utils.mIOUS import mIOU
from torch.utils.tensorboard import SummaryWriter
# from cityscapeLoader import CityscapeLoader
# from gta5Loader import GTA5Loader
from torch.utils.data import DataLoader
# from model.Unet import Unet
# from model.Discriminator import Discriminator
import numpy as np
from tqdm import tqdm
import warnings
from PIL import Image

def online_eval(valset, valloader, model, writer, device, save_epoch_interval, epoch, flag = False):
    warnings.filterwarnings('ignore')
    model.eval()
    metric, num = 0, 0
    for i,batch in tqdm(enumerate(valloader, 0), unit="batch", total=len(valloader)):
        
        img, label = batch
        img, label = img.float().to(device), label.float().to(device)
        # print(type(img), type(label))
        # print(img.shape, label.shape)


        # pred = model(label)
        pred = model(img)


        # sf = nn.Sigmoid()
        # pred = sf(pred)

        img = img.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        label = label.detach().cpu().numpy()

        img, label, pred = (img + 1) *127.5, (label + 1) * 127.5, (pred + 1) * 127.5
        img, label, pred = np.round(img).astype(np.uint8), np.round(label).astype(np.uint8), np.round(pred).astype(np.uint8)
        
        # n = pred.shape[0]
        # num += n
        
        # for j in range(n):
        #     metric += mIOU(pred= valset.decoder_label(valset, pred[j,:,:,:]), 
        #                    label= valset.decoder_label(valset, label[j,:,:,:]))

        
        # if i == 0 or i == 1:
            # img = img.detach().cpu().numpy().astype('uint8')
            # decoder_preds = np.zeros(shape=(pred.shape[0], pred.shape[2], pred.shape[3], 3))
            # decoder_labels = np.zeros(shape=(label.shape[0], label.shape[2], label.shape[3], 3))
            # for j in range(n):
            #     tmp = valset.decoder_label(valset, pred[j,:,:,:])
            #     decoder_preds[j,:,:,:] = valset.colourful(valset, tmp)

            #     tmp1 = valset.decoder_label(valset, label[j,:,:,:])
            #     decoder_labels[j,:,:,:] = valset.colourful(valset, tmp1)
            # decoder_preds = decoder_preds.astype('uint8')
            # decoder_labels = decoder_labels.astype('uint8')
        if flag == False:
          writer.add_images(tag = "gt", img_tensor = img, global_step = epoch, dataformats = "NCHW")
          # writer.add_images(tag = "input", img_tensor = label, global_step = epoch, dataformats = "NCHW")
          writer.add_images(tag = "pred", img_tensor = pred, global_step = epoch, dataformats = "NCHW")
          # pred_img = pred[0].transpose(1, 2, 0)
          # pred_img = Image.fromarray(pred_img).convert('RGB')
        else:
          writer.add_images(tag = "gt1", img_tensor = img, global_step = epoch, dataformats = "NCHW")
          # writer.add_images(tag = "input", img_tensor = label, global_step = epoch, dataformats = "NCHW")
          writer.add_images(tag = "pred1", img_tensor = pred, global_step = epoch, dataformats = "NCHW")

        # break
        if i == 1:
          break  
    # writer.add_scalar(tag = "val mIOU", scalar_value = metric / num, global_step = epoch)      
    # print("Eval for epoch: {}, Use {} images for evaluation, mIOU is {}".format(epoch, num, metric / num))
    # return metric / num