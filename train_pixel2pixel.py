import torch
import torch.nn as nn
import numpy as np
from cityscapeLoader import CityscapeLoader
from gta5Loader import GTA5Loader
from torch.utils.data import DataLoader
from model.Unet import Unet
from model.Discriminator import Discriminator
from utils.eval2 import online_eval
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from PIL import Image

def save(path, g, d, optimizer_G, optimizer_D , best_metric, lr_scheduler = None):
  ckpt = {
      'generator' : g.state_dict(),
      'discriminator' : d.state_dict(),
      'optimizer_G' : optimizer_G.state_dict(),
      'optimizer_D' : optimizer_D.state_dict(),
      'best_metric' : best_metric
  }
  if lr_scheduler:
    ckpt['lr_scheduler'] = lr_scheduler.state_dict()
  torch.save(ckpt, path)
  print("save {} successfully".format(path))
  
def train_model(
        trainset,
        valset,
        writer,  # tensorboard object
        device = torch.device,
        epochs=5,
        batch_size=1,
        lr=0.00001,
        load_exist=False,
        load_path = 'checkpoints/',
        checkpoint_interval=1,
        checkpoint_dir='checkpoints/',
        best_dir = 'checkpoints/',
        val_interval=1,
        lambda_ratio = 100,
        discriminator_size = 96
):  
    
    print("Start training on {}".format(device))
    trainloader = DataLoader(dataset= trainset, batch_size= batch_size, shuffle= True, num_workers = 2, drop_last=False)
    valloader = DataLoader(dataset= valset, batch_size= 1, shuffle= True, num_workers = 2, drop_last=False)
    
    generator = Unet(channel_list = [3, 64, 128, 256, 512, 512, 512, 512, 1024], num_class = 3, dropout = 0.1).float()

    discriminator = Discriminator(channel_list= [6, 64, 128, 256], dropout = 0.5).float()
    
    generator = torch.compile(generator)
    discriminator = torch.compile(discriminator)
    
    # gan_loss = torch.nn.MSELoss()
    gan_loss = torch.nn.BCEWithLogitsLoss()
    l1_loss = torch.nn.L1Loss()
    
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    G_scheduler = torch.optim.lr_scheduler.scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer_G,base_lr=1e-4,max_lr=1e-4,step_size_up=2500,mode="triangular2",cycle_momentum=False)
    D_scheduler = torch.optim.lr_scheduler.scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer_D,base_lr=1e-4,max_lr=1e-4,step_size_up=2500,mode="triangular2",cycle_momentum=False)

    best_metric = -1
    iteration = 0
    if load_exist:
      ckpt = torch.load(load_path, map_location='cpu')
      generator.load_state_dict(ckpt['generator'])
      discriminator.load_state_dict(ckpt['discriminator'])
      optimizer_G.load_state_dict(ckpt['optimizer_G'])
      optimizer_D.load_state_dict(ckpt['optimizer_D'])

      # Adam only!
      for key in optimizer_G.state.items():
        key[1]['exp_avg'] = key[1]['exp_avg'].to(device)
        key[1]['exp_avg_sq'] = key[1]['exp_avg_sq'].to(device)
        # key[1]['max_exp_avg_sq'] = key[1]['max_exp_avg_sq'].to(device)

      for key in optimizer_D.state.items():
        key[1]['exp_avg'] = key[1]['exp_avg'].to(device)
        key[1]['exp_avg_sq'] = key[1]['exp_avg_sq'].to(device)
        # key[1]['max_exp_avg_sq'] = key[1]['max_exp_avg_sq'].to(device)
      best_metric = ckpt['best_metric']

      print("Init training from {} ! Now the best metric is {}".format(load_path, best_metric))

    
    generator.to(device)
    discriminator.to(device)

    
    for epoch in range(epochs):
        with tqdm(unit = "batch", total=len(trainloader)) as pbar:
        # for i,batch in tqdm(enumerate(trainloader, 0), unit="batch", total=len(trainloader)):
          for i, batch in enumerate(trainloader):
              generator.train()
              discriminator.train()
              img, label = batch #normalized to -1 to 1
              img, label = img.float().to(device), label.float().to(device)
              # print(type(img), type(label))
              # If we want to do from label to image, just change here!
              # real_A, real_B = img, label
              real_A, real_B = label, img
              
              # label for discriminator
              real = torch.ones(size= (real_B.size(0), 1, discriminator_size, discriminator_size), device= device)
              fake = torch.zeros(size= (real_B.size(0), 1, discriminator_size, discriminator_size), device= device)
              # print(real.size, fake.size, img.size, label.size)
              # G part
              optimizer_G.zero_grad()
              fake_B = generator(real_A) # From A to B
              
              discriminator_input = torch.cat((fake_B, real_A), dim = 1)
              discriminator_res = discriminator(discriminator_input)
              
              loss_gan = gan_loss(discriminator_res, real)
              loss_l1 = l1_loss(fake_B, real_B)
              
              loss_G = loss_gan + lambda_ratio * loss_l1
              
              loss_G.backward()
              optimizer_G.step()
              G_scheduler.step()
              lr_G = optimizer_G.state_dict()['param_groups'][0]['lr']
              writer.add_scalar(tag = "lr_G", scalar_value= lr_G, global_step= iteration)
              # G part over
              
              # D part
              optimizer_D.zero_grad()
              
              discriminator_input = torch.cat((real_B, real_A), dim = 1)
              pred_real = discriminator(discriminator_input)
              loss_true = gan_loss(pred_real, real)
              
              discriminator_input = torch.cat((fake_B.detach(), real_A), dim = 1)
              pred_fake = discriminator(discriminator_input)
              loss_false = gan_loss(pred_fake, fake)
              
              loss_D = 0.5 * loss_true + 0.5 * loss_false
              
              loss_D.backward()
              optimizer_D.step()
              D_scheduler.step()
              lr_D = optimizer_D.state_dict()['param_groups'][0]['lr']
              writer.add_scalar(tag = "lr_D", scalar_value= lr_D, global_step= iteration)
              pbar.set_postfix(loss_G=loss_G.item(), loss_D = loss_D.item())
              pbar.update(1)
              writer.add_scalar(tag = "loss_G", scalar_value = loss_G.item(), global_step = iteration)
              writer.add_scalar(tag = "loss_D", scalar_value = loss_D.item(), global_step = iteration)
              iteration += 1
              # D part over
              # print("test")
              # if i == 10:
              #    break
        
        print("Finish epoch: {}, the loss_G: {}, the loss_D:{}".format(epoch, loss_G.item(), loss_D.item()))
        print("Evaluating...")
        online_eval(valset = valset, valloader= valloader, model= generator, writer= writer, device= device, save_epoch_interval= 1, epoch= epoch)
        save(path = checkpoint_dir, g = generator, d = discriminator, optimizer_G = optimizer_G, optimizer_D = optimizer_D, best_metric = best_metric)
        # if eval_res > best_metric:
        #   best_metric = eval_res
        #   save(path = best_dir, g = generator, d = discriminator, optimizer_G = optimizer_G, optimizer_D = optimizer_D, best_metric = best_metric)

def check_result(
        writer,
        valset,
        device = torch.device,
        num = 100,
        load_exist = True,
        load_path = "checkpoints/",
):  
    
    print("Start training on {}".format(device))
    

    
    valloader = DataLoader(dataset= valset, batch_size= 1, shuffle= False, num_workers = 2, drop_last=False)
    
    generator_g = Unet(channel_list = [3, 64, 128, 256, 512, 512, 512, 512, 1024], num_class = 3).float()


    
    generator_g = torch.compile(generator_g)

    best_metric = -1
    iteration = 0
    if load_exist:
      ckpt = torch.load(load_path, map_location='cpu')
      generator_g.load_state_dict(ckpt['generator'])

    
    generator_g.to(device)

    generator_g.eval()

    global_id = 0

    model = generator_g
    it = 0
    model.eval()

    with tqdm(unit = "batch", total=len(valloader)) as pbar:
      for i, batch in enumerate(valloader):

        img, label = batch
        img, label = img.float().to(device), label.float().to(device)
        # print(type(img), type(label))
        # print(img.shape, label.shape)


        # pred = model(label)
        pred = model(label)


        # sf = nn.Sigmoid()
        # pred = sf(pred)

        img = img.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        label = label.detach().cpu().numpy()

        img, label, pred = (img + 1) *127.5, (label + 1) * 127.5, (pred + 1) * 127.5
        img, label, pred = np.round(img).astype(np.uint8), np.round(label).astype(np.uint8), np.round(pred).astype(np.uint8)
        

        img, pred = img[0].transpose(1, 2, 0), pred[0].transpose(1, 2, 0)
        label = label[0].transpose(1, 2, 0)

        img = Image.fromarray(img).convert('RGB')
        pred = Image.fromarray(pred).convert('RGB')
        label = Image.fromarray(label).convert('RGB')

        # online_eval(valset = valset, valloader = valloader, model = model, writer = writer,
        #             save_epoch_interval = 1, epoch = it, device = device)
        it += 1

        path = "/content/Results_gta5_gen"
        input = str(global_id) + "_" + "input" + ".jpg"
        output = str(global_id) + "_" + "output" + ".jpg"
        seg = str(global_id) + "_" + "seg_label" + ".jpg"

        path_1 = os.path.join(path, input)
        path_2 = os.path.join(path, output)
        path_3 = os.path.join(path, seg)

        img.save(path_1)
        pred.save(path_2)
        label.save(path_3)
        global_id += 1
        pbar.update(1)

        # if i == 100:
        #   break

if __name__ == "__main__":
    path_city = os.path.join(os.getcwd(), 'data', 'cityscape')
    path_gta5 = os.path.join(os.getcwd(), "data", "GTA5")
    print(path_city, path_gta5)
    dataset = "gta5"
    if dataset == "city":
        trainset = CityscapeLoader(root= path_city, split= 'train', resize = (512, 512))
        valset = CityscapeLoader(root= path_city, split= 'val', resize = (512, 512))
    elif dataset == "gta5":
        trainset = GTA5Loader(root= path_gta5, split= 'train', resize = (512, 512))
        valset = GTA5Loader(root= path_gta5, split= 'val', resize = (512, 512))
    else:
        raise NotImplementedError
    
    lr = 0.0001
    epoch = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_name = "gta5p2p_512"
    log_dir = os.path.join(os.getcwd(), project_name, "log")
    writer = SummaryWriter(log_dir = log_dir, comment = project_name)
    save_path = os.path.join(os.getcwd(), project_name, "checkpoints", "latest.pkl")
    best_path = os.path.join(os.getcwd(), project_name, "checkpoints", "best_val.pkl")
    load_path = os.path.join(os.getcwd(), project_name, "checkpoints", "best_val.pkl")


    train_model(trainset = trainset, valset = valset, device = device, writer = writer, batch_size = 8, lr = lr, 
                checkpoint_dir= save_path, best_dir = best_path, load_exist = True, 
                load_path = "/content/drive/MyDrive/ECE285_FinalProject/GAN/gta5p2p_512/checkpoints/latest.pkl", 
                epochs = epoch, 
                discriminator_size = 64)

