import torch
import torch.nn as nn
import numpy as np
from cityscapeLoader import CityscapeLoader
from gta5Loader import GTA5Loader
from torch.utils.data import DataLoader
from model.resnet import GeneratorResNet
from model.Unet import Unet
from model.Discriminator import Discriminator
from utils.eval import online_eval
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import itertools
import random
import os
from PIL import Image
import matplotlib.pyplot as plt

def save(path, g, f, x, y, optimizer_g, optimizer_x , optimizer_y , best_metric, lr_scheduler = None):
  ckpt = {
      'generator_g' : g.state_dict(),
      'discriminator_x' : x.state_dict(),
      'generator_f' : f.state_dict(),
      'discriminator_y' : x.state_dict(),
      'optimizer_g' : optimizer_g.state_dict(),
      'optimizer_x' : optimizer_x.state_dict(),
      'optimizer_y' : optimizer_y.state_dict(),
      'best_metric' : best_metric
  }
  if lr_scheduler:
    ckpt['lr_scheduler'] = lr_scheduler.state_dict()
  torch.save(ckpt, path)
  print("save {} successfully".format(path))
  
def get_loss_d(loss_fn, real_img, generated_img):
    real = torch.ones_like(real_img, device = device)
    fake = torch.zeros_like(generated_img, device = device)
    loss_real = loss_fn(real, real_img)
    loss_fake = loss_fn(fake, generated_img)
    return 0.5 * (loss_real + loss_fake)

def get_loss_g(loss_fn, generated_img):
  real = torch.ones_like(generated_img, device = device)
  return loss_fn(real, generated_img)

def get_cycle_loss(loss_fn, real_img, cycled_img, alpha):
  # return alpha * torch.mean(torch.abs(real_img - cycled_img))
  return alpha * loss_fn(real_img, cycled_img)

def get_identity_loss(loss_fn, real_image, same_image, alpha):
  # return alpha * torch.mean(torch.abs(real_image - same_image))
  return alpha * loss_fn(real_image, same_image)


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
    trainset_1, trainset_2 = trainset
    valset_1, valset_2 = valset

    trainloader_1 = DataLoader(dataset= trainset_1, batch_size= batch_size, shuffle= True, num_workers = 2, drop_last=True)
    valloader_1 = DataLoader(dataset= valset_1, batch_size= 1, shuffle= False, num_workers = 2, drop_last=False)

    trainloader_2 = DataLoader(dataset= trainset_2, batch_size= batch_size, shuffle= True, num_workers = 2, drop_last=True)
    valloader_2 = DataLoader(dataset= valset_2, batch_size= 1, shuffle= False, num_workers = 2, drop_last=False)
    
    # generator_g = Unet(channel_list = [3, 64, 128, 256, 512, 512, 512, 512, 1024], num_class = 3, dropout = 0.0).float()
    # generator_f = Unet(channel_list = [3, 64, 128, 256, 512, 512, 512, 512, 1024], num_class = 3, dropout = 0.0).float()

    generator_g = GeneratorResNet(input_shape= (3, 512, 512), num_residual_blocks= 9).float()
    generator_f = GeneratorResNet(input_shape= (3, 512, 512), num_residual_blocks= 9).float()

    discriminator_x = Discriminator(channel_list= [3, 64, 128, 256]).float()
    discriminator_y = Discriminator(channel_list= [3, 64, 128, 256]).float()
    
    generator_g = torch.compile(generator_g)
    generator_f = torch.compile(generator_f)
    discriminator_x = torch.compile(discriminator_x)
    discriminator_y = torch.compile(discriminator_y)
    
    # gan_loss = torch.nn.MSELoss()
    gan_loss = torch.nn.MSELoss()
    # cycle_loss = torch.nn.L1Loss()
    # identity_loss = torch.nn.L1Loss()
    l1_loss = torch.nn.L1Loss()
    
    # optimizer_g = torch.optim.Adam(generator_g.parameters(), lr=lr, betas=(0.5, 0.999))
   

    optimizer_Gen = torch.optim.Adam(itertools.chain(generator_g.parameters(), generator_f.parameters()), lr=lr, betas=(0.5, 0.999))
    # optimizer_f = torch.optim.Adam(generator_f.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_y = torch.optim.Adam(discriminator_y.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_x = torch.optim.Adam(discriminator_x.parameters(), lr=lr, betas=(0.5, 0.999))
    
    Gen_scheduler = torch.optim.lr_scheduler.scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer_Gen,base_lr=2e-4,max_lr=2e-4,step_size_up=2500,mode="triangular2",cycle_momentum=False)
    x_scheduler = torch.optim.lr_scheduler.scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer_x,base_lr=2e-4,max_lr=1e-4,step_size_up=2500,mode="triangular2",cycle_momentum=False)
    # f_scheduler = torch.optim.lr_scheduler.scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer_f,base_lr=2e-4,max_lr=2e-4,step_size_up=2500,mode="triangular2",cycle_momentum=False)
    y_scheduler = torch.optim.lr_scheduler.scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer_y,base_lr=2e-4,max_lr=1e-4,step_size_up=2500,mode="triangular2",cycle_momentum=False)

    fake_x_buffer = ReplayBuffer()
    fake_y_buffer = ReplayBuffer()


    best_metric = -1
    iteration = 0
    if load_exist:
      ckpt = torch.load(load_path, map_location='cpu')
      generator_g.load_state_dict(ckpt['generator_g'])
      discriminator_x.load_state_dict(ckpt['discriminator_x'])
      generator_f.load_state_dict(ckpt['generator_f'])
      discriminator_y.load_state_dict(ckpt['discriminator_y'])

      # optimizer_g.load_state_dict(ckpt['optimizer_g'])
      optimizer_x.load_state_dict(ckpt['optimizer_x'])
      optimizer_Gen.load_state_dict(ckpt['optimizer_g'])
      # optimizer_f.load_state_dict(ckpt['optimizer_f'])
      optimizer_y.load_state_dict(ckpt['optimizer_y'])

      # Adam only!
      for key in optimizer_Gen.state.items():
        key[1]['exp_avg'] = key[1]['exp_avg'].to(device)
        key[1]['exp_avg_sq'] = key[1]['exp_avg_sq'].to(device)
        # key[1]['max_exp_avg_sq'] = key[1]['max_exp_avg_sq'].to(device)

      for key in optimizer_x.state.items():
        key[1]['exp_avg'] = key[1]['exp_avg'].to(device)
        key[1]['exp_avg_sq'] = key[1]['exp_avg_sq'].to(device)
        # key[1]['max_exp_avg_sq'] = key[1]['max_exp_avg_sq'].to(device)
      best_metric = ckpt['best_metric']

      # for key in optimizer_f.state.items():
      #   key[1]['exp_avg'] = key[1]['exp_avg'].to(device)
      #   key[1]['exp_avg_sq'] = key[1]['exp_avg_sq'].to(device)
      #   # key[1]['max_exp_avg_sq'] = key[1]['max_exp_avg_sq'].to(device)

      for key in optimizer_y.state.items():
        key[1]['exp_avg'] = key[1]['exp_avg'].to(device)
        key[1]['exp_avg_sq'] = key[1]['exp_avg_sq'].to(device)
        # key[1]['max_exp_avg_sq'] = key[1]['max_exp_avg_sq'].to(device)
      best_metric = ckpt['best_metric']

      print("Init training from {} ! Now the best metric is {}".format(load_path, best_metric))

    
    generator_g.to(device)
    discriminator_x.to(device)
    generator_f.to(device)
    discriminator_y.to(device)

    
    for epoch in range(epochs):
        with tqdm(unit = "batch", total=len(trainloader_1)) as pbar:
        # for i,batch in tqdm(enumerate(trainloader, 0), unit="batch", total=len(trainloader)):
          for i, batch in enumerate(zip(trainloader_1, trainloader_2)):
              generator_f.train()
              generator_g.train()
              discriminator_x.train()
              discriminator_y.train()

              real_x, _ = batch[0] #normalized to -1 to 1
              real_y, _ = batch[1]
              real_x = real_x.float().to(device)
              real_y = real_y.float().to(device)
              #img, label = img.float().to(device), label.float().to(device)
              # print(type(img), type(label))
              # If we want to do from label to image, just change here!
              # real_A, real_B = img, label
              # real_A, real_B = label, img

              # Generators
              # G : X->Y; F : Y->X
              optimizer_Gen.zero_grad()
              fake_y = generator_g(real_x)
              cycled_x = generator_f(fake_y)

              fake_x = generator_f(real_y)
              cycled_y = generator_g(fake_x)

              same_x = generator_f(real_x)
              same_y = generator_g(real_y)

              g_id_loss =  get_identity_loss(l1_loss, real_y, same_y, 5)
              f_id_loss =  get_identity_loss(l1_loss, real_x, same_x, 5)

              tmp_fake_y = discriminator_y(fake_y)
              tmp_fake_x = discriminator_x(fake_x)

              g_gan_loss = get_loss_g(gan_loss, tmp_fake_y)
              f_gan_loss = get_loss_g(gan_loss, tmp_fake_x)

              identity_loss = (g_id_loss + f_id_loss) / 2
              generators_loss = (g_gan_loss + f_gan_loss) / 2
              cycle_loss = (get_cycle_loss(l1_loss, real_x, cycled_x, 10) + get_cycle_loss(l1_loss, real_y, cycled_y, 10)) / 2

              total_loss_gen = generators_loss + cycle_loss + identity_loss

              total_loss_gen.backward()
              optimizer_Gen.step()
              #Generators over

              # Total loss for optimizer
              # total_g_loss = g_gan_loss + cycle_loss + identity_loss(l1_loss, real_y, same_y, 5)
              # total_f_loss = f_gan_loss + cycle_loss + identity_loss(l1_loss, real_x, same_x, 5)

              # Discriminator
              dis_real_x = discriminator_x(real_x)
              dis_real_y = discriminator_y(real_y)

              fake_x_ = fake_x_buffer.push_and_pop(fake_x)
              dis_fake_x = discriminator_x(fake_x_.detach())

              fake_y_ = fake_y_buffer.push_and_pop(fake_y)
              dis_fake_y = discriminator_y(fake_y_.detach())

              optimizer_x.zero_grad()
              dis_x_loss = get_loss_d(gan_loss, dis_real_x, dis_fake_x)
              dis_x_loss.backward()
              optimizer_x.step()


              optimizer_y.zero_grad()
              dis_y_loss = get_loss_d(gan_loss, dis_real_y, dis_fake_y)
              dis_y_loss.backward()
              optimizer_y.step()

              # total_g_loss.backward(retain_graph=True)
              # total_f_loss.backward(retain_graph=True)
              # dis_x_loss.backward(retain_graph=True)
              # dis_y_loss.backward(retain_graph=True)

              # total_g_loss.backward()
              # total_f_loss.backward()

              # optimizer_g.step()
              # optimizer_f.step()
              
              
              Gen_scheduler.step()
              # g_scheduler.step()
              # f_scheduler.step()
              x_scheduler.step()
              y_scheduler.step()

              lr_g = optimizer_Gen.state_dict()['param_groups'][0]['lr']
              writer.add_scalar(tag = "lr_D", scalar_value= lr_g, global_step= iteration)
              pbar.set_postfix(loss_G = total_loss_gen.item(), loss_x = dis_x_loss.item(), loss_y = dis_y_loss.item())
              pbar.update(1)
              writer.add_scalar(tag = "loss_G", scalar_value = total_loss_gen.item(), global_step = iteration)
              # writer.add_scalar(tag = "loss_f", scalar_value = total_f_loss.item(), global_step = iteration)
              writer.add_scalar(tag = "loss_x", scalar_value = dis_x_loss.item(), global_step = iteration)
              writer.add_scalar(tag = "loss_y", scalar_value = dis_y_loss.item(), global_step = iteration)
              iteration += 1
              # D part over
              # print("test")
              # if i == 5:
              #    break
        
        print("Finish epoch: {}, the loss_G: {}, the loss_x:{}".format(epoch, total_loss_gen.item(), dis_x_loss.item()))
        print("Evaluating...")
        online_eval(valset = valset_1, valloader= valloader_1, model= generator_g, writer= writer, device= device, save_epoch_interval= 1, epoch= epoch)
        online_eval(valset = valset_2, valloader= valloader_2, model= generator_f, writer= writer, device= device, save_epoch_interval= 1, epoch= epoch, flag = True)
        save(path = checkpoint_dir, g = generator_g, f = generator_f, x = discriminator_x, y = discriminator_y, 
             optimizer_g = optimizer_Gen, optimizer_x = optimizer_x, optimizer_y = optimizer_y, best_metric = best_metric)
        # if eval_res > best_metric:
        #   best_metric = eval_res
        #   save(path = best_dir, g = generator, d = discriminator, optimizer_G = optimizer_G, optimizer_D = optimizer_D, best_metric = best_metric)
        
def check_result(
        trainset,
        valset,
        device = torch.device,
        num = 100,
        load_exist = True,
        load_path = "checkpoints/"
):  
    
    print("Start training on {}".format(device))
    trainset_1, trainset_2 = trainset
    valset_1, valset_2 = valset

    trainloader_1 = DataLoader(dataset= trainset_1, batch_size= 1, shuffle= True, num_workers = 2, drop_last=True)
    valloader_1 = DataLoader(dataset= valset_1, batch_size= 1, shuffle= False, num_workers = 2, drop_last=False)

    trainloader_2 = DataLoader(dataset= trainset_2, batch_size= 1, shuffle= True, num_workers = 2, drop_last=True)
    valloader_2 = DataLoader(dataset= valset_2, batch_size= 1, shuffle= False, num_workers = 2, drop_last=False)
    
    generator_g = GeneratorResNet(input_shape= (3, 256, 256), num_residual_blocks= 9).float()
    generator_f = GeneratorResNet(input_shape= (3, 256, 256), num_residual_blocks= 9).float()

    discriminator_x = Discriminator(channel_list= [3, 64, 128, 256]).float()
    discriminator_y = Discriminator(channel_list= [3, 64, 128, 256]).float()
    
    generator_g = torch.compile(generator_g)
    generator_f = torch.compile(generator_f)
    discriminator_x = torch.compile(discriminator_x)
    discriminator_y = torch.compile(discriminator_y)

    best_metric = -1
    iteration = 0
    if load_exist:
      ckpt = torch.load(load_path, map_location='cpu')
      generator_g.load_state_dict(ckpt['generator_g'])
      discriminator_x.load_state_dict(ckpt['discriminator_x'])
      generator_f.load_state_dict(ckpt['generator_f'])
      discriminator_y.load_state_dict(ckpt['discriminator_y'])

    
    generator_g.to(device)
    discriminator_x.to(device)
    generator_f.to(device)
    discriminator_y.to(device)

    generator_f.eval()
    generator_g.eval()
    discriminator_x.eval()
    discriminator_y.eval()


    global_id = 0

    model = generator_g

    with tqdm(unit = "batch", total=len(valloader_1)) as pbar:
      for i, batch in enumerate(valloader_1):

        img, label = batch
        img, label = img.float().to(device), label.float().to(device)

        pred = model(img)

        img = img.detach().cpu().numpy()[0, :, :, :].transpose(1, 2, 0)
        pred = pred.detach().cpu().numpy()[0, :, :, :].transpose(1, 2, 0)
        label = label.detach().cpu().numpy()[0, :, :, :].transpose(1, 2, 0)

        img, label, pred = (img + 1) *127.5, (label + 1) * 127.5, (pred + 1) * 127.5
        img, label, pred = np.round(img).astype(np.uint8), np.round(label).astype(np.uint8), np.round(pred).astype(np.uint8)

        img = Image.fromarray(img).convert('RGB')
        pred = Image.fromarray(pred).convert('RGB')

        path = "/content/Results_ctg"
        input = str(global_id) + "_" + "input" + ".jpg"
        output = str(global_id) + "_" + "output" + ".jpg"

        path_1 = os.path.join(path, input)
        path_2 = os.path.join(path, output)

        img.save(path_1)
        pred.save(path_2)
        global_id += 1
        pbar.update(1)

        # if i == 100:
        #   break



if __name__ == "__main__":
    path_city = os.path.join(os.getcwd(), 'data', 'cityscape')
    path_gta5 = os.path.join(os.getcwd(), "data", "GTA5")
    print(path_city, path_gta5)


    trainset1 = CityscapeLoader(root= path_city, split= 'train', resize = (512, 512))
    valset1 = CityscapeLoader(root= path_city, split= 'val', resize = (512, 512))

    trainset2 = GTA5Loader(root= path_gta5, split= 'train', resize = (512, 512))
    valset2 = GTA5Loader(root= path_gta5, split= 'val', resize = (512, 512))
    
    lr = 0.0001
    epoch = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_name = "cycle_3"
    log_dir = os.path.join(os.getcwd(), project_name, "log")
    writer = SummaryWriter(log_dir = log_dir, comment = project_name)
    save_path = os.path.join(os.getcwd(), project_name, "checkpoints_2", "latest.pkl")
    best_path = os.path.join(os.getcwd(), project_name, "checkpoints_2", "best_val.pkl")
    load_path = os.path.join(os.getcwd(), project_name, "checkpoints", "best_val.pkl")

    trainset = (trainset1, trainset2)
    valset = (valset1, valset2)

    train_model(trainset = trainset, valset = valset, device = device, writer = writer, batch_size = 2, lr = lr, 
                checkpoint_dir= save_path, best_dir = best_path, load_exist = True, 
                load_path = "/content/drive/MyDrive/ECE285_FinalProject/GAN/cycle_3/checkpoints/latest.pkl", 
                epochs = epoch)
        
    
