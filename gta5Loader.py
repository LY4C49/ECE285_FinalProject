import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils import data
from PIL import Image
import torchvision.transforms as ts

class GTA5Loader(data.Dataset):
    def __init__(self, root, split = 'train',
                 resize = (768, 768), norm = False, ratio = [6, 3, 1],
                 seg = False) -> None:
        super().__init__()
        
        self.root = root
        self.all_dirs = os.listdir(root)
        
        self.img_path = []
        self.label_path = []
        self.num_imgs, self.num_labels = 0, 0
        for dir in self.all_dirs:
            if os.path.isdir(os.path.join(root, dir)):
                if "images" in dir:
                    self.img_path.append(dir)
                    self.num_imgs += len(os.listdir(os.path.join(root, dir, 'images')))
                if "labels" in dir:
                    self.label_path.append(dir)
                    self.num_labels += len(os.listdir(os.path.join(root, dir, 'labels')))
        
        assert self.num_imgs == self.num_labels, "Images != Labels, Data missing!"
        self.img_path.sort()
        self.label_path.sort()
        
        self.all_ids = [i for i in range(1, self.num_imgs + 1)]
        np.random.seed(1127)
        np.random.shuffle(self.all_ids)
        
        self.train_n, self.val_n, self.test_n = int(self.num_imgs * 0.6), int(self.num_imgs * 0.3), int(self.num_imgs * 0.1)
        
        self.train , self.val, self.test= self.all_ids[: self.train_n], self.all_ids[self.train_n: self.train_n + self.val_n], self.all_ids[self.train_n + self.val_n:]
        
        self.dataset = None
        if split == "train":
            self.dataset = self.train
            self.dataset = self.dataset[:10000]
        elif split == "val":
            self.dataset = self.val
            self.dataset = self.dataset[:500]
        elif split == "test":
            self.dataset = self.test

        self.colors = [
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [0, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32],
        ]
        
        self.label_color = dict(zip(range(len(self.colors)), self.colors))
        
        self.background = [-1, 0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30]
        self.class_id = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26,
                         27, 28, 31, 32, 33]
        self.all_class = len(self.background) + len(self.class_id)
        self.class_num = len(self.class_id)
        
        if split == "train":
          self.ts = ts.Compose([
              ts.ToPILImage(),
              # ts.CenterCrop(resize),
              ts.RandomCrop(resize),
              # ts.ToTensor()
          ])
        else:
          self.ts = ts.Compose([
              ts.ToPILImage(),
              ts.CenterCrop(resize),
              # ts.ToTensor()
          ])
        
        self.seg = seg

        self.bad_ids = [20819]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Get the raw pics and corresponding labels
        img_id = self.dataset[index]
        if img_id == 0:
          img_id += 1
        # if 20001<=img_id <= 22500:
        #   img_id -= 2500
        ith_dir = img_id // 2500
        if img_id % 2500 == 0 and ith_dir != 0:
          ith_dir -= 1
        suffix = str(img_id).zfill(5) + ".png"
        # print(self.img_path, img_id, ith_dir)
        real_img_path = os.path.join(self.root, self.img_path[ith_dir], "images", suffix)
        real_label_path = os.path.join(self.root, self.label_path[ith_dir], "labels", suffix)

        backup_suffix = str(5).zfill(5) + ".png"
        backup_img_path = os.path.join(self.root, self.img_path[0], "images", backup_suffix)
        backup_label_path = os.path.join(self.root, self.label_path[0], "labels", backup_suffix)


        img, label = Image.open(real_img_path), Image.open(real_label_path)
        img, label = np.array(img, dtype= np.uint8), np.array(label, dtype= np.uint8)
  
        
        # img, label = Image.open(real_img_path), Image.open(real_label_path)
        # img, label = np.array(img, dtype= np.uint8), np.array(label, dtype= np.uint8)


        if img.shape[0] != label.shape[0] or img.shape[1] != label.shape[1]:
          # print("ERROR")
          # print(real_img_path, real_label_path)
          # print(img.shape, label.shape)
          img, label = Image.open(backup_img_path), Image.open(backup_label_path)
          img, label = np.array(img, dtype= np.uint8), np.array(label, dtype= np.uint8)

        cat = np.dstack((img,label))
        # print(cat.shape)
        # seed = np.random.randint(2147483647)
        # random.seed(seed)

        cat = self.ts(cat)
        
        cat = np.array(cat, dtype= np.float64)
        # print(cat.shape)
        img, label = cat[:,:,:3], cat[:,:,-1]
        
        if self.seg == True:
          # Segmentation
          img, label = np.array(img, dtype= np.float64), np.array(label, dtype= np.float64)
          img, label = img.transpose(2, 0, 1), self.encoder_label(self,label).transpose(2, 0, 1)
          img = img / 127.5 - 1
          #tensor + random crop
          img, label = torch.tensor(img), torch.tensor(label)

        else:
          # P2P
          img, label = img.transpose(2, 0, 1), self.encoder_label(self, label).transpose(2, 0, 1) # C, H, W
          
          label = self.decoder_label(self, label)
          label = self.colourful(self, label).transpose(2, 0, 1)
          # Transformer + convert tensor!
          img, label = torch.tensor(img), torch.tensor(label)
          img, label = img / 127.5 - 1, label / 127.5 - 1
        return img, label #, real_img_path, real_label_path, backup_img_path, backup_label_path
        #print(img.shape, label.shape)
    
    @staticmethod
    def encoder_label(self, label):
        """
            Map labels to its corresponding channel
        """
        new_label = np.zeros((label.shape[0], label.shape[1], self.class_num + 1))
        
        for i in range(-1, 34):
            if i in self.background:
                new_label[:,:,0] += np.where(label == i, 1, 0)
            else:
                channel = self.class_id.index(i)
                new_label[:,:,channel + 1] += np.where(label == i , 1, 0)
        
        return new_label # H, W, C
    
    @staticmethod
    def decoder_label(self, output):
        # C, H, W
        output = output.transpose(1, 2, 0)
        output_matrix = np.argmax(output, axis=-1)
        return output_matrix
    
    @staticmethod
    def colourful(self, output_matrix):
        size = output_matrix.shape
        colorful_label = np.zeros((size[0], size[1], 3))
        r = np.zeros((size[0], size[1]))
        g = np.zeros((size[0], size[1]))
        b = np.zeros((size[0], size[1]))
        for i in range(1, len(self.colors)):
            r[output_matrix == i] = self.colors[i - 1][0]
            g[output_matrix == i] = self.colors[i - 1][1]
            b[output_matrix == i] = self.colors[i - 1][2]
        colorful_label[:,:,0] = r
        colorful_label[:,:,1] = g
        colorful_label[:,:,2] = b
        # colorful_label = Image.fromarray(colorful_label.astype('uint8')).convert('RGB')
        return colorful_label

if __name__ == "__main__":
    from tqdm import tqdm
    path_gta5 = os.path.join("D:\Systems\Desktop\ECE285 Visual Learning\Final_project", "GTA5")
    gta5set = GTA5Loader(root= path_gta5)
    gta5dataloader = data.DataLoader(gta5set, batch_size= 1, shuffle= True)
    for i, batch in tqdm(enumerate(gta5dataloader)):
        # print(len(batch))
        img,label = batch
        print(img.shape, label.shape) # torch.Size([1, 3, 768, 768]) torch.Size([1, 20, 768, 768])
        img = img.numpy()
        img = np.squeeze(img, axis= 0).transpose(1,2,0)
        label = label.numpy()
        label = np.squeeze(label, axis= 0)
        
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        plt.figure()
        plt.imshow(img)
        plt.show()
        c = gta5set.decoder_label(gta5set, label)
        plt.figure()
        plt.imshow(gta5set.colourful(gta5set, c))
        plt.show()
        break