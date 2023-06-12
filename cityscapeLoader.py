import os
from torchvision.datasets import Cityscapes 
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils import data
from PIL import Image
import torchvision.transforms as ts

class CityscapeLoader(data.Dataset):
    def __init__(self, root, split = 'train', mode = 'fine', target_type = 'semantic',
                 resize = (768, 768), norm = False,
                 seg = False) -> None:
        super().__init__()
        
        self.dataset = Cityscapes(root= root, split= split, mode= mode, target_type= target_type)
        
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
              ts.RandomCrop(resize),
              # ts.CenterCrop(resize),
              # ts.ToTensor()
          ])
        else:
          self.ts = ts.Compose([
              ts.ToPILImage(),
              ts.CenterCrop(resize),
              # ts.ToTensor()
          ])
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        
        # Get the raw pics and corresponding labels
        img, label = self.dataset[index]
        # print(type(img), type(label))
        img, label = np.array(img, dtype= np.uint8), np.array(label, dtype= np.uint8)
        cat = np.dstack((img,label))
        # print(cat.shape)
        # seed = np.random.randint(2147483647)
        # random.seed(seed)
        cat = self.ts(cat)
        cat = np.array(cat, dtype= np.float64)
        # print(cat.shape)
        img, label = cat[:,:,:3], cat[:,:,-1]
        # print(type(img), type(label))
        # img, label = np.array(img, dtype= np.float64), np.array(label, dtype= np.float64)
        
        img, label = img.transpose(2, 0, 1), self.encoder_label(self, label).transpose(2, 0, 1) # C, H, W
        
        label = self.decoder_label(self, label)
        label = self.colourful(self, label).transpose(2, 0, 1)
        # Transformer + convert tensor!
        img, label = torch.tensor(img), torch.tensor(label)
        img, label = img / 127.5 - 1, label / 127.5 - 1
        
        return img,label
        
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
    path = os.path.join("D:\Systems\Desktop\ECE285 Visual Learning\Final_project", 'datasets', 'cityscape')
    cityset = CityscapeLoader(root= path)
    cityloader = data.DataLoader(cityset, batch_size= 1, shuffle= True)
    
    for i, batch in tqdm(enumerate(cityloader)):
        img,label = batch
        print(img.shape, label.shape)
        img = img.numpy()
        img = np.squeeze(img, axis= 0).transpose(1,2,0)
        label = label.numpy()
        label = np.squeeze(label, axis= 0)
        
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        plt.figure()
        plt.imshow(img)
        plt.show()
        c = cityset.decoder_label(cityset, label)
        plt.figure()
        plt.imshow(cityset.colourful(cityset, c))
        plt.show()
        break