import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding = 'same', apply_bn = True, dropout = 0.0) -> None:
        super(BasicConv, self).__init__()
        
        self.conv = nn.Conv2d(in_channels= in_channel, out_channels= out_channel, kernel_size= kernel_size, padding= padding, stride= stride)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace= True)
        self.apply_bn = apply_bn
        self.dropout = nn.Dropout2d(dropout)
    def forward(self, x):
        x = self.conv(x)
        if self.apply_bn: 
            x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding = 'same', dropout = 0.0) -> None:
        super(EncoderLayer, self).__init__()
        
        self.conv1 = BasicConv(in_channel= in_channel, out_channel= out_channel, kernel_size= kernel_size, stride= stride, padding = padding)
        self.dropout = nn.Dropout2d(dropout)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        return x

class UpConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding = 'same', dropout = 0.0, apply_bn = True) -> None:
        super(UpConv, self).__init__()
        
        self.conv = nn.ConvTranspose2d(in_channels= in_channel, out_channels= out_channel, kernel_size=  kernel_size, stride= stride, padding= padding)
        self.bn = nn.BatchNorm2d(num_features= out_channel)
        self.relu = nn.LeakyReLU( negative_slope= 0.2, inplace= True)
        self.dropout = nn.Dropout2d(p = dropout)
        self.apply_bn = apply_bn
    def forward(self, x):
        x = self.conv(x)
        if self.apply_bn:
            x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding = 'same', dropout = 0.0, apply_bn = True) -> None:
        super(DecoderLayer, self).__init__()
        
        self.up = UpConv(in_channel= in_channel, out_channel= out_channel, kernel_size= kernel_size, stride= stride, padding= padding, dropout= dropout, apply_bn= apply_bn)
        self.dropout = nn.Dropout2d(dropout)
    
    def forward(self, up_input, skip_input):
        
        x = torch.cat((up_input, skip_input), dim = 1)
        x = self.up(x)
        # x = self.dropout(x)
        
        return x
        

class Unet(nn.Module):
    def __init__(self, channel_list:list = [3, 64, 128, 256, 512, 512, 512, 512, 1024], num_class = 3, dropout = 0.0) -> None:
        super(Unet, self).__init__()
        
        self.encoder1 = EncoderLayer(in_channel = channel_list[0], out_channel= channel_list[1], kernel_size = 4, stride = 2, padding= 1, dropout = dropout)
        
        self.encoder2 = EncoderLayer(in_channel = channel_list[1], out_channel= channel_list[2], kernel_size = 4, stride = 2, padding= 1, dropout = dropout)
        
        self.encoder3 = EncoderLayer(in_channel = channel_list[2], out_channel= channel_list[3], kernel_size = 4, stride = 2, padding= 1, dropout = dropout)
        
        self.encoder4 = EncoderLayer(in_channel = channel_list[3], out_channel= channel_list[4], kernel_size = 4, stride = 2, padding= 1, dropout = dropout)
        
        self.encoder5 = EncoderLayer(in_channel = channel_list[4], out_channel= channel_list[5], kernel_size = 4, stride = 2, padding= 1, dropout = dropout)
        
        self.encoder6 = EncoderLayer(in_channel = channel_list[5], out_channel= channel_list[6], kernel_size = 4, stride = 2, padding= 1, dropout = dropout)
        
        self.encoder7 = EncoderLayer(in_channel = channel_list[6], out_channel= channel_list[7], kernel_size = 4, stride = 2, padding= 1, dropout = dropout)
        
        self.encoder8 = EncoderLayer(in_channel = channel_list[7], out_channel= channel_list[8], kernel_size = 4, stride = 2, padding= 1, dropout = dropout)
        
        self.decoder8 = UpConv(in_channel = channel_list[-1], out_channel = channel_list[-2], kernel_size = 4, stride = 2, padding= 1, dropout = dropout)
        
        self.decoder7 = DecoderLayer(in_channel = channel_list[-2] * 2, out_channel = channel_list[-3], kernel_size = 4, stride = 2, padding= 1, dropout = dropout)
        
        self.decoder6 = DecoderLayer(in_channel = channel_list[-3] * 2, out_channel = channel_list[-4], kernel_size = 4, stride = 2, padding= 1, dropout = dropout)
        
        self.decoder5 = DecoderLayer(in_channel = channel_list[-4] * 2, out_channel = channel_list[-5], kernel_size = 4, stride = 2, padding= 1, dropout = dropout)
        
        self.decoder4 = DecoderLayer(in_channel = channel_list[-5] * 2, out_channel = channel_list[-6], kernel_size = 4, stride = 2, padding= 1, dropout = dropout)
        
        self.decoder3 = DecoderLayer(in_channel = channel_list[-6] * 2, out_channel = channel_list[-7], kernel_size = 4, stride = 2, padding= 1, dropout = dropout)

        self.decoder2 = DecoderLayer(in_channel = channel_list[-7] * 2, out_channel = channel_list[-8], kernel_size = 4, stride = 2, padding= 1, dropout = dropout)
        
        self.decoder1 = DecoderLayer(in_channel = channel_list[-8] * 2, out_channel = channel_list[-8], kernel_size = 4, stride = 2, padding= 1, dropout = dropout)
        
        self.post_process = nn.Sequential(
                nn.Conv2d(in_channels = channel_list[-8], out_channels = num_class, kernel_size = 1, stride = 1, padding = 0),
                nn.Tanh()
            )

    def forward(self, x):
        encoder1 = self.encoder1(x)
        #print(encoder1.shape)
        encoder2 = self.encoder2(encoder1)
        #print(encoder2.shape)
        encoder3 = self.encoder3(encoder2)
        #print(encoder3.shape)
        encoder4 = self.encoder4(encoder3)
        #print(encoder4.shape)
        encoder5 = self.encoder5(encoder4)
        #print(encoder5.shape)
        encoder6 = self.encoder6(encoder5)
        #print(encoder6.shape)
        encoder7 = self.encoder7(encoder6)
        #print(encoder7.shape)
        encoder8 = self.encoder8(encoder7)
        #print(encoder8.shape)
        
        decoder8 = self.decoder8(encoder8)
        #print(decoder8.shape)
        
        decoder7 = self.decoder7(decoder8, encoder7)
        #print(decoder7.shape)
        
        decoder6 = self.decoder6(decoder7, encoder6)
        #print(decoder6.shape)
        
        decoder5 = self.decoder5(decoder6, encoder5)
        #print(decoder5.shape)
        
        decoder4 = self.decoder4(decoder5, encoder4)
        #print(decoder4.shape)
        
        decoder3 = self.decoder3(decoder4, encoder3)
        #print(decoder3.shape)
        
        decoder2 = self.decoder2(decoder3, encoder2)
        #print(decoder2.shape)
        
        decoder1 = self.decoder1(decoder2, encoder1)
      
        res = self.post_process(decoder1)
        return res

if __name__ == "__main__":
    with open('ModelStats_unet.txt', 'a', encoding="utf-8") as file:
        file.write("-------------------------Model Summary----------------------------")
        file.write(str(summary(Unet(),input_size=(1,3,768,768), verbose=0)))
        file.write("\n")