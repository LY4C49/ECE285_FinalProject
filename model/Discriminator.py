import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class Discriminator(nn.Module):
    def __init__(self, channel_list = [3, 64, 128, 256], dropout = 0.0) -> None:
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels = channel_list[0], out_channels = channel_list[1], kernel_size = 4, padding = 1, stride= 2),
                nn.BatchNorm2d(channel_list[1]),
                nn.ReLU(inplace= True)
            )
        
        self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels = channel_list[1], out_channels = channel_list[2], kernel_size = 4, padding = 1, stride= 2),
                nn.BatchNorm2d(channel_list[2]),
                nn.ReLU(inplace= True)
            )
        
        self.conv3 = nn.Sequential(
                nn.Conv2d(in_channels = channel_list[2], out_channels = channel_list[3], kernel_size = 4, padding = 1, stride= 2),
                nn.BatchNorm2d(channel_list[3]),
                nn.ReLU(inplace= True)
            )
        
        self.post_processor = nn.Conv2d(in_channels= channel_list[-1], out_channels= 1, kernel_size= 1, stride= 1, padding= 0)

        self.dropout = nn.Dropout(dropout)

        for m in self.modules():
             if isinstance(m, nn.Conv2d):
                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                 nn.init.constant_(m.weight, 1)
                 nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.post_processor(x)
        x = self.dropout(x)
        return x
        
if __name__ == "__main__":
    with open('ModelStats_dis.txt', 'a', encoding="utf-8") as file:
        file.write("-------------------------Model Summary----------------------------")
        file.write(str(summary(Discriminator(),input_size=(1,6,768,768), verbose=0)))
        file.write("\n") 
    
    src = torch.randn(size = (1, 6, 768, 768))
    model = Discriminator()
    x = model(src)
    print(x.shape)