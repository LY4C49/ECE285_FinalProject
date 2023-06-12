import torch.nn as nn
import torch.nn.functional as F
import torch
from torchinfo import summary

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]
        
        # Initial convolution block
        out_features = 64
        
        self.pre_process = nn.Sequential(
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        )

        in_features = out_features
        
        # Downsampling
        self.downsampling = nn.Sequential(
            nn.Conv2d(in_features, out_features * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_features * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features * 2, out_features * 4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_features * 4),
            nn.ReLU(inplace=True),
        )
        
        out_features *= 4
        in_features = out_features

        # Residual blocks
        res = []
        for _ in range(num_residual_blocks):
            res += [ResidualBlock(out_features)]
        
        self.res = nn.Sequential(*res)
        
        # Upsampling
        self.upsampling = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_features, out_features // 2, 3, stride=1, padding=1),
            nn.InstanceNorm2d(out_features // 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_features // 2, out_features // 4, 3, stride=1, padding=1),
            nn.InstanceNorm2d(out_features // 4),
            nn.ReLU(inplace=True),
        )
        
        out_features //= 4
        
        self.post_process = nn.Sequential(
            nn.ReflectionPad2d(channels), 
            nn.Conv2d(out_features, channels, 7), 
            nn.Tanh()
        )

    def forward(self, x):
        x = self.pre_process(x)
        x = self.downsampling(x)
        x = self.res(x)
        x = self.upsampling(x)
        x = self.post_process(x)
        return x

if __name__ == "__main__":
    model = GeneratorResNet(input_shape= (3, 512, 512), num_residual_blocks= 9)
    
    src = torch.randn(size= (1, 3, 512, 512))
    
    res = model(src)
    
    print("res:", res.shape)
    
    with open('ModelStats_res.txt', 'a', encoding="utf-8") as file:
        file.write("-------------------------Model Summary----------------------------")
        file.write(str(summary(model,input_size=(1,3,512,512), verbose=0)))
        file.write("\n")