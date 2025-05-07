import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        # Enhanced encoder
        self.inc = DoubleConv(3, 64)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256)
        )
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_conv1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up_conv2 = DoubleConv(128, 64)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_conv4 = DoubleConv(256, 128)
        self.up5 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_conv5 = DoubleConv(256, 128)
        self.up6 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_conv6 = DoubleConv(256, 128)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_conv7 = DoubleConv(256, 128)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_conv8 = DoubleConv(256, 128)
        
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        # Decoder
        x = self.up1(x3)
        x = self.up_conv1(torch.cat([x, x2], dim=1))
        x = self.up2(x)
        x = self.up_conv2(torch.cat([x, x1], dim=1))
        return self.outc(x)