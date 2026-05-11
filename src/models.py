import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch)
        )
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch)
            )
            
    def forward(self, x):
        return F.relu(self.conv(x) + self.shortcut(x), inplace=True)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UNetDecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.att_gate = AttentionGate(F_g=in_ch // 2, F_l=in_ch // 2, F_int=in_ch // 4)
        self.conv = ResidualBlock(in_ch, out_ch)
        
    def forward(self, x, skip):
        x = self.upsample(x)
        skip = self.att_gate(g=x, x=skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.enc1 = ResidualBlock(3, 32)
        self.enc2 = ResidualBlock(32, 64)
        self.enc3 = ResidualBlock(64, 128)
        self.enc4 = ResidualBlock(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ResidualBlock(256, 512)

        self.dec4 = UNetDecoderBlock(512, 256)
        self.dec3 = UNetDecoderBlock(256, 128)
        self.dec2 = UNetDecoderBlock(128, 64)
        self.dec1 = UNetDecoderBlock(64, 32)
        self.final = nn.Conv2d(32, 3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s1 = self.enc1(x)
        p1 = self.pool(s1)
        s2 = self.enc2(p1)
        p2 = self.pool(s2)
        s3 = self.enc3(p2)
        p3 = self.pool(s3)
        s4 = self.enc4(p3)
        p4 = self.pool(s4)
        b = self.bottleneck(p4)
        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        return self.sigmoid(self.final(d1))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        att = torch.cat([avg_out, max_out], dim=1)
        att = self.conv(att)
        return x * self.sigmoid(att)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.initial = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.initial(x)
        l1 = self.layer1(x)
        l2 = self.layer2(l1) # 128x16x16 for 128x128 input
        l3 = self.layer3(l2) # 256x8x8
        l4 = self.layer4(l3) # 512x4x4
        return l2, l3, l4

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.conv5(self.pool(x))
        x5 = F.interpolate(x5, size=x.shape[2:], mode='bilinear', align_corners=True)
        return self.out_conv(torch.cat((x1, x2, x3, x4, x5), dim=1))

class PredictorFCN(nn.Module):
    def __init__(self):
        super(PredictorFCN, self).__init__()
        # Input: residual_map (1) + l4 (512) = 513
        self.aspp = ASPP(in_channels=513, out_channels=128)

        self.att1 = SpatialAttention()
        
        self.conv1 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        
        self.att2 = SpatialAttention()
        
        self.conv3 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 1, 1)

    def forward(self, res_map, resnet_feats):
        l2, l3, l4 = resnet_feats
        
        # Upscale multi-scale features to match res_map
        l4_up = F.interpolate(l4, size=res_map.shape[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat([res_map, l4_up], dim=1)

        x = self.aspp(x)
        x = self.att1(x)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        x = self.att2(x)
        
        x = F.relu(self.conv3(x))
        return self.conv4(x)

class RLAgent(nn.Module):
    def __init__(self, out_dim=128):
        super(RLAgent, self).__init__()
        # State: 6 channels (RGB, Context, History, Pos) + Backbone features
        # Backbone l3 is 256x8x8. We'll pool state to 8x8 as well.
        self.shared = nn.Sequential(
            nn.Conv2d(6 + 256, 128, 3, stride=2, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, stride=2, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)), # Cố định đầu ra 2x2 để Linear luôn là 256
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        self.action_head = nn.Linear(256, out_dim)
        self.threshold_head = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, state, resnet_feats):
        l2, l3, l4 = resnet_feats
        # Ensure state_pooled spatial size matches l3 (typically 8x8 or 16x16 depending on input size)
        target_h, target_w = l3.shape[2], l3.shape[3]
        state_pooled = F.adaptive_avg_pool2d(state, (target_h, target_w))
        
        x = torch.cat([state_pooled, l3], dim=1)
        
        features = self.shared(x)
        
        probs = F.softmax(self.action_head(features), dim=-1)
        threshold = self.threshold_head(features)
        
        return probs, threshold
