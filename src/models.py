import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class UNetEncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNetDecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = UNetEncoderBlock(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.enc1 = UNetEncoderBlock(3, 64)
        self.enc2 = UNetEncoderBlock(64, 128)
        self.enc3 = UNetEncoderBlock(128, 256)
        self.enc4 = UNetEncoderBlock(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = UNetEncoderBlock(512, 1024)
        self.dec4 = UNetDecoderBlock(1024, 512)
        self.dec3 = UNetDecoderBlock(512, 256)
        self.dec2 = UNetDecoderBlock(256, 128)
        self.dec1 = UNetDecoderBlock(128, 64)
        self.final = nn.Conv2d(64, 3, 1)
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

class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-2]) 
        # For 256x256 input, output is 512x8x8
    def forward(self, x):
        return self.features(x)

class PredictorFCN(nn.Module):
    def __init__(self):
        super(PredictorFCN, self).__init__()
        # Bottleneck ResNet features from 512 to 64 to save memory during upsampling
        self.resnet_bottleneck = nn.Conv2d(512, 64, 1)
        # Input: Residual Map(1) concatenated with Upsampled Bottlenecked features(64) = 65
        self.conv1 = nn.Conv2d(65, 128, 3, padding=1)
        self.att1 = SpatialAttention()
        self.conv2 = nn.Conv2d(128, 64, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(64, 32, 3, padding=4, dilation=4)
        self.att2 = SpatialAttention()
        self.conv4 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv5 = nn.Conv2d(16, 1, 1)

    def forward(self, res_map, resnet_features):
        # resnet_features is 512x8x8, bottleneck to 64x8x8
        feat_reduced = self.resnet_bottleneck(resnet_features)
        # upscale to 256x256
        feat_upscaled = F.interpolate(feat_reduced, size=res_map.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([res_map, feat_upscaled], dim=1)
        
        x = F.relu(self.conv1(x))
        x = self.att1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.att2(x)
        x = F.relu(self.conv4(x))
        # Return logits for stability with BCEWithLogitsLoss
        return self.conv5(x)

class RLAgent(nn.Module):
    def __init__(self):
        super(RLAgent, self).__init__()
        # Input 6 channels + ResNet Context
        # Let's simplify and use ResNet features as primary state
        self.features = nn.Sequential(
            nn.Conv2d(518, 128, 3, stride=2, padding=1), # 4x4 if input is 8x8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 256) 
        )

    def forward(self, state, resnet_features):
        # state is 6x256x256, pool down to 8x8 to match resnet_features
        state_pooled = F.adaptive_avg_pool2d(state, (8, 8))
        x = torch.cat([state_pooled, resnet_features], dim=1) # 6 + 512 = 518 channels
        logits = self.features(x)
        return F.softmax(logits, dim=-1)
