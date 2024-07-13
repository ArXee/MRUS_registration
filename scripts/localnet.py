import torch
import torch.nn as nn
import torch.nn.functional as F
class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()

    def forward(self,image, ddf):
        '''B, C, D, H, W = moving.size()
        # Create meshgrid for 3D volume
        gridD, gridH, gridW = torch.meshgrid(
            torch.linspace(-1, 1, D),
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            indexing="ij"
        )
        base_grid = torch.stack([gridW, gridH, gridD], dim=-1).to(moving.device)  # Shape (D, H, W, 3)
        base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1, 1)  # Shape (B, D, H, W, 3)

        # Apply displacement
        warped_grid = base_grid + ddf.permute(0, 2, 3, 4, 1)  # Ensure ddf is (B, D, H, W, 3)
        warped_grid = warped_grid.permute(0, 4, 1, 2, 3)  # Convert to (B, 3, D, H, W) for grid_sample

        # Grid sample expects grid in the range of [-1, 1]
        moved = F.grid_sample(moving, warped_grid, mode='bilinear', padding_mode='border', align_corners=True)'''

        N, _, D, H, W = image.size()
        # 创建一个批量大小与输入图像相匹配的单位矩阵
        identity = torch.eye(3, 4).unsqueeze(0).repeat(N, 1, 1).to(image.device)
        base_grid = F.affine_grid(identity, [N, _, D, H, W], align_corners=True)

        # 将位移场(DDF)添加到网格上
        warped_grid = base_grid + ddf.permute(0, 2, 3, 4, 1)

        # 使用warped_grid采样原始图像
        warped_image = F.grid_sample(image, warped_grid, mode='bilinear', padding_mode='border', align_corners=True)

        return warped_image



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.pool = nn.MaxPool3d(3, stride=2, padding=1)
        self.residual_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)  # 提前定义

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        # 使用预定义的卷积
        residual = nn.functional.interpolate(residual, scale_factor=0.5, mode='trilinear', align_corners=False)
        residual = self.residual_conv(residual)
        return x + residual

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        # 上采样时，先进行转置卷积调整尺寸
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv1 = ConvBlock(out_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        # 1x1 卷积用于残差连接的通道匹配
        self.residual_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        # ReLU激活函数
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        # 保存并调整残差

        residual = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        residual = self.residual_conv(residual)
        x = self.upconv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        # print(x.shape)
        # print( residual.shape)
        # 将残差添加到输出
        x += residual
        # 应用ReLU激活函数
        x = self.relu(x)
        return x

class UNet3D(nn.Module):
    def __init__(self):
        super(UNet3D, self).__init__()
        self.down1 = DownBlock(2, 32)
        self.down2 = DownBlock(32, 64)
        self.down3 = DownBlock(64, 128)
        self.down4 = DownBlock(128, 256)

        # 确保为每个上采样块传递正确的跳跃连接通道数
        self.up1 = UpBlock(256, 128 )  # skip来自down3
        self.up2 = UpBlock(128, 64)    # skip来自down2
        self.up3 = UpBlock(64, 32)     # skip来自down1
        self.up4 = UpBlock(32, 16)  # skip来自down1

        self.final_conv = nn.Conv3d(16, 3, kernel_size=1)

    def forward(self, moving, fixed):
        x = torch.cat([moving, fixed], dim=1)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x5= self.up1(x4)
        x6 = self.up2(x5)
        x7 = self.up3(x6)
        x8=self.up4(x7)
        return self.final_conv(x8)


class Localnet(nn.Module):
    def __init__(self):
        super(Localnet, self).__init__()
        self.unet = UNet3D()  # Assuming UNet3D is defined as in the previous example
        self.stn = SpatialTransformer()

    def forward(self, moving, fixed, moving_labels):
        ddf = self.unet(moving, fixed)
        # print(ddf.shape)
        moved_labels = self.stn(moving_labels, ddf)

        return ddf,moved_labels


# Example usage
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# x=torch.randn((1,1,64,64,64))
# y=torch.randn((1,1,64,64,64))
# model = UNet3D().to(device)
# ddf=model(x,y)
# print(model)
