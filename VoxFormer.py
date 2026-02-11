import sys
import os

# 현재 파일의 위치를 기준으로 mobilestereonet 폴더 경로를 추가합니다.
# current_path = os.path.dirname(os.path.abspath(__file__))
# lib_path = os.path.join(current_path, 'mobilestereonet')
# sys.path.append(lib_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
#from mobilestereonet.models.MSNet3D import MSNet3D

class depthNet(nn.Module):
    def __init__(self, C, D): # C : 512, D : 45
        super(depthNet, self).__init__()
        self.depth = D
        self.depthnet = nn.Conv2d(C, D, kernel_size=1, padding=0)

    def forward(self, x):
        depth = self.depthnet(x)

        depth = F.softmax(depth, dim=1) # 깊이를 기준으로 확률분포를 생성, 어느 깊이가 가장 가능성 있는 깊이인지 알 수 있음

        return depth

class occupiedNet(nn.Module):
    def __init__(self):
        super(occupiedNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), # 256 -> 128
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x : [B, 32, 256, 256]
        x = self.layer1(x)
        x = self.layer2(x)
        x_out = self.layer3(x)
        return x_out

class VoxFormer(nn.Module):
    def __init__(self):
        super(VoxFormer, self).__init__()
        # self.mobilestereonet = MSNet3D(maxdisp=64) # 무조건 out of memory 남
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT) # out of memory를 방지하기 위해 50이 아닌 18로
        self.backbone = nn.Sequential(*list(resnet.children())[:-2]) # 마지막 Pooling/FC 제거
        self.depthNet = depthNet(512, 45) # resnet이후 사용할 것
        self.occupiedNet = occupiedNet() # 확률 복셀을 보다 더 확실하게 만들어줌

        self.image_height = 370 # 원본 이미지 높이
        self.image_width = 1220 # 원본 이미지 너비
        self.grid_height = 12 # depthNet이후 이미지 높이
        self.grid_width = 39 # depthNet이후 이미지 너비
        self.depth_min = 1 # 최소 깊이 
        self.depth_max = 46 # 최대 깊이 + 1
        self.depth_steps = 1 # 깊이의 단계
        self.depth_bin = 45 # 깊이 차원 개수
        self.voxel_y = 32
        self.voxel_x = 256
        self.voxel_z = 256
        self.voxel_size = 0.2

        self.register_buffer('frustum', self.create_frustum())

    def forward(self, image, intrinics, rots, trans):
        # image : [B, N, 3, 370, 1220] 
        # intrinics : [B, N, 3, 3]
        # rots : [B, N, 3, 3]
        # trans : [B, N, 3]
        # out of memory때문에 단안으로 진행

        B, N, C, H, W = image.shape

        # --- Stage 1 ---
        image = image.view(B * N, C, H, W)
        features = self.backbone(image) # [B * N, 512, 12, 39]
        depth_disparity = self.depthNet(features) # [B * N, 45, 12, 39]
        _, D, H_new, W_new = depth_disparity.shape
        depth_disparity = depth_disparity.view(B, N, D, H_new, W_new) # [B, N, 45, 12, 39] 깊이, 높이, 너비 | 2.5D 표현 | 깊이 추정

        geometry = self.get_geometry(intrinics, rots, trans) # geometry : [B, N, 45, 12, 39, 3] | 3차원 실수 좌표값 | 역투영

        voxelized_pseudo_point_cloud = self.voxelization(depth_disparity, geometry) # voxelized_pseudo_point_cloud : [B, 32, 256, 256] | M_in

        M_out = self.occupiedNet(voxelized_pseudo_point_cloud) # [B, 16, 128, 128] 높이, 좌우, 앞뒤 

        query_proposals = self.query_proposal(M_out) # [B, 5000, 3] | 이 안에는 복셀중 가장 가능성이 있는 복셀이 5000개 들어있음

        # --- Stage 2 ---  
              

        return query_proposals

    def create_frustum(self):
        xs = torch.linspace(0, self.image_width - 1, self.grid_width).float()
        ys = torch.linspace(0, self.image_height - 1, self.grid_height).float()
        ds = torch.arange(self.depth_min, self.depth_max, self.depth_steps).float()

        d, y, x = torch.meshgrid(ds, ys, xs, indexing='ij') 
        frustum = torch.stack([x * d, y * d, d], dim=-1)

        return frustum
    
    def get_geometry(self, intrinics, rots, trans):
        B, N = intrinics.shape[0], intrinics.shape[1]
        frustum = self.frustum # [45, 12, 39, 3] 깊이, 높이, 너비, xyz

        inv_intrincs = torch.inverse(intrinics) # [B, N, 3, 3]

        # inv_inrincs : [B, N, 1,  1,  1,  3, 3]
        # frustum :     [1, 1, 45, 12, 39, 3, 1]
        points_c = torch.matmul(inv_intrincs.view(B, N, 1, 1, 1, 3, 3), frustum.view(1, 1, self.depth_bin, self.grid_height, self.grid_width, 3, 1))
        # points_c : [B, N, 45, 12, 39, 3, 1]
        # print(f'points_c : {points_c.shape}')

        points_w = torch.matmul(rots.view(B, N, 1, 1, 1, 3, 3), points_c) + trans.view(B, N, 1, 1, 1, 3, 1)
        # points_w : [B, N, 45, 12, 39, 3, 1]
        # print(f'points_w : {points_w.shape}')
        return points_w.squeeze(-1)
    
    def voxelization(self, depth_disparity, geometry):
        # depth_disparity : [B, N, 45, 12, 39]
        # geometry : [B, N, 45, 12, 39, 3] | 3차원 실수 좌표값 | 역투영
        B, N, D, H, W, C = geometry.shape
        device = geometry.device

        geom = geometry.reshape(-1, 3) # [B * N * 45 * 12 * 39, 3]
        flat_depth_disparity = depth_disparity.reshape(-1)

        nx = ((geom[:, 0] + self.voxel_x/2) / self.voxel_size).long()
        ny = ((geom[:, 2] + self.voxel_y/2) / self.voxel_size).long()
        nz = ((geom[:, 1] + self.voxel_z/2) / self.voxel_size).long()

        x_valid = (0 <= nx) & (nx < self.voxel_x)
        y_valid = (0 <= ny) & (ny < self.voxel_y)
        z_valid = (0 <= nz) & (nz < self.voxel_z)

        mask = x_valid & y_valid & z_valid

        nx = nx[mask]
        ny = ny[mask]
        nz = nz[mask]
        flat_depth_disparity = flat_depth_disparity[mask]
        geom = geom[mask]

        batch_idx = torch.arange(B, device=device).view(B, 1, 1, 1, 1).expand(B, N, D, H, W)
        batch_idx = batch_idx.reshape(-1)[mask]
        idx = batch_idx * (self.voxel_x * self.voxel_y * self.voxel_z) + nx * (self.voxel_y * self.voxel_z) + ny * self.voxel_z + nz

        voxels = torch.zeros((B * self.voxel_x * self.voxel_y * self.voxel_z), device=device) # [B, 32, 256, 256] 상하, 좌우, 앞뒤
        voxels.index_add_(0, idx, flat_depth_disparity)

        voxels = voxels.view(B, self.voxel_x, self.voxel_y, self.voxel_z).permute(0, 2, 1, 3)
        voxels = torch.clamp(voxels, 0.0, 1.0)

        return voxels

    def query_proposal(self, M_out):
        # M_out : [B, 16, 128, 128] 높이, 좌우, 앞뒤 
        B = M_out.shape[0]
        M_out = M_out.view(B, -1) # [B, 16 * 128 * 128]
        topk_value, topk_indexes = torch.topk(M_out, 5000, dim=-1)

        y = topk_indexes // (128 * 128)
        x = (topk_indexes // 128) % 128
        z = topk_indexes % 128

        query_proposals = torch.stack([y, x, z], dim=-1) # [B, 5000, 3] 높이, 좌우, 앞뒤

        return query_proposals
