import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import math


# --- Stage 1 ---

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
    def __init__(self, channel):
        super(occupiedNet, self).__init__()
        self.layer1 = nn.Sequential( # 차원을 늘려 특징 뽑아내기 + 크기 압축
            nn.Conv2d(in_channels=channel, out_channels=2 * channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(2 * channel),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential( # 특징을 더 농축하기
            nn.Conv2d(in_channels=2 * channel, out_channels=2 * channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * channel),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential( # 차원을 다시 줄이기
            nn.Conv2d(in_channels=2 * channel, out_channels=channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x : [B, 16, 64, 64]
        x = self.layer1(x) # x : [B, 16, 64, 64]
        x = self.layer2(x) # x : [B, 16, 64, 64]
        x_out = self.layer3(x) # x_out : [B, 16, 64, 64]
        return x_out # x_out : [B, 16, 64, 64]

# --- Stage 2 ---

class deformable_cross_attention(nn.Module):
    def __init__(self, d_model=128, Ns=8):
        super(deformable_cross_attention, self).__init__()
        self.d_model = d_model
        self.Ns = Ns
        
        self.delta_points = nn.Linear(d_model, Ns * 2) # delta point 뽑아내기 x랑 y
        self.A = nn.Linear(d_model, Ns) # As, q를 받아서 sample point의 중요도를 결정
        self.W = nn.Linear(d_model, d_model) # Ws, 2D특징을 3D복셀 차원에 맞게 가공
        self.value_proj = nn.Linear(d_model, d_model) # image feature를 미리 선형 변환
        self.output_proj = nn.Linear(d_model, d_model) # 다 하고 나서 쿼리 차원에 맞게 정리
    
    def forward(self, query, ref_points, image_features, mask):
        B, N, num_query, _ = ref_points.shape
        C = query.shape[-1]
        n_points = self.Ns

        p = ref_points # [B, N, 5000, 2]
        delta_p = self.delta_points(query).view(B, num_query, n_points, 2) # [B, 5000, 8, 2]
        p_delta_p = p.unsqueeze(3) + delta_p.unsqueeze(1) # [B, N, 5000, 8, 2]
        p_delta_p = p_delta_p * 2 - 1 # -1 ~ 1로 만들기
        p_delta_p = p_delta_p.view(B * N, num_query, n_points, 2) # [B * N, 5000, 8, 2]

        # image_features : [B * N, 128, 24, 77] p_delta_p : [B * N, 5000, 8, 2]
        # ImageF : [B, N, 5000, 8, 128]
        image_features = self.value_proj(image_features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        ImageF = F.grid_sample(image_features, p_delta_p, mode='bilinear', align_corners=True)
        ImageF = ImageF.view(B, N, C, num_query, n_points).permute(0, 1, 3, 4, 2)

        W_ImageF = self.W(ImageF) # W_ImageF : [B, N, 5000, 8, 128]

        As = F.softmax(self.A(query), dim=-1) # As = [B, 5000, 8]

        As_W_ImageF = W_ImageF * As.view(B, 1, num_query, n_points, 1) # [B, N, 5000, 8, 128]

        # mask : [B, N, 5000]
        summed_As_W_ImageF = As_W_ImageF.sum(dim=3) # [B, N, 5000, 128]
        summed_As_W_ImageF = summed_As_W_ImageF * mask.unsqueeze(-1) # [B, N, 5000, 128]
        summed_As_W_ImageF = summed_As_W_ImageF.sum(dim=1) # [B, 5000, 128]
        summed_As_W_ImageF = summed_As_W_ImageF / (mask.sum(dim=1).unsqueeze(-1) + 1e-6) # [B, 5000, 128]

        return self.output_proj(summed_As_W_ImageF) # [B, 5000, 128]
    
class FeedForwardNetwork(nn.Module): # 더 고차원적으로 특징을 추출
    def __init__(self, d_model=128, d_ff=512, dropout=0.1): # 512, 2048
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # 512, 2048
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model) # 2048, 512
        self.act = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x

class deformable_self_attention(nn.Module):
    def __init__(self, d_model, Ns):
        super(deformable_self_attention, self).__init__()
        self.d_model = d_model
        self.Ns = Ns
        
        self.delta_points = nn.Linear(d_model, Ns * 3) # delta point 뽑아내기 x랑 y랑 z
        self.A = nn.Linear(d_model, Ns) # As, q를 받아서 sample point의 중요도를 결정
        self.W = nn.Linear(d_model, d_model) # Ws, 2D특징을 3D복셀 차원에 맞게 가공
        self.value_proj = nn.Linear(d_model, d_model) # image feature를 미리 선형 변환
        self.output_proj = nn.Linear(d_model, d_model) # 다 하고 나서 쿼리 차원에 맞게 정리

    def forward(self, query, ref_points, image_features, mask):
        # query : [B, 5000, 128]
        # ref_points : [B, 5000, 3]
        # image_features : [B, 128, 16, 128, 128]
        # mask : [B, 5000]
        B, num_query, _ = ref_points.shape
        C = query.shape[-1]
        n_points = self.Ns

        p = ref_points # [B, 5000, 3]
        delta_p = self.delta_points(query).view(B, num_query, n_points, 3) # [B, 5000, 8, 3]
        p_delta_p = p.unsqueeze(2) + delta_p # [B, 5000, 8, 3]
        p_delta_p = p_delta_p[..., [1, 0, 2]]
        # p_delta_p = p_delta_p * 2 - 1 # -1 ~ 1로 만들기

        # image_features : [B, 128, 16, 128, 128] p_delta_p : [B, 5000, 8, 3]
        # ImageF : [B, 5000, 8, 128]
        ImageF = F.grid_sample(image_features, p_delta_p.unsqueeze(3), mode='bilinear', align_corners=True)
        ImageF = ImageF.squeeze(-1).permute(0, 2, 3, 1)
        ImageF = self.value_proj(ImageF)

        W_ImageF = self.W(ImageF) # W_ImageF : [B, 5000, 8, 128]

        As = F.softmax(self.A(query), dim=-1) # As = [B, 5000, 8]
        As_W_ImageF = W_ImageF * As.view(B, num_query, n_points, 1) # [B, 5000, 8, 128]

        summed_As_W_ImageF = As_W_ImageF.sum(dim=2) # [B, 5000, 128]
        mask = mask.any(dim=1)
        summed_As_W_ImageF = summed_As_W_ImageF * mask.unsqueeze(-1) # [B, 5000, 128]

        return self.output_proj(summed_As_W_ImageF) # [B, 5000, 128]

class VoxFormer(nn.Module):
    def __init__(self, d_model = 128, pred_num=4):
        super(VoxFormer, self).__init__()
        # --- Stage 1 ---
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential( # resnet18. 정보 응축
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3 
            # [256, 24, 77] 채널은 256, 크기는 1/16
        )
        self.neck = nn.Sequential( # resnet에서 나온 채널 256을 d_model로 만들어줌
            nn.Conv2d(256, d_model, kernel_size=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU()
        )
        self.depthNet = depthNet(d_model, 45) # neck이후 사용할 것. 깊이를 뽑아낼거임
        self.occupiedNet = occupiedNet(16) # 확률 복셀을 보다 더 확실하게 만들어줌

        # --- Stage 2 ---
        self.deformable_cross_attention = nn.ModuleList([
            deformable_cross_attention(128, 8) for _ in range(3)
        ])
        self.layer_norm_c_1 = nn.ModuleList([
            nn.LayerNorm(128) for _ in range(3)
        ])
        self.feedforwardNetwork_c = nn.ModuleList([
            FeedForwardNetwork(128, 512) for _ in range(3)
        ])
        self.layer_norm_c_2 = nn.ModuleList([
            nn.LayerNorm(128) for _ in range(3)
        ])
        self.deformable_self_attention = nn.ModuleList([
            deformable_self_attention(128, 8) for _ in range(2)
        ])
        self.layer_norm_s_1 = nn.ModuleList([
            nn.LayerNorm(128) for _ in range(2)
        ])
        self.feedforwardNetwork_s = nn.ModuleList([
            FeedForwardNetwork(128, 512) for _ in range(2)
        ])
        self.layer_norm_s_2 = nn.ModuleList([
            nn.LayerNorm(128) for _ in range(2)
        ])
        self.classifier = nn.Linear(d_model, pred_num)
        self.completion_head = nn.Sequential(
            nn.Conv3d(in_channels=pred_num, out_channels=2*pred_num, kernel_size=3, padding=1),
            nn.BatchNorm3d(2*pred_num),
            nn.ReLU(),
            nn.Conv3d(in_channels=2*pred_num, out_channels=pred_num, kernel_size=3, padding=1),
            nn.BatchNorm3d(pred_num),
        )
        self.upsample = nn.Upsample(size=(16, 64, 64), mode='trilinear', align_corners=True)

        # 시간을 11초에서 4초로 일단 단축

        self.image_height = 370 # 원본 이미지 높이
        self.image_width = 1220 # 원본 이미지 너비
        self.grid_height = 24 # depthNet이후 이미지 높이
        self.grid_width = 77 # depthNet이후 이미지 너비
        self.depth_min = 1 # 최소 깊이 
        self.depth_max = 46 # 최대 깊이 + 1
        self.depth_steps = 1 # 깊이의 단계
        self.depth_bin = 45 # 깊이 차원 개수
        self.voxel_x = 64 # 복셀 좌우
        self.voxel_y = 16 # 복셀 앞뒤
        self.voxel_z = 64 # 복셀 상하
        self.voxel_size = 0.5 # 복셀 사이즈
        self.k = 2500

        self.query_embedding = nn.Embedding(self.k, d_model)
        self.position_encoder = nn.Linear(3, d_model)

        self.register_buffer('frustum', self.create_frustum())

    def forward(self, image, intrinics, rots, trans):
        # image : [B, N, 3, 370, 1220] 
        # intrinics : [B, N, 3, 3]
        # rots : [B, N, 3, 3]
        # trans : [B, N, 3]
        # out of memory때문에 단안으로 진행

        B, N, C, H, W = image.shape
        t0 = torch.cuda.Event(enable_timing=True) # 얼마나 걸리는지 시간 체크용
        t1 = torch.cuda.Event(enable_timing=True) # 얼마나 걸리는지 시간 체크용
        # --- Stage 1 ---
        t0.record()

        image = image.view(B * N, C, H, W) # [B * N, C, H, W] 이미지를 Conv2d에 넣기 좋게 만듬
        image_features = self.backbone(image) # [B * N, C, H, W] -> [B * N, 256, 24, 77] 정보를 뽑아냄
        image_features = self.neck(image_features) # [B * N, 256, 24, 77] -> [B * N, d_model, 24, 77] 차원 낮추기
        
        depth_disparity = self.depthNet(image_features) # [B * N, d_model, 24, 77] -> [B * N, 45, 24, 77] 45개 깊이의 가능성
        _, D, H_new, W_new = depth_disparity.shape # [D는 45, H_new, W_new는 resnet 이후의 높이와 너비]
        depth_disparity = depth_disparity.view(B, N, D, H_new, W_new) # [B, N, 45, 24, 77] 깊이, 높이, 너비 | 2.5D 표현 | 깊이 추정

        geometry = self.get_geometry(intrinics, rots, trans) # geometry : [B, N, 45, 24, 77, 3] | 3차원 실수 좌표값 | 역투영
        voxelized_pseudo_point_cloud = self.voxelization(depth_disparity, geometry) # voxelized_pseudo_point_cloud : [B, 16, 64, 64] | M_in
        M_out = self.occupiedNet(voxelized_pseudo_point_cloud) # [B, 16, 32, 32] 높이, 좌우, 앞뒤 

        query_proposals = self.get_query_proposal(M_out) # [B, 2500, 3] | 이 안에는 복셀중 물체가 있을 가능성이 가장 높은 복셀의 인덱스 2500개 들어있음 | Q_p
        
        t1.record()
        torch.cuda.synchronize()
        print(f'Stage 1 : {t0.elapsed_time(t1)/1000:.4f}s')
        # --- Stage 2 ---  

        t0.record()
        # reference_point : [B, N, 2500, 2] 복셀 3차원을 이미지 2차원 좌표로 다시 되돌림
        # mask : [B, N, 2500] 복셀이 N개의 이미지중 어디에 있는지 True와 False로 
        reference_points, mask = self.get_reference_point(query_proposals, intrinics, rots, trans)

        # [B, 2500, 128]
        # 위치좌표 였던 3이 128개의 위치적 의미로 변한다. 어디를 볼지이다.
        position_embedding = self.position_encoder(query_proposals.float())
        
        # [B, 2500, 128]
        # 2500개의 가능성이 있는 복셀안의 내용을 추적할것이다. 
        content_embedding = self.query_embedding.weight.unsqueeze(0).expand(B, -1, -1)

        # [B, 2500, 128] 
        # 위치적 의미 + 내용 = 위치를 알고 내용도 안다.
        query = content_embedding + position_embedding
        batch_idx = torch.arange(B, device=query.device).view(B, 1).expand(B, 2500)
        grid_size = torch.tensor([32, 16, 32], device=query_proposals.device)
        reference_points_3d = query_proposals / (grid_size - 1.0)
        reference_points_3d = reference_points_3d * 2 - 1 # [B, 2500, 3]
        x_idx = torch.clamp(query_proposals[..., 0].long(), 0, 32 -1) # [B, 2500] 좌우
        y_idx = torch.clamp(query_proposals[..., 1].long(), 0, 16 -1) # [B, 2500] 상하
        z_idx = torch.clamp(query_proposals[..., 2].long(), 0, 32 -1) # [B, 2500] 앞뒤

        t1.record()
        torch.cuda.synchronize()
        print(f'Before transformer : {t0.elapsed_time(t1)/1000:.4f}s')


        t0.record()
        voxel_volume = torch.zeros(B, 128, 16, 32, 32, device=query.device)
        for i in range(3):
            query = query + self.deformable_cross_attention[i](self.layer_norm_c_1[i](query), reference_points, image_features, mask)
            query = query + self.feedforwardNetwork_c[i](self.layer_norm_c_2[i](query))

        t1.record()
        torch.cuda.synchronize()
        print(f'Until DCA : {t0.elapsed_time(t1)/1000:.4f}s')

        t0.record()
        for i in range(2):
            voxel_volume.zero_()
            voxel_volume[batch_idx, :, y_idx, x_idx, z_idx] = query # [B, 128, 16, 32, 32]
            query = query + self.deformable_self_attention[i](self.layer_norm_s_1[i](query), reference_points_3d, voxel_volume, mask)
            query = query + self.feedforwardNetwork_s[i](self.layer_norm_s_2[i](query))

        t1.record()
        torch.cuda.synchronize()
        print(f'Until DSA : {t0.elapsed_time(t1)/1000:.4f}s')

        # 이제 query는 [B, 2500, 128]
        t0.record()
        logits = self.classifier(query) # [B, 2500, 20]
        low_res_grid = torch.zeros(B, 4, 16, 32, 32, device=logits.device) # [B, 4, 16, 32, 32]
        low_res_grid[batch_idx, :, y_idx, x_idx, z_idx] = logits # [B, 4, 16, 32, 32]

        out = low_res_grid + self.completion_head(low_res_grid) # [B, 4, 16, 32, 32]
        out = self.upsample(out) # [B, 4, 16, 32, 32]

        t1.record()
        torch.cuda.synchronize()
        print(f'Until End : {t0.elapsed_time(t1)/1000:.4f}s')

        return out, M_out

# --- Stage 1 ---

    def create_frustum(self):
        xs = torch.linspace(0, self.image_width - 1, self.grid_width).float()
        ys = torch.linspace(0, self.image_height - 1, self.grid_height).float()
        ds = torch.arange(self.depth_min, self.depth_max, self.depth_steps).float()

        d, y, x = torch.meshgrid(ds, ys, xs, indexing='ij') 
        frustum = torch.stack([x * d, y * d, d], dim=-1)

        return frustum
    
    def get_geometry(self, intrinics, rots, trans):
        B, N = intrinics.shape[0], intrinics.shape[1]
        frustum = self.frustum # [45, 24, 77, 3] 깊이, 높이, 너비, xyz

        inv_intrincs = torch.inverse(intrinics) # [B, N, 3, 3]

        # inv_inrincs : [B, N, 1,  1,  1,  3, 3]
        # frustum :     [1, 1, 45, 24, 77, 3, 1]
        points_c = torch.matmul(inv_intrincs.view(B, N, 1, 1, 1, 3, 3), frustum.view(1, 1, self.depth_bin, self.grid_height, self.grid_width, 3, 1))
        # points_c : [B, N, 45, 24, 77, 3, 1]
        # print(f'points_c : {points_c.shape}')

        points_w = torch.matmul(rots.view(B, N, 1, 1, 1, 3, 3), points_c) + trans.view(B, N, 1, 1, 1, 3, 1)
        # points_w : [B, N, 45, 24, 77, 3, 1]
        # print(f'points_w : {points_w.shape}')
        return points_w.squeeze(-1)
    
    def voxelization(self, depth_disparity, geometry):
        # depth_disparity : [B, N, 45, 24, 77]
        # geometry : [B, N, 45, 24, 77, 3] | 3차원 실수 좌표값 | 역투영
        B, N, D, H, W, C = geometry.shape
        device = geometry.device

        geom = geometry.reshape(-1, 3) # [B * N * 45 * 24 * 77, 3]
        flat_depth_disparity = depth_disparity.reshape(-1)

        nx = ((geom[:, 0] + self.voxel_x * self.voxel_size / 2) / self.voxel_size).long()
        ny = ((geom[:, 1] + self.voxel_y * self.voxel_size / 2) / self.voxel_size).long()
        nz = ((geom[:, 2] + self.voxel_z * self.voxel_size / 2) / self.voxel_size).long()

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

        voxels = torch.zeros((B * self.voxel_x * self.voxel_y * self.voxel_z), device=device)
        voxels.index_add_(0, idx, flat_depth_disparity)

        voxels = voxels.view(B, self.voxel_x, self.voxel_y, self.voxel_z).permute(0, 2, 1, 3) # [B, 16, 64, 64]
        voxels = torch.clamp(voxels, 0.0, 1.0)

        return voxels # [B, 16, 64, 64]

    def get_query_proposal(self, M_out):
        # M_out : [B, 16, 32, 32] 높이, 좌우, 앞뒤 
        B, Y, X, Z = M_out.shape
        M_out = M_out.view(B, -1) # [B, 16 * 32 * 32]
        topk_value, topk_indexes = torch.topk(M_out, self.k, dim=-1) # 2500개만 뽑기

        y = topk_indexes // (X * Z)
        x = (topk_indexes // X) % Z
        z = topk_indexes % Z

        query_proposals = torch.stack([y, x, z], dim=-1) # [B, 2500, 3] 높이, 좌우, 앞뒤

        return query_proposals

# --- Stage 2 ---  

    def get_reference_point(self, query_proposals, intrinics, rots, trans):
        # query_proposal : [B, 2500, 3]
        B, N, _, _ = intrinics.shape # B, N

        # 현재 query_proposals안의 값은 복셀의 인덱스임, 이걸 원래 좌표계로 돌려야함
        # 지금까지 한거 반대로 하면 됨
        # 0.5는 중앙을 보기 하기 위함이고 2 * self.voxel_size는 occupiedNet에서 크기가 절반이 되었기 때문
        nx = (query_proposals[:, :, 1] + 0.5) * 2 * self.voxel_size - (self.voxel_x * self.voxel_size / 2)
        ny = (query_proposals[:, :, 0] + 0.5) * 2 * self.voxel_size - (self.voxel_y * self.voxel_size / 2)
        nz = (query_proposals[:, :, 2] + 0.5) * 2 * self.voxel_size - (self.voxel_z * self.voxel_size / 2)

        points_world = torch.stack([nx, ny, nz], dim=-1).unsqueeze(1) # [B, 1, 2500, 3]
        points_world = points_world - trans.view(B, N, 1, 3) # 원래는 [B, N, 3]
        # rots         -> [B, N, 1, 3, 3]인데 뒤의 3, 3이 바뀜
        # points_world -> [B, 1, 2500, 3, 1]
        # points_camera : [B, N, 2500, 3, 1(사라짐)]
        points_camera = torch.matmul(rots.transpose(-1, -2).view(B, N, 1, 3, 3), points_world.unsqueeze(-1)).squeeze(-1)

        # intrinics -> [B, N, 1, 3, 3]
        # points_camera -> [B, N, 2500, 3, 1]
        # points_image : [B, N, 2500, 3, 1(사라짐)]
        points_image = torch.matmul(intrinics.view(B, N, 1, 3, 3), points_camera.unsqueeze(-1)).squeeze(-1)

        depth = points_image[..., 2] # xd, yd, d
        xy = points_image[..., :2] / (depth.unsqueeze(-1) + 1e-6) # xd / d, yd / d

        # 내 앞에 있는지, 범위 안에 있는지
        mask = (depth > 0.1) & (0 <= xy[..., 0]) & (xy[..., 0] < self.image_width) & (0 <= xy[..., 1]) & (xy[..., 1] < self.image_height)

        reference_point = torch.zeros_like(xy) # 이건 좌표가 아닌 위치 비율
        reference_point[..., 0] = xy[..., 0] / self.image_width # 정규화 0~1
        reference_point[..., 1] = xy[..., 1] / self.image_height # 정규화 0~1
        
        return reference_point, mask

    

