import torch
import torch.nn as nn
import torch.nn.functional as F
from VoxFormer import VoxFormer

B, N, c, h, w = 4, 6, 3, 370, 1220
# B, N, c, h, w = 4, 6, 2, 3, 128, 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VoxFormer().to(device)
model.eval()

test_image = torch.randn(B, N, c, h, w).to(device)
test_intrinsics = torch.randn(B, N, 3, 3).to(device)
test_rots= torch.randn(B, N, 3, 3).to(device)
test_trans = torch.randn(B, N, 3).to(device)

outputs = model(test_image, test_intrinsics, test_rots, test_trans)

print(outputs.shape)