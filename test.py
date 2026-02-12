import torch
import torch.nn as nn
import torch.nn.functional as F
from VoxFormer import VoxFormer
import time

# 배치가 8이면 out of memory가 뜨고
# 배치가 4이면 out of memory가 뜨고
# 배치가 2이면 10초 걸린다
# 배치가 1이면 0.7초 걸린다
B, N, c, h, w = 2, 6, 3, 370, 1220
# B, N, c, h, w = 4, 6, 2, 3, 128, 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VoxFormer().to(device)
model.eval()

test_image = torch.randn(B, N, c, h, w).to(device)
test_intrinsics = torch.randn(B, N, 3, 3).to(device)
test_rots= torch.randn(B, N, 3, 3).to(device)
test_trans = torch.randn(B, N, 3).to(device)

start_time = time.time()
outputs = model(test_image, test_intrinsics, test_rots, test_trans)
end_time = time.time()
print(f'걸린 시간 : {end_time - start_time:.5f}')

print(outputs.shape)