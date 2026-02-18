import torch
import torch.nn as nn
import torch.nn.functional as F
from VoxFormer import VoxFormer
import time

# 현재 목표
# 복셀맵 사이즈 : [64, 16, 64]
# 복셀 사이즈 : 0.5
# 이유 : 0.2로 하면 너무 가깝고 0.5가 적당하다. 사이즈를 더 늘리면 이미지로 보이지 않는 너무 멀리까지 상상을 해야한다. 모델에 찍신이 강림해야한다.
# 총 4개를 예측한다. 빈공간, 도로, 자동차, 장애물

# 배치가 8이면 out of memory가 뜨고
# 배치가 4이면 out of memory가 뜨고
# 배치가 2이면 10초 걸린다, 이제 4초 걸린다.
# 배치가 1이면 0.7초 걸린다

# resnet50을 18로 낮추고 나서는
# 배치가 8이면 16초 걸리고
# 배치가 4이면 3.5초 걸리고
# 배치가 2이면 0.7초 걸리고
# 배치가 1이면 0.5초 걸린다

# 최종 다이어트를 마치고
# 배치가 8이면 7초
# 배치가 4이면 0.64초
# 배치가 2이면 0.58초
# 배치가 1이면 0.55초

B, N, c, h, w = 4, 6, 3, 192, 624
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VoxFormer(d_model=128, pred_num=4, image_height=192, image_width=624).to(device)
model.load_state_dict(torch.load(r'model_save\model_epoch_60.pth', weights_only=True))
model.eval()

test_image = torch.randn(B, N, c, h, w).to(device)
test_intrinsics = torch.randn(B, N, 3, 3).to(device)
test_rots= torch.randn(B, N, 3, 3).to(device)
test_trans = torch.randn(B, N, 3).to(device)

start_time = time.time()
outputs, M_out = model(test_image, test_intrinsics, test_rots, test_trans)
end_time = time.time()
print(f'걸린 시간 : {end_time - start_time:.5f}')

print(M_out.shape) # [B, 16, 64, 64]
print(outputs.shape) # [B, 4, 16, 64, 64]