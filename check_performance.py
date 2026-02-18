import torch
import torch.nn as nn
import torch.nn.functional as F
from VoxFormer import VoxFormer
from drawVoxel import drawVoxel
from UnityDataset import UnityDataset
from LossFunction import LossFunction
import random
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VoxFormer(d_model=128, pred_num=4, image_height=192, image_width=624).to(device)
model.load_state_dict(torch.load(r'model_save\model_epoch_75.pth', weights_only=True))
model.eval()

root_dirs = [r'C:\Users\MSI\Desktop\DrivingData\data_1', r'C:\Users\MSI\Desktop\DrivingData\data_2', r'C:\Users\MSI\Desktop\DrivingData\data_3', r'C:\Users\MSI\Desktop\DrivingData\data_4']
full_dataset = UnityDataset(root_dirs)
sample_idx = random.randint(0, len(full_dataset))
sample = full_dataset[sample_idx]

imgs = sample['imgs'].unsqueeze(0).to(device)
rots = sample['rots'].unsqueeze(0).to(device)
trans = sample['trans'].unsqueeze(0).to(device)
intrins = sample['intrinsics'].unsqueeze(0).to(device)
label = sample['label_3d'].to(device).long()

class_weight = torch.tensor([1.0, 2.0, 2.0, 4.0]).to(device)
criterion = LossFunction(class_weight=class_weight)

threshold = 0.5

with torch.no_grad():
    pred_1, pred_2 = model(imgs, intrins, rots, trans) 
    _, H, W, D = pred_2[0].shape
    pred_voxel = torch.argmax(pred_2[0], dim=0).cpu().numpy()
    # probs = F.softmax(pred_2[0], dim=0)
    # pred_voxel = np.zeros((H, W, D))

    # mask_road = probs[1] > 0.5
    # pred_voxel[mask_road.cpu().numpy()] = 1

    # mask_car = probs[2] > 0.5
    # pred_voxel[mask_car.cpu().numpy()] = 2

    # mask_obstacle = probs[3] > threshold
    # pred_voxel[mask_obstacle.cpu().numpy()] = 3

    total_loss, loss_occ, loss_sem = criterion(pred_1, pred_2, label.unsqueeze(0))
    print(f'total_loss : {total_loss:.4f} loss_occ : {loss_occ:.4f} loss_sem : {loss_sem:.4f}')

    drawVoxel(label.cpu().numpy())
    drawVoxel(pred_voxel)