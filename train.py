import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from UnityDataset import UnityDataset
from VoxFormer import VoxFormer
from LossFunction import LossFunction
import os
import time

Epochs = 20
learning_rate = 0.00001
batch_size = 4
image_height = 192
image_width = 624
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

save_interval = 5
model_save_path = r'./model_save'
if not os.path.exists(model_save_path): os.makedirs(model_save_path)

root_dirs = [r'C:\Users\MSI\Desktop\DrivingData\data_1', r'C:\Users\MSI\Desktop\DrivingData\data_2', r'C:\Users\MSI\Desktop\DrivingData\data_3', r'C:\Users\MSI\Desktop\DrivingData\data_4']
full_dataset = UnityDataset(root_dirs=root_dirs)
datasetSize = len(full_dataset)
train_size = int(0.8 * datasetSize)
test_size = datasetSize - train_size

train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader( # 학습용 데이터로더
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_loader = DataLoader( # 테스트용 데이터로더
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

model = VoxFormer(d_model=128, pred_num=4, image_height=image_height, image_width=image_width).to(device)
model.load_state_dict(torch.load(r'model_save\model_epoch_60.pth', weights_only=True))

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
class_weight = torch.tensor([1.0, 2.0, 2.0, 4.0]).to(device) # 빈공간, 도로, 자동차, 장애물
other_weight = torch.tensor([0.0, 1.0, 1.0, 3.0]).to(device) # 빈공간, 도로, 자동차, 장애물
criterion = LossFunction(class_weight=class_weight, other_weight=other_weight)

scheduler = optim.lr_scheduler.StepLR(optimizer, 20, 0.5)

def train():
    best_avg_test_loss = float('inf')
    print('학습 시작')
    for epoch in range(Epochs + 1):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0

        batch_start_time = time.time()
        for batch_idx, batch in enumerate(train_loader):
            imgs = batch['imgs'].to(device)
            rots = batch['rots'].to(device)
            trans = batch['trans'].to(device)
            intrins = batch['intrinsics'].to(device)
            labels = batch['label_3d'].to(device).long() # [B, 16, 64, 64]

            optimizer.zero_grad()
            pred_stage1, pred_stage2 = model(imgs, intrins, rots, trans)
            
            total_loss, loss_occ, loss_sem = criterion(pred_stage1, pred_stage2, labels)
            total_loss.backward()

            optimizer.step()

            train_loss += total_loss.item()

            if batch_idx % 10 == 0:
                batch_end_time = time.time()
                print(f'Epoch [{epoch}/{Epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss_total : {total_loss.item():.4f} Loss_occ : {loss_occ.item():.4f} Loss_sem : {loss_sem.item():.4f} Time : {batch_end_time-batch_start_time:.4f}')
                batch_start_time = time.time()

            

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                imgs = batch['imgs'].to(device)
                rots = batch['rots'].to(device)
                trans = batch['trans'].to(device)
                intrins = batch['intrinsics'].to(device)
                labels = batch['label_3d'].to(device).long() # [B, 16, 64, 64]

                pred_stage1, pred_stage2 = model(imgs, intrins, rots, trans)
            
                total_loss, loss_occ, loss_sem = criterion(pred_stage1, pred_stage2, labels)

                test_loss += total_loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        epoch_end_time = time.time()
        print(f'==> Epoch {epoch} 완료 Train Loss : {avg_train_loss:.4f} Test Loss : {avg_test_loss:.4f} Epoch Time : {epoch_end_time-epoch_start_time:.4f}')
        epoch_start_time = time.time()

        scheduler.step()

        if epoch % save_interval == 0:
            save_path = os.path.join(model_save_path, f'model_epoch_{epoch+60}.pth')
            torch.save(model.state_dict(), save_path)
            print(f'Saved : {save_path}')
        
        if avg_test_loss < best_avg_test_loss:
            best_avg_test_loss = avg_test_loss
            save_path = os.path.join(model_save_path, f'model_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f'New Best Model Saved! Loss : {best_avg_test_loss:.4f}')


        print('-' * 50)

if __name__ == "__main__":
    train()