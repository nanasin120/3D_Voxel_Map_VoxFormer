import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os

class UnityDataset(Dataset):
    def __init__(self, root_dirs):
        self.root_dir = root_dirs
        self.all_frames = []

        for root_dir in root_dirs:
            ext_df = pd.read_csv(os.path.join(root_dir, 'extrinsics.csv'))
            int_df = pd.read_csv(os.path.join(root_dir, 'intrinsics.csv'))

            f_ids = sorted(ext_df['frame_id'].unique())

            for f_id in f_ids:
                ext_rows = ext_df[ext_df['frame_id'] == f_id].sort_values('camera_id').values
                int_rows = int_df[int_df['frame_id'] == f_id].sort_values('camera_id').values
                
                self.all_frames.append({
                    'root': root_dir,
                    'frame_id': f_id,
                    'ext_rows': ext_rows,
                    'int_rows': int_rows
                })

    def __len__(self):
        return len(self.all_frames)
    
    def __getitem__(self, idx):
        data_info = self.all_frames[idx]
        root = data_info['root']
        f_id = data_info['frame_id']
        ext_rows = data_info['ext_rows']
        int_rows = data_info['int_rows']

        imgs, rots, trans, intrins = [], [], [], []

        for i in range(6):
            img_path = os.path.join(root, 'images', f'frame_{f_id:06d}_cam_{i}.jpg')
            img = Image.open(img_path).convert('RGB')
            img = np.array(img).transpose(2, 0, 1)
            imgs.append(img)

            r = [[ext_rows[i, 2], ext_rows[i, 3], ext_rows[i, 4]],
                [ext_rows[i, 6], ext_rows[i, 7], ext_rows[i, 8]],
                [ext_rows[i, 10], ext_rows[i, 11], ext_rows[i, 12]]]

            t = [ext_rows[i, 5], ext_rows[i, 9], ext_rows[i, 13]]
            rots.append(r)
            trans.append(t)

            it = int_rows[i]
            k = [[it[2], 0,     it[4]],
                [0,     it[6], it[7]],
                [0,     0,     1]]
            intrins.append(k)

        voxel_path = os.path.join(root, 'voxels', f"frame_{f_id:06d}_voxel.bin")
        label_3d = np.fromfile(voxel_path, dtype=np.uint8).reshape(64, 16, 64) # 좌우, 상하, 앞뒤
        label_3d = np.transpose(label_3d, (1, 0, 2)) # 상하, 좌우, 앞뒤 16, 64, 64

        return {
            'imgs': torch.from_numpy(np.stack(imgs)).float() / 255.0,
            'rots': torch.as_tensor(rots, dtype=torch.float32),
            'trans': torch.as_tensor(trans, dtype=torch.float32),
            'intrinsics': torch.as_tensor(intrins, dtype=torch.float32),
            'label_3d': torch.from_numpy(label_3d.copy()).float() # 16, 64, 64
        }