import torch
import torch.nn as nn
import torch.nn.functional as F

class LossFunction(nn.Module):
    def __init__(self, class_weight=None, ignore_index=255):
        super(LossFunction, self).__init__()
        self.ignore_index = ignore_index
        self.crossEntropy = nn.CrossEntropyLoss(weight=class_weight, ignore_index=ignore_index)

    def forward(self, pred_stage1, pred_stage2, targets):
        # pred_stage1 : [B, 16, 32, 32]
        # pred_stage2 : [B, 4, 16, 64 , 64]
        # targets : [B, 16, 64, 64]

        # stage1 Loss
        target_occ = F.interpolate(targets.unsqueeze(1).float(), size=(16, 32, 32), mode='nearest').squeeze(1)
        target_occ = (target_occ > 0).float()
        loss_occ = F.binary_cross_entropy(pred_stage1, target_occ) # 점유했냐 이걸 확인

        loss_sem = self.crossEntropy(pred_stage2, targets) # 잘 맞추었나 이걸 확인

        total_loss = loss_occ + loss_sem

        return total_loss, loss_occ, loss_sem