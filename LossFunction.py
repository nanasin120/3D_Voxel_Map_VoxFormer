import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss_CrossEntropy(nn.Module):
    def __init__(self, c_weight=None, d_weight = None, num_classes = 4):
        super(DiceLoss_CrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.crossEntropy = nn.CrossEntropyLoss(weight=c_weight, ignore_index=255)
        
        if d_weight == None: d_weight = [0.0, 1.0, 1.0, 3.0]
        if not isinstance(d_weight, torch.Tensor): d_weight = torch.tensor(d_weight).float()
        self.d_weight = d_weight

    def forward(self, preds, targets):
        # pred : [B, 4, 32, 64, 64]
        # target : [B, 32, 64, 64]
        ce_loss = self.crossEntropy(preds, targets) 

        pred_probs = F.softmax(preds, dim=1) # 4개 있는걸 확률값으로 변경
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).to(preds.device) # [B, 32, 64, 64, 4]로 변함
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float() # [B, 4, 32, 64, 64]로 변함
        
        dims = (2, 3, 4)

        TP = torch.sum(pred_probs * targets_one_hot, dim=dims).to(preds.device)
        FP = torch.sum(pred_probs * (1 - targets_one_hot), dim=dims).to(preds.device)
        FN = torch.sum((1 - pred_probs) * targets_one_hot, dim=dims).to(preds.device)

        alpha = 0.5
        beta = 0.5

        tversky_score = (TP + 1e-6) / (TP + alpha * FP + beta * FN + 1e-6)
        tversky_loss = torch.mean((1. - tversky_score) * self.d_weight.to(preds.device))

        return 0.5 * ce_loss + 0.5 * tversky_loss

        # intersection = torch.sum(pred_probs * targets_one_hot, dims).to(preds.device) # 4
        # cardinality = torch.sum(pred_probs + targets_one_hot, dims).to(preds.device) # 4

        # dice_score = (2. * intersection + 1e-6) / (cardinality + 1e-6) # 모두를 따로 계산하는것, 한꺼번에 하는게 아니라

        # dice_loss = torch.mean((1. - dice_score) * self.d_weight.to(preds.device))

        # return 0.1 * ce_loss + 0.9 * dice_loss

class LossFunction(nn.Module):
    def __init__(self, class_weight=None, other_weight=None):
        super(LossFunction, self).__init__()
        self.Dice_Cross_Entropy = DiceLoss_CrossEntropy(class_weight, other_weight)

    def forward(self, pred_stage1, pred_stage2, targets):
        # pred_stage1 : [B, 16, 32, 32]
        # pred_stage2 : [B, 4, 16, 64, 64]
        # targets : [B, 16, 64, 64]

        target_binary = (targets > 0).float()
        target_occ = F.max_pool3d(target_binary.unsqueeze(1), kernel_size=(1, 2, 2), stride=(1, 2, 2)).squeeze(1)

        loss_occ = F.binary_cross_entropy(pred_stage1, target_occ)
        loss_sem = self.Dice_Cross_Entropy(pred_stage2, targets)

        total_loss = loss_occ + loss_sem

        return total_loss, loss_occ, loss_sem