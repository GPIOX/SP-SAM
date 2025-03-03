import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.pytorch_ssim
import utils.pytorch_iou
ssim_loss = utils.pytorch_ssim.SSIM(window_size=7, size_average=True)
iou_loss = utils.pytorch_iou.IOU(size_average=True)

bce_loss = nn.BCELoss()



def bce_ssim_loss(pred, target):
    # 将target转换为one-hot编码
    target = torch.nn.functional.one_hot(target.squeeze().long(), num_classes=2).permute(0,3,1,2).float()
    # pred = pred.unsqueeze(dim=1)
    bce_out = bce_loss(pred, target)
    ssim_out = ssim_loss(pred.argmax(dim=1).unsqueeze(dim=1).float().cpu(), target.argmax(dim=1).unsqueeze(dim=1).float().cpu())
    iou_out = iou_loss(pred, target)
    loss = bce_out + iou_out + (1-ssim_out)

    return loss


def loss_task(output, labels):
    loss = bce_ssim_loss(output, labels)
    return loss