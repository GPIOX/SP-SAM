import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything.segment_anything import build_sam, SamPredictor
from segment_anything.segment_anything import sam_model_registry
from lora import LoRA_sam

# 将输入修改为512*512，并且加入LoRA
# 有问题的是把num_multimask_outputs=3改为了num_multimask_outputs=num_class
sam = sam_model_registry["vit_b"](image_size=512, checkpoint='checkpoint/sam_vit_b_01ec64.pth')
sam = sam[0]
lora_sam = LoRA_sam(sam, 512).cuda()

print(lora_sam.sam.image_encoder(torch.randn(1,3,512,512).cuda()).shape)