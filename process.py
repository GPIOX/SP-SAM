from segment_anything.segment_anything import sam_model_registry
import torch
from lora import LoRA_mobilesam

model_type = "vit_t"
sam_checkpoint = "checkpoint/mobile_sam.pt"
mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)[0]
mobile_sam.cuda()
mobile_sam.eval()

LoRA_mobilesam(mobile_sam, rank=4)

# output_dict = mobile_sam(torch.randn(1, 3, 512, 512).cuda(), multimask_output=2, image_size=512, fft_result_shifted=torch.randn(1, 3, 512, 512).cuda())