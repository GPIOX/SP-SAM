import os
import torch
import glob
import numpy as np
import yaml
from PIL import Image
from metrics import *
from collections import OrderedDict
from src.data_loader import test_loader
from thop import profile
from torch.nn.functional import threshold, normalize
# from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
import matplotlib.pyplot as plt

# SAM
from segment_anything.segment_anything import build_sam, SamPredictor
from segment_anything.segment_anything import sam_model_registry
from lora import LoRA_sam


CONFIG_PATH='config/UFO120_vit_l.yaml'
def get_gonfig(config_file=CONFIG_PATH):
    with open(config_file,'r') as cfg:
        args = yaml.load(cfg, Loader=yaml.FullLoader)
    return args

def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)
	dn = (d-mi)/(ma-mi)
	return dn


def feature_process(feature: torch.Tensor, size: int = 512):
	feature = torch.nn.functional.interpolate(feature, size=(512,512), mode='bilinear')
	feature = feature.mean(dim=1).squeeze(0)
	x_visualize = feature.cpu().detach().numpy()
	x_visualize = (((x_visualize - np.min(x_visualize))/(np.max(x_visualize)-np.min(x_visualize)))*255).astype(np.uint8)
	x_visualize = cv2.applyColorMap(x_visualize, cv2.COLORMAP_JET)  # 伪彩色处理   
	x_visualize = cv2.cvtColor(x_visualize, cv2.COLOR_BGR2RGB)

	return x_visualize

if __name__ == '__main__':
	# --------- 1. get image path and name ---------
	args = get_gonfig(CONFIG_PATH)

	test_img_name_list = glob.glob(args['Test']['test_image_dir'] + '*' + '.jpg')
	testset_name = args['Test']['testset_name']
	prediction_dir = 'runs/pred/' + testset_name + '/'
	if not os.path.exists(prediction_dir):
		os.makedirs(prediction_dir, exist_ok=True)

	net_dir = args['Test']['ckpt']
	# ckpt = torch.load(net_dir)

	# --------- 2. dataloader ---------
	test_img_name_list = glob.glob(args['Test']['test_image_dir'] + '*' + '.jpg')
	tes_lbl_list = []
	for img_path in test_img_name_list:
		img_name = img_path.split("/")[-1]
		imidx = img_name.split(".")[0]
		tes_lbl_list.append(args['Test']['test_lbl_dir'] + imidx + '.png')  # MAS3K:png  RMAS:jpg 

	test_dataloader = test_loader(test_img_name_list, args['Test']['batch_size'], args['Test']['test_size'], tes_lbl_list=tes_lbl_list)
	# --------- 3. model define ---------

	print("...load Net...")
	net_name = args['Train']['net']
	# sam = sam_model_registry["vit_b"](image_size=512, checkpoint='checkpoint/sam_vit_b_01ec64.pth')
	sam = sam_model_registry["vit_l"](image_size=512, checkpoint='checkpoint/sam_vit_l_0b3195.pth')
	sam = sam[0]
	net = LoRA_sam(sam, 4).cuda()
	net.load_lora_parameters(net_dir)
	# torch.save(net.sam.decoder.state_dict(), os.path.join(ckpt_path, f"decoder_epoch_{epoch}.pth"))
	best_other = torch.load(os.path.join(args['Test']['decoder']))
	net.sam.prompt_change.load_state_dict(best_other['prompt_change'])
	net.sam.adapter.load_state_dict(best_other['adapter'])
	net.sam.prompt_encoder.load_state_dict(best_other['prompt_encoder'])
	net.sam.mask_decoder.load_state_dict(best_other['mask_decoder'])

	# state = ckpt['net']
	# new_state = OrderedDict()
	# for k, v in state.items():
	# 	name = k[7:]
	# 	new_state[name] = v

	# net.load_state_dict(new_state)
	net.cuda()
	# net.eval()

	loop = tqdm(enumerate(test_dataloader), total=len(test_dataloader), ncols=100, desc='Eval')
	# --------- 4. inference for each image ---------
	for i_test, data_test in loop:
	
		# print("\rinferencing:", test_img_name_list[i_test].split("/")[-1], end='')
	
		inputs_test = data_test[0]
		inputs_test = inputs_test
		fft_result_shifted = data_test[3]
	
		if torch.cuda.is_available():
			inputs_test = inputs_test.cuda()
			fft_result_shifted = fft_result_shifted.cuda()
		else:
			inputs_test = inputs_test

		# output_dict = net.sam(inputs_test, multimask_output=2, image_size=512, fft_result_shifted=fft_result_shifted)
		# nodes, _ = get_graph_node_names(net.sam.prompt_change)
		return_nodes = {
			'fuse_conv': 'fuse_conv',
		}
		model = create_feature_extractor(net.sam.prompt_change, return_nodes=return_nodes)
		output_dict = model(net.sam.image_encoder(inputs_test, net.sam.adapter), fft_result_shifted=fft_result_shifted)
		feature = output_dict['fuse_conv']
		x_visualize = feature_process(feature)

		img = cv2.imread(test_img_name_list[i_test])
		img = cv2.resize(img, (512,512))

		mask = cv2.imread(tes_lbl_list[i_test], cv2.IMREAD_UNCHANGED)
		mask = cv2.resize(mask, (512,512))

		fig, ax = plt.subplots(1, 4, figsize=(10, 15))
		
		ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
		ax[1].imshow(mask, cmap='gray')
		ax[2].imshow(x_visualize)

		hock_feature = None
		def hock_func(module, input, output):
			global hock_feature
			hock_feature = output

		hook = net.sam.image_encoder.neck.register_forward_hook(hock_func)
		net.sam.image_encoder(inputs_test, net.sam.adapter)

		x_visualize = feature_process(hock_feature)
		ax[3].imshow(x_visualize)
		
		# 去掉各个子图的坐标轴和框
		for i in range(4):
			ax[i].axis('off')
		os.makedirs(f'temp/{testset_name}/', exist_ok=True)
		plt.savefig(f'temp/{testset_name}/'+'{}'.format(os.path.basename(test_img_name_list[i_test]).replace('jpg', 'pdf')), bbox_inches='tight', dpi=300)
		plt.close()

		# print(output_dict)
		# break
		# image_weight = 0.8
		# cam = (1 - image_weight) * x_visualize + image_weight * (img)
		# cam = cam / np.max(cam)
		# cam_image = show_cam_on_image(np.float32(img) /255, grayscale_cam.squeeze(0), use_rgb=False)
		# cv2.imwrite('temp/cam/'+os.path.basename(test_img_name_list[i_test]), (cam*255).astype(np.uint8))
		# grayscale_cam = grayscale_cam[0, :]

		# 将 grad-cam 的输出叠加到原始图像上
		# visualization = show_cam_on_image(rgb_img, grayscale_cam)


		# mask = Image.fromarray((pred).astype(np.uint8))
		# name = test_img_name_list[i_test].split("/")[-1]
		# name = name.split(".")[0]
		# mask.save(prediction_dir+'{}'.format(name+'.png')) 

