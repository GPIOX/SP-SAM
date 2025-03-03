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

# SAM
from segment_anything.segment_anything import build_sam, SamPredictor
from segment_anything.segment_anything import sam_model_registry
from lora import LoRA_sam
from lora import LoRA_sam, LoRA_mobilesam


# CONFIG_PATH='config.yaml'
def get_gonfig(config_file):
    with open(config_file,'r') as cfg:
        args = yaml.load(cfg, Loader=yaml.FullLoader)
    return args

def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)
	dn = (d-mi)/(ma-mi)
	return dn


if __name__ == '__main__':
	# get configs
	import argparse
	
	parser = argparse.ArgumentParser(description='Description of your program')
	parser.add_argument('--config_path', type=str, default='config file path')

	args = parser.parse_args()

	# --------- 1. get image path and name ---------
	args = get_gonfig(args.config_path)

	test_img_name_list = glob.glob(args['Test']['test_image_dir'] + '*' + '.jpg')
	if len(test_img_name_list) < 10:
		test_img_name_list = glob.glob(args['Test']['test_image_dir'] + '*' + '.png')
	testset_name = args['Test']['testset_name']
	prediction_dir = 'runs/pred/' + testset_name + '/'
	if not os.path.exists(prediction_dir):
		os.makedirs(prediction_dir, exist_ok=True)

	net_dir = args['Test']['ckpt']
	# ckpt = torch.load(net_dir)

	# --------- 2. dataloader ---------
	test_img_name_list = glob.glob(args['Test']['test_image_dir'] + '*' + '.jpg')
	if len(test_img_name_list) < 10:
		test_img_name_list = glob.glob(args['Test']['test_image_dir'] + '*' + '.png')
	tes_lbl_list = []
	for img_path in test_img_name_list:
		img_name = img_path.split("/")[-1]
		imidx = img_name.split(".")[0]
		tes_lbl_list.append(args['Test']['test_lbl_dir'] + imidx + '.png')  # MAS3K:png  RMAS:jpg 

	test_dataloader = test_loader(test_img_name_list, args['Test']['batch_size'], args['Test']['test_size'], tes_lbl_list=tes_lbl_list)
	# --------- 3. model define ---------

	print("...load Net...")
    # define the net
	net_name = args['Train']['net']

	Net_dict = {
		'vit_t': 'checkpoint/mobile_sam.pt',
		'vit_b': 'checkpoint/sam_vit_b_01ec64.pth',
		'vit_l': 'checkpoint/sam_vit_l_0b3195.pth',
		'vit_h': 'checkpoint/sam_vit_h_4b8939.pth'
	}
	sam = sam_model_registry[args['Net']](image_size=512, checkpoint=Net_dict[args['Net']])[0] # Net_dict[args['Net']][0]
	# net = LoRA_sam(sam, 4).cuda()
	net = LoRA_sam(sam, 4).cuda() if args['Net'] != 'vit_t' else LoRA_mobilesam(sam, 4).cuda()

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
	net.eval()
	IOU = []
	FM = Fmeasure()
	WFM = WeightedFmeasure()
	SM = Smeasure()
	EM = Emeasure()
	MAE = MAE()
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
		with torch.no_grad():
			output_dict = net.sam(inputs_test, multimask_output=2, image_size=512, fft_result_shifted=fft_result_shifted)
		# pred = output_dict['masks']#.argmax(dim=1)# [:, 0, :, :]# .argmax(dim=1)
		# pred = normPRED(pred)
		pred = (torch.softmax(output_dict['masks'], dim=1)).argmax(dim=1)
		# pred = normalize(threshold(output_dict['masks'] , 0.0, 0)).argmax(dim=1)

		# flops, params = profile(net, (inputs_test,))
		# print('flops: ', flops, 'params: ', params)

		# pred = d0[:, 0, :, :]
		# pred = normPRED(pred)

		pred = pred.squeeze().cpu().detach().numpy()
		label = data_test[1].squeeze().squeeze().cpu().numpy()
		label, pred = np.uint8(label*255), np.uint8(pred*255)
		intersection = np.logical_and(label, pred)
		union = np.logical_or(label, pred)
		iou_score = 1.0 * np.sum(intersection) / np.sum(union)

		IOU.append(iou_score)
		FM.step(pred=pred, gt=label)
		WFM.step(pred=pred, gt=label)
		SM.step(pred=pred, gt=label)
		EM.step(pred=pred, gt=label)
		MAE.step(pred=pred, gt=label)

		mask = Image.fromarray((pred).astype(np.uint8))
		name = test_img_name_list[i_test].split("/")[-1]
		name = name.split(".")[0]
		mask.save(prediction_dir+'{}'.format(name+'.png')) 

	fm = FM.get_results()["fm"]
	wfm = WFM.get_results()["wfm"]
	sm = SM.get_results()["sm"]
	em = EM.get_results()["em"]
	mae = MAE.get_results()["mae"]
	results = {
        "mIoU": f'{np.mean(IOU):.3f}',
        "Smeasure": f'{sm:.3f}',
        "wFmeasure": f'{wfm:.3f}',
       "meanEm": f'{em["curve"].mean():.3f}',
        "MAE": f'{mae:.3f}',

    }
	print(results)
	
	# metrics(args['Test']['test_lbl_dir'], prediction_dir)




