import os
import torch
import torch.optim as optim
import glob
import random
import numpy as np
import torch.optim.lr_scheduler as lrs
import torch.utils
import torch.utils.data
from src.data_loader import train_loader, test_loader
from src.bbox import get_min_area_rect_from_mask
from src.FDA import FDA_source_to_target
from utils.loss import loss_task
from utils.log import get_logger, save_ckpt
from tqdm import tqdm
from torch.nn.functional import threshold, normalize
from metrics import IoU_bin
import argparse
import yaml
import datetime

# SAM
from segment_anything.segment_anything import build_sam, SamPredictor
from segment_anything.segment_anything import sam_model_registry
from lora import LoRA_sam, LoRA_mobilesam

CONFIG_PATH='config_vit-l.yaml'
def get_gonfig(config_file=CONFIG_PATH):
    with open(config_file,'r') as cfg:
        args = yaml.load(cfg, Loader=yaml.FullLoader)
    return args

def seed_torch(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_torch()

def main():
    # get configs
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--config_path', type=str, default='config/RMAS_vit_b.yaml')
    
    args = parser.parse_args()
    
    args = get_gonfig(config_file=args.config_path)
    train(args)


def train(args):
    exp_name = args['exp_name']
    ckpt_path = os.path.join('best', args['Train']['net'], datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(ckpt_path, exist_ok=True)
    # --------------------------------------loading data------------------------------------------
    img_name_list = glob.glob(args['Train']['tra_image_dir'] + '*' + '.jpg')
    if len(img_name_list) < 10:
        img_name_list = glob.glob(args['Train']['tra_image_dir'] + '*' + '.png')
    lab_name_list = []
    for img_path in img_name_list:
        img_name = img_path.split("/")[-1]
        imidx = img_name.split(".")[0]
        lab_name_list.append(args['Train']['tra_lbl_dir'] + imidx + '.png')  # MAS3K:png  RMAS:jpg 

    logger = get_logger(exp_name)
    logger.info("The code of this experiment has been saved to {}".format(os.path.abspath('..')))
    logger.info("---")
    logger.info(exp_name)
    logger.info("train images: {}".format(len(img_name_list)))
    logger.info("train labels: {}".format(len(lab_name_list)))
    logger.info("---")

    print("-------------------")
    print(exp_name)
    print("train images: {}".format(len(img_name_list)))
    print("train labels: {}".format(len(lab_name_list)))
    print("-------------------")

    test_img_name_list = glob.glob(args['Test']['test_image_dir'] + '*' + '.jpg')
    if len(test_img_name_list) < 10:
        test_img_name_list = glob.glob(args['Test']['test_image_dir'] + '*' + '.png')
    tes_lbl_list = []
    for img_path in test_img_name_list:
        img_name = img_path.split("/")[-1]
        imidx = img_name.split(".")[0]
        tes_lbl_list.append(args['Test']['test_lbl_dir'] + imidx + '.png')  # MAS3K:png  RMAS:jpg 


    train_dataloader = train_loader(img_name_list, lab_name_list, args['Train']['batch_size'], args['Train']['train_size'])
    test_dataloader = test_loader(test_img_name_list, args['Test']['batch_size'], args['Test']['test_size'], tes_lbl_list=tes_lbl_list)

    # ----------------------------------define model --------------------------------------------
    # define the net
    net_name = args['Train']['net']
    
    Net_dict = {
        'vit_t': 'checkpoint/mobile_sam.pt',
        'vit_b': 'checkpoint/sam_vit_b_01ec64.pth',
        'vit_l': 'checkpoint/sam_vit_l_0b3195.pth',
        'vit_h': 'checkpoint/sam_vit_h_4b8939.pth'
    }
    sam = sam_model_registry[args['Net']](image_size=512, checkpoint=Net_dict[args['Net']])[0] # Net_dict[args['Net']][0]
    net = LoRA_sam(sam, 4).cuda() if args['Net'] != 'vit_t' else LoRA_mobilesam(sam, 4).cuda()

    # net = lora_sam
    # net = torch.nn.DataParallel(net)
    # net = net.cuda()
    # net_dir = args['Test']['ckpt']
    # net.load_lora_parameters(net_dir)
    # # torch.save(net.sam.decoder.state_dict(), os.path.join(ckpt_path, f"decoder_epoch_{epoch}.pth"))
    # best_other = torch.load(os.path.join(args['Test']['decoder']))
    # net.sam.prompt_change.load_state_dict(best_other['prompt_change'])
    # net.sam.adapter.load_state_dict(best_other['adapter'])
    # net.sam.prompt_encoder.load_state_dict(best_other['prompt_encoder'])
    # net.sam.mask_decoder.load_state_dict(best_other['mask_decoder'])
    # net = net.cuda()


    loss_mse = torch.nn.MSELoss()

    logger.info("Total number of net paramerters {}".format(sum(x.numel() for x in net.parameters())))

    # ------------------------------ define optimizer ------------------------------------------
    print("---Define Optimizer---")
    optimizer = optim.Adam(net.parameters(), lr=args['Train']['optimizer_lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    milestones = []
    for i in range(1, args['Train']['epoch_num']):
        if i % 100 == 0:
            milestones.append(i)
    scheduler = lrs.MultiStepLR(optimizer, milestones, 0.5)


    # ======================================= training process ===================================================

    logger.info("---Start Training---")
    print("---Start Training---")
    ite_num = 0
    running_loss = 0.0
    ite_num4val = 0
    best_miou = 0
    best_epoch = 0

    # ----------------------------------------training stage----------------------------------------------------------
    for epoch in range(0, args['Train']['epoch_num']):
        net.train()

        bar_format = '{desc}{percentage:3.0f}%|{bar}|{n_fmt:.5s}/{total_fmt}[{elapsed}<{remaining}{postfix}]'
        loop = tqdm(enumerate(train_dataloader, 1), total=len(train_dataloader), ncols=100, bar_format=bar_format)

        for i, data in loop:
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1
            inputs, labels, label_, fft_result_shifted = data[0], data[1], data[2], data[3]

            inputs = inputs.cuda()
            labels = labels.cuda()
            fft_result_shifted = fft_result_shifted.cuda()
            bbox_list = []
            for m in label_:
                bbox = get_min_area_rect_from_mask((m.numpy()*255).astype(np.uint8))
                bbox = torch.tensor(bbox).cuda()
                bbox_list.append(bbox)

            output_dict = net.sam(inputs, multimask_output=2, image_size=512, fft_result_shifted=fft_result_shifted)
            output = torch.softmax(output_dict['masks'], dim=1)

            loss = loss_task(output, labels.to(torch.float))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()           
            
 
    # ---------------------------------------logging info-----------------------------------------------------
            # logger.info("%s  :[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, lr: %6f" % (net_name,
            # epoch + 1, args['Train']['epoch_num'], (i + 1) * args['Train']['batch_size'], 
            # len(img_name_list), ite_num, running_loss / ite_num4val, optimizer.param_groups[0]['lr']))  # writing to log file

            loop.set_description(f"{ net_name } : Epoch [{ epoch+1 }/{ args['Train']['epoch_num'] }]")  # Set the description of tqdm bar
            loop.set_postfix(train_loss=f'{running_loss / ite_num4val:.3f}')   # Set the postfix of tqdm bar, edit as needed
            
            del loss

        scheduler.step()
    # ---------------------------------------save checkpoint--------------------------------------------------
        
        mIoU = eval_model(net, test_dataloader)
        logger.info("%s  :[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, lr: %6f" % (net_name,
            epoch + 1, args['Train']['epoch_num'], (i + 1) * args['Train']['batch_size'], 
            len(img_name_list), ite_num, running_loss / ite_num4val, optimizer.param_groups[0]['lr']))  # writing to log file

        if mIoU > best_miou:  
            best_miou = mIoU
            best_epoch = epoch
            logger.info(f"{net_name}  :[epoch: {epoch:3d}, MIoU {mIoU:.3f}")  # writing to log file
            # ckpt = {
            #         "net": net.state_dict(),
            #         "current_epoch": epoch,
            #         "ite_num": ite_num
            #         }
            # save_ckpt(ckpt, name=net_name, epoch=epoch)
            # net.save_lora_parameters(os.path.join(ckpt_path, f"epoch_{epoch}.pth"))
            net.save_lora_parameters(os.path.join(ckpt_path, f"best_lora_{epoch}.pth"))
            ckpt = {
                "prompt_change": net.sam.prompt_change.state_dict(),
                "adapter": net.sam.adapter.state_dict(),
                "mask_decoder": net.sam.mask_decoder.state_dict(),
                "prompt_encoder": net.sam.prompt_encoder.state_dict(),
            }
            torch.save(ckpt, os.path.join(ckpt_path, f"best_other_{epoch}.pth"))
            # torch.save(net.sam.decoder.state_dict(), os.path.join(ckpt_path, f"decoder_epoch_{epoch}.pth"))
            print("Saved best checkpoint at epoch:", epoch)
            # net.save_lora_parameters(os.path.join(ckpt_path, f"epoch_{epoch}.pth"))

            running_loss = 0.0
            ite_num4val = 0

        loop.close()

    print('\033[1;33m-------------Congratulations! Training Done!!!-------------\033[0m')
    logger.info(f'Best epoch : {best_epoch}' )
    logger.info(f'Best mIou  : {best_miou:.3f}')
    logger.info('-------------Congratulations! Training Done!!!-------------')

def eval_model(model: LoRA_sam, test_dataloader: torch.utils.data.DataLoader):
    model.eval()
    loop = tqdm(enumerate(test_dataloader, 1), total=len(test_dataloader), ncols=100, desc='Eval')
    IOU = []
    for i_test, data_test in loop:
        inputs_test = data_test[0]
        inputs_test = inputs_test
        label = data_test[1].squeeze().squeeze().cpu().numpy()
        fft_result_shifted = data_test[3]
	
        if torch.cuda.is_available():
            inputs_test = inputs_test.cuda()
            fft_result_shifted = fft_result_shifted.cuda()
        else:
            inputs_test = inputs_test

        output_dict = model.sam(inputs_test, multimask_output=2, image_size=512, fft_result_shifted=fft_result_shifted)
        pred = (torch.softmax(output_dict['masks'], dim=1)).argmax(dim=1)
        pred = pred.squeeze().cpu().detach().numpy()

        label, pred = np.uint8(label*255), np.uint8(pred*255)
        intersection = np.logical_and(label, pred)
        union = np.logical_or(label, pred)
        iou_score = 1.0 * np.sum(intersection) / np.sum(union)
        IOU.append(iou_score)

    model.train()
    mIoU = np.mean(IOU)
    print(f'Mean IoU: {mIoU}')
    return mIoU

if __name__ == '__main__':
    main()