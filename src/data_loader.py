import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
import torch.fft as fft
from .bbox import get_min_area_rect_from_mask

def transform(train_size):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((train_size, train_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.RandomAffine(20),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.ToTensor(),
        # transforms.ToPILImage(),
    ])

def transform_sod():
    return transforms.Compose([
        transforms.ColorJitter(0.5,0.5,0.5,0.5)
        # transforms.RandomApply([transforms.ColorJitter(0.5,0.5,0.5,0.5)], p=0.8)
    ])

def transform_test(test_size):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((test_size, test_size), interpolation=transforms.InterpolationMode.NEAREST),
    ])

class Dataset_train(Dataset):
    def __init__(self, img_name_list, lbl_name_list, train_size):
        super(Dataset_train, self).__init__()
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.train_size = train_size
        self.transform = transform(self.train_size)
        self.sod = transform_sod()

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_name_list[idx], cv2.IMREAD_UNCHANGED)
        label = cv2.imread(self.label_name_list[idx], 0)

        if self.transform:
            seed = np.random.randint(3407)
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.transform(image)
            random.seed(seed)
            torch.manual_seed(seed)
            label = self.transform(label)
            label_ = label.permute(1, 2, 0)
        # bbox = get_min_area_rect_from_mask((label_.numpy()*255).astype(np.uint8))

        fft_result = fft.fft2(image)
        # 将零频率分量移到图像中心
        fft_result_shifted = fft.fftshift(fft_result, dim=(-2, -1))


        return image, label, label_, fft_result_shifted

class Dataset_test(Dataset):
    def __init__(self, img_name_list, lbl_name_list, test_size):
        super(Dataset_test, self).__init__()
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.test_size=test_size
        self.transform = transform_test(self.test_size)

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_name_list[idx], cv2.IMREAD_UNCHANGED)
        label = cv2.imread(self.label_name_list[idx], cv2.IMREAD_GRAYSCALE)
        _,label=cv2.threshold(label,127,255,cv2.THRESH_BINARY)

        if self.transform:
            image = self.transform(image)
            # image = self.image_final_transform(image)
            label = self.transform(label)
            label_ = label.permute(1, 2, 0)
            # bbox = get_min_area_rect_from_mask((label_.numpy()*255).astype(np.uint8))

        fft_result = fft.fft2(image)
        # 将零频率分量移到图像中心
        fft_result_shifted = fft.fftshift(fft_result, dim=(-2, -1))

        return image, label, label_, fft_result_shifted

def train_loader(tra_image_list, tra_lbl_list, batch_size_train, train_img_size):
    self_dataset = Dataset_train(img_name_list=tra_image_list, lbl_name_list=tra_lbl_list, train_size=train_img_size)
    train_dataloader = DataLoader(self_dataset, batch_size=batch_size_train, shuffle=True, num_workers=8, drop_last=True)
    return train_dataloader

def test_loader(test_image_list, batch_size_val, test_img_size, **kwargs):
    self_dataset_test = Dataset_test(img_name_list=test_image_list, lbl_name_list=kwargs['tes_lbl_list'], test_size=test_img_size)
    test_dataloader = DataLoader(self_dataset_test, batch_size=batch_size_val, shuffle=False, num_workers=8, drop_last=False)
    return test_dataloader