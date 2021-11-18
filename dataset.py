import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms

class DeblurDataset(Dataset):
    def __init__(self, root_dir, blur_image_files, sharp_image_files, rotation=False, adjust=False, crop=False, crop_size=256,
                 transform=None):
        self.root_dir = root_dir
        blur_file = open(blur_image_files, 'r')
        blur_images = []
        sharp_file = open(sharp_image_files, 'r')
        sharp_images = []
        for blur_line in blur_file:
            blur_line = blur_line.rstrip()
            image_name = blur_line.split('/')
            image_path = self.root_dir
            for i_path in image_name:
                image_path = os.path.join(image_path, i_path)
            blur_images.append(image_path)
        for sharp_line in sharp_file:
            sharp_line = sharp_line.rstrip()
            image_name = sharp_line.split('/')
            image_path = self.root_dir
            for i_path in image_name:
                image_path = os.path.join(image_path, i_path)
            sharp_images.append(image_path)
        self.blur_images = blur_images
        self.sharp_images = sharp_images
        self.rotation = rotation
        self.adjust = adjust
        self.crop = crop
        self.crop_size = crop_size
        self.transform = transform

    def __len__(self):
        return len(self.blur_images)

    def __getitem__(self, index):
        blur_image = Image.open(self.blur_images[index]).convert('RGB')
        sharp_image = Image.open(self.sharp_images[index]).convert('RGB')
        if self.rotation:
            degree = random.choice([90, 180, 270, 360])
            blur_image = transforms.functional.rotate(blur_image, degree, expand=True)
            sharp_image = transforms.functional.rotate(sharp_image, degree, expand=True)

        if self.adjust:
            blur_image = transforms.functional.adjust_gamma(blur_image, 1)
            sharp_image = transforms.functional.adjust_gamma(sharp_image, 1)
            hue_factor = 1 + 0.5 - np.random.rand()
            blur_image = transforms.functional.adjust_saturation(blur_image, hue_factor)
            sharp_image = transforms.functional.adjust_saturation(sharp_image, hue_factor)

        if self.transform:
            blur_image = self.transform(blur_image)
            sharp_image = self.transform(sharp_image)

        if self.crop:
            height = blur_image.size()[1]
            width = blur_image.size()[2]

            new_h = np.random.randint(height-self.crop_size)
            new_w = np.random.randint(width-self.crop_size)

            blur_image = blur_image[:, new_h:new_h+self.crop_size, new_w:new_w+self.crop_size]
            sharp_image = sharp_image[:, new_h:new_h+self.crop_size, new_w:new_w+self.crop_size]

        return sharp_image, blur_image