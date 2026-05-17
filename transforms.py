import random
from torchvision.transforms import *
from PIL import Image
import math
from config.config import cfg
from channel_aug import ChannelT, ChannelAdapGray,ChannelExchange

class RandomErasing(object):
	""" Randomly selects a rectangle region in an image and erases its pixels.
		'Random Erasing Data Augmentation' by Zhong et al.
		See https://arxiv.org/pdf/1708.04896.pdf
	Args:
		p: The prob that the Random Erasing operation will be performed.
		sl: Minimum proportion of erased area against input image.
		sh: Maximum proportion of erased area against input image.
		r1: Minimum aspect ratio of erased area.
		mean: Erasing value.
	"""
	def __init__(self, p=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.485, 0.456, 0.406)):
		self.p = p
		self.mean = mean
		self.sl = sl
		self.sh = sh
		self.r1 = r1

	def __call__(self, img):
		if random.uniform(0, 1) >= self.p:
			return img

		for attempt in range(100):
			area = img.size()[1] * img.size()[2]

			target_area = random.uniform(self.sl, self.sh) * area
			aspect_ratio = random.uniform(self.r1, 1 / self.r1)

			h = int(round(math.sqrt(target_area * aspect_ratio)))
			w = int(round(math.sqrt(target_area / aspect_ratio)))

			if w < img.size()[2] and h < img.size()[1]:
				x1 = random.randint(0, img.size()[1] - h)
				y1 = random.randint(0, img.size()[2] - w)
				if img.size()[0] == 3:
					img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
					img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
					img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
				else:
					img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
				return img

		return img


class RectScale(object):
	def __init__(self, height, width, interpolation=Image.BILINEAR):
		self.height = height
		self.width = width
		self.interpolation = interpolation

	def __call__(self, img):
		w, h = img.size
		if h == self.height and w == self.width:
			return img
		return img.resize((self.width, self.height), self.interpolation)


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


transform_mix_aug = [transforms.ColorJitter(brightness=0.3,contrast=0.3),
					 transforms.GaussianBlur(21, sigma=(0.1, 3))]

def train_transforms_gray(H,W,normalize):
    transform_train = transforms.Compose([
            transforms.ToPILImage(),
            RectScale(cfg.H, cfg.W),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize,
            RandomErasing(p=0.5)
        ])
    return transform_train

transform_test = transforms.Compose([
	transforms.ToPILImage(),
	RectScale(cfg.H, cfg.W),
	transforms.ToTensor(),
	normalize
])
def transform_test_gray(H,W,normalize):
    transform_test_gray= transforms.Compose([
        transforms.ToPILImage(),
        RectScale(cfg.H, cfg.W),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        normalize,
    ])
    return transform_test_gray

def test_transforms(H,W,normalize):
    transform_test = transforms.Compose( [
        transforms.ToPILImage(),
        RectScale(cfg.H, cfg.W),
        transforms.ToTensor(),
        normalize])
    return transform_test

def train_transforms_color1(H,W,normalize):
    transform_train=transforms.Compose( [
            transforms.ToPILImage(),
            RectScale(cfg.H, cfg.W),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            RandomErasing(p=0.5)
            ])
    return transform_train

def train_transforms_color2(H,W,normalize):
    transform_train=transforms.Compose( [
            transforms.ToPILImage(),
            RectScale(cfg.H, cfg.W),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            RandomErasing(p=0.5),
            ChannelExchange(gray=2)
            ])
    return transform_train

def train_transforms_thermal1(H,W,normalize):
    transform_train=transforms.Compose( [
            transforms.ToPILImage(),
            RectScale(cfg.H, cfg.W),
            transforms.RandomHorizontalFlip(),
			transforms.RandomChoice(transform_mix_aug),
            transforms.ToTensor(),
            normalize,
            RandomErasing(p=0.5)
            ])
    return transform_train


def train_transforms_thermal2(H,W,normalize):
    transform_train=transforms.Compose( [
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.5),
            RectScale(cfg.H, cfg.W),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            RandomErasing(p=0.5),
            ChannelT(probability =0.5)           
            ])
    return transform_train