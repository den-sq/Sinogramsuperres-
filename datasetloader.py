import numpy as np
import os
from pathlib import Path
from torch.utils.data import Dataset

import tifffile as tl
from skimage.restoration import denoise_nl_means, estimate_sigma
import math
from torchvision import transforms
from glob import glob
import cv2


##################################################################################################
class DataLoaderTrain(Dataset):
	def __init__(self, sino_dir, upscale=4, target_size=256):
		super(DataLoaderTrain, self).__init__()

		# expects parent path to have folders of individual fish sinograms
		self.clean_files = [str(x) for x in Path(sino_dir).glob("**/*.tif*")]
		print(f"{sino_dir} - {len(list(self.clean_files))}")

		# self.tar_size = len(self.clean_filenames)  # get the size of target
		self.trans = transforms.Compose([transforms.ToTensor()])
		self.target_size = target_size
		self.upscale = upscale

	def __len__(self):
		return len(self.clean_files)

	def minmaxscale(self, img):
		img = img.astype('float32')
		norm = (img - np.min(img)) / (np.max(img) - np.min(img))
		return norm

	def remove_outlier(self, img):
		a_sigma_est = estimate_sigma(img, channel_axis=None, average_sigmas=True)
		asig = denoise_nl_means(img, patch_size=9, patch_distance=5,
								fast_mode=True, sigma=0.001 * a_sigma_est,
								preserve_range=False, channel_axis=None)
		return asig

	def normalize_reshape(self, clean, noisy, scale=4):
		clean = self.minmaxscale(clean)
		noisy = self.minmaxscale(noisy)
		m, n = clean.shape
		randomx = np.random.randint(0, m - self.target_size - 5)
		randomy = np.random.randint(0, n - self.target_size - 5)
		clean = clean[randomx:randomx + self.target_size, randomy:randomy + self.target_size]
		noisy = noisy[randomx:randomx + self.target_size, randomy:randomy + self.target_size]
		clean = self.remove_outlier(clean)
		noisy = self.remove_outlier(noisy)
		reduction = int(math.log(scale, 2))
		for _ in range(0, reduction):
			noisy = np.delete(noisy, range(1, noisy.shape[0], 2), axis=0)
		noisy = cv2.resize(noisy, (self.target_size, self.target_size), interpolation=cv2.INTER_CUBIC)
		return clean, noisy

	def __getitem__(self, index):
		tar_index = index % len(self.clean_files)
		clean = tl.imread(self.clean_files[tar_index]).astype('float32')
		noisy = tl.imread(self.clean_files[tar_index]).astype('float32')
		clean_filename = self.clean_files[tar_index].split('/')[-1]
		noisy_filename = self.clean_files[tar_index].split('/')[-1]

		clean, noisy = self.normalize_reshape(clean, noisy, self.upscale)
		# noisy = self.normalize_reshape(noisy)

		noisy = self.trans(noisy)
		clean = self.trans(clean)
		return clean, noisy, clean_filename, noisy_filename


##################################################################################################
class DataLoaderVal(Dataset):
	def __init__(self, sino_dir, upscale='4x', target_size=256):
		super(DataLoaderVal, self).__init__()

		# expects parent path to have folders of individual fish sinograms
		self.clean_files = [str(x) for x in Path(sino_dir).glob("**/*.tif*")][-32:]
		# self.tar_size = len(self.clean_filenames)
		# get the size of target
		self.trans = transforms.Compose([transforms.ToTensor()])
		self.target_size = target_size

	def __len__(self):
		return len(self.clean_files)

	def minmaxscale(self, img):
		img = img.astype('float32')
		norm = (img - np.min(img)) / (np.max(img) - np.min(img))
		return norm

	def remove_outlier(self, img):
		a_sigma_est = estimate_sigma(img, channel_axis=None, average_sigmas=True)
		asig = denoise_nl_means(img, patch_size=9, patch_distance=5,
					h=0.1, fast_mode=True, sigma=0.001 * a_sigma_est,
					preserve_range=False, channel_axis=None)
		return asig

	def normalize_reshape(self, clean, noisy, scale=4):
		clean = self.minmaxscale(clean)
		noisy = self.minmaxscale(noisy)
		m, n = clean.shape
		randomx = np.random.randint(0, m - self.target_size - 5)
		randomy = np.random.randint(0, n - self.target_size - 5)
		clean = clean[randomx:randomx + self.target_size, randomy:randomy + self.target_size]
		noisy = noisy[randomx:randomx + self.target_size, randomy:randomy + self.target_size]
		clean = self.remove_outlier(clean)
		noisy = self.remove_outlier(noisy)
		reduction = int(math.log(scale, 2))
		for _ in range(0, reduction):
			noisy = np.delete(noisy, range(1, noisy.shape[0], 2), axis=0)
		noisy = cv2.resize(noisy, (self.target_size, self.target_size), interpolation=cv2.INTER_CUBIC)
		return clean, noisy

	def __getitem__(self, index):
		tar_index	= index % len(self.clean_files)
		clean = tl.imread(self.clean_files[tar_index]).astype('float32')
		noisy = tl.imread(self.clean_files[tar_index]).astype('float32')
		clean_filename = self.clean_files[tar_index].split('/')[-1]
		noisy_filename = self.clean_files[tar_index].split('/')[-1]

		clean, noisy = self.normalize_reshape(clean, noisy)
		# noisy = self.normalize_reshape(noisy)

		noisy = self.trans(noisy)
		clean = self.trans(clean)
		return clean, noisy, clean_filename, noisy_filename

##################################################################################################


def get_training_data(sino_dir, upscale=4, target_size=256):
	if not os.path.exists(sino_dir):
		print("Invalid training directory")
		assert os.path.exists(sino_dir)
	return DataLoaderTrain(sino_dir, upscale, target_size)


def get_validation_data(sino_dir, upscale, target_size):
	if not os.path.exists(sino_dir):
		print("Invalid validation directory")
		assert os.path.exists(sino_dir)
	return DataLoaderVal(sino_dir, upscale, target_size)
