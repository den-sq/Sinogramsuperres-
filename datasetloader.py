from copy import deepcopy

import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

import tifffile as tf
from skimage.restoration import denoise_nl_means, estimate_sigma
import math
from torchvision import transforms
from glob import glob
import cv2

def memmap_helper(image, i_dtype, offsets, width):
	""" Sinogram order-capable reader using direct buffer reading.

		:param image: Path to image to read from.
		:param i_dtype: Data type of image.
		:param offsets: Array of memory offsets to read into.  Should be sequential in file.
		:param width: Width of chunk of memory to read per offset.
	"""
	size = width * np.dtype(i_dtype).itemsize

	target_array = np.empty((len(offsets), width), dtype=i_dtype)

	def set_array(offset):
		target_array[offset[0], :] = np.memmap(image, dtype=i_dtype, mode="r+", offset=offset[1], shape=width, order='C').copy()

	map(set_array, enumerate(offsets))
	
	return target_array

class SinglePatchDataset(Dataset):
	def __init__(self, sino_dir, run_normalization=True, bound_vals=None, subset_start=None, upscale=4, target_size=256):
		super(SinglePatchDataset, self).__init__()

		# expects parent path to have folders of individual fish sinograms
		self.clean_files = [x for x in Path(sino_dir).glob("**/*.tif*")][subset_start:]
		print(f"{sino_dir} - {len(list(self.clean_files))}")

		self.trans = transforms.Compose([transforms.ToTensor()])
		self.target_size = target_size
		self.upscale = upscale
		self.__bounds = bound_vals
		self.__run_normalization = run_normalization

	def __len__(self):
		return len(self.clean_files)

	def scale_from(self, img, bounds):
		return (img - bounds[0]) / (bounds[1] - bounds[0])

	def remove_outlier(self, img):
		a_sigma_est = estimate_sigma(img, channel_axis=None, average_sigmas=True)
		asig = denoise_nl_means(img, patch_size=9, patch_distance=5,
								fast_mode=True, sigma=0.001 * a_sigma_est,
								preserve_range=False, channel_axis=None)
		return asig

	def get_target_area(self, shape):
		y_start = np.random.randint(0, shape[0] - self.target_size - 5)
		x_start = np.random.randint(0, shape[1] - self.target_size - 5)
		return np.s_[y_start:y_start + self.target_size, x_start:x_start + self.target_size]

	def normalize_reshape(self, clean, noisy, target_area, bounds, scale=4):
		if self.__run_normalization:
			clean = self.scale_from(clean, bounds)
			noisy = self.scale_from(noisy, bounds)
			clean = self.remove_outlier(clean)
			noisy = self.remove_outlier(noisy)

		reduction = int(math.log(scale, 2))
		for _ in range(0, reduction):
			noisy = np.delete(noisy, range(1, noisy.shape[0], 2), axis=0)
		noisy = cv2.resize(noisy, (self.target_size, self.target_size), interpolation=cv2.INTER_CUBIC)
		return clean, noisy

	def __getitem__(self, index):
		tar_index = index % len(self.clean_files)
		with tf.TiffFile(self.clean_files[tar_index]) as clean_file:
			shape = clean_file.pages[0].shape
			data_type = clean_file.pages[0].dtype
			offset = clean_file.pages[0].dataoffsets[0]
			target_area = self.get_target_area(shape)

		if (self.__bounds is None) and self.__run_normalization:
			clean = tf.imread(self.clean_files[tar_index]).astype('float32')
			noisy = np.copy(clean)

			bounds = [np.min(clean), np.max(clean)]

			clean = clean[target_area]
			noisy = noisy[target_area]
		else:
			# Come back for speedup later.
			clean = tf.imread(self.clean_files[tar_index]).astype('float32')
			noisy = np.copy(clean)

			clean = clean[target_area]
			noisy = noisy[target_area]

			# Calculate offsets to just read what you need from file.
			# offset_base_pixels = target_area[0].start * shape[1] + target_area[1].start  # Start point for target area.
			# offset_set = [offset + data_type.itemsize * (offset_base_pixels + shape[1] * i) for i in range(self.target_size)]

			# Copy / deepcopy didn't work for some reason.
			# clean = memmap_helper(self.clean_files[tar_index], data_type, offset_set, self.target_size).astype('float32')
			# noisy = memmap_helper(self.clean_files[tar_index], data_type, offset_set, self.target_size).astype('float32')
			
			bounds = self.__bounds

		clean, noisy = self.normalize_reshape(clean, noisy, target_area, bounds, self.upscale)

		noisy = self.trans(noisy)
		clean = self.trans(clean)
		return clean, noisy, self.clean_files[tar_index].name, self.clean_files[tar_index].name

