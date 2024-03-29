import argparse
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch

import cv2
from empatches import EMPatches
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma
import tifffile as tf

import options
import util

start = datetime.now()

print(f"Starting: {start}")

emp = EMPatches()
opt = options.Options().init(argparse.ArgumentParser(description='Image denoising')).parse_args()
model_restoration = util.get_arch(opt)

precalc_min = None
precalc_max = None

# path_chk_rest = opt.pretrain_weights
# print("Resume from "+path_chk_rest)
# util.load_checkpoint(model_restoration, path_chk_rest)

class CTInferenceDataset(Dataset):
	def __init__(self, patch_set):	  
		self.__images = patch_set
		
	def __len__(self):
		return len(self.__images)

	def __getitem__(self, index):
		noisy_img = self.__images[index]
		noisy_img = self.__transform(noisy_img)
		return noisy_img

	def __transform(self, img):
		img = torch.Tensor(img)
		# Make it 3D for analysis and to allow batching.
		img = img.unsqueeze(0)
		return img.cuda()

def dataload(source: Dataset, batch_size: int = 1, shuffle: bool = False):
	return DataLoader(source, batch_size=batch_size, shuffle=shuffle)

def time_gap():
	return datetime.now() - start

def load_checkpoint(model, weights):
	checkpoint = torch.load(weights)
	try:
		model.load_state_dict(checkpoint["state_dict"])
	except:
		state_dict = checkpoint["state_dict"]
		new_state_dict = OrderedDict()
		for k, v in state_dict.items():
			name = k[7:] if 'module.' in k else k
			new_state_dict[name] = v
		model.load_state_dict(new_state_dict)
	return model


def minmaxscale(img, minval=None, maxval=None):
	img = img.astype('float32')
	if minval is None:
		minval = np.min(img)
	if maxval is None:
		maxval = np.max(img)
	norm = (img - minval) / (maxval - minval)
	return norm


def remove_outlier(img):
	a_sigma_est = estimate_sigma(img, channel_axis=None, average_sigmas=True)
	asig = denoise_nl_means(img, patch_size=9, patch_distance=5,
							fast_mode=True, sigma=0.001 * a_sigma_est,
							preserve_range=False, channel_axis=None)
	return asig


def preprocess(img, scale, minval=None, maxval=None):
	img = minmaxscale(img, minval, maxval)
	img = remove_outlier(img)
	m, n = img.shape
	img = cv2.resize(img, (n, int(m * scale)), interpolation=cv2.INTER_CUBIC)
	return img


pathtosinograms = Path("D:\\", "Valid_Sino")
allsinograms = pathtosinograms.glob('**/*.tif')

storepath = Path("D:\\", "restored_images")
storepath.mkdir(parents=True, exist_ok=True)

patch_size = 256
upscale = 4
path_to_trained_model = Path(opt.save_dir).joinpath("log_custom_Drht", "models", "model_best.pth")

# path_to_trained_model = './lgos/uformer/customzebra/denoising/custom/Drht_/models/model_best.pth'
model_restoration = load_checkpoint(model_restoration, path_to_trained_model)
model_restoration = model_restoration.cuda()

batch_size = 150

print(f"model loaded {time_gap()}")
with torch.no_grad():
	model_restoration.eval()
	for item in allsinograms:
		if item.is_file():
			input_image = tf.imread(item)
			input_image = preprocess(input_image, upscale, precalc_min, precalc_max)
			print(f"PreProcessed {time_gap()}")

			patches, indices = emp.extract_patches(input_image, patchsize=patch_size, overlap=0.5)
			ds = CTInferenceDataset(patches)
			print(f"Dataset Created {len(ds)} {time_gap()}")

			collect = []
			print("restored patch 0", "/", str(np.ceil(len(ds) / batch_size)), str(time_gap()), end='')
			for i, (patch) in enumerate(dataload(ds, batch_size=batch_size)):
				with torch.cuda.amp.autocast():
					restored, _, _, _ = model_restoration(patch)
					print("\rrestored patch ", str(i), "/", str(np.ceil(len(ds) / batch_size)), str(time_gap()), end='')
				collect += [x.squeeze(0).detach().cpu().numpy() for x in restored]
			print(f"collected {len(collect)} {time_gap()}")
			merged_img = emp.merge_patches(collect, indices, mode='avg')
			tf.imsave(storepath.joinpath(item.name.replace('.tif', '_sr.tif')), merged_img)

	print(f"finished {time_gap()}")
