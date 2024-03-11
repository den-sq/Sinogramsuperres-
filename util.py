
import torch
import os
from collections import OrderedDict


def freeze(model):
	for p in model.parameters():
		p.requires_grad = False


def unfreeze(model):
	for p in model.parameters():
		p.requires_grad = True


def is_frozen(model):
	x = [p.requires_grad for p in model.parameters()]
	return not all(x)


def save_checkpoint(model_dir, state, session):
	epoch = state['epoch']
	model_out_path = os.path.join(model_dir, "model_epoch_{}_{}.pth".format(epoch, session))
	torch.save(state, model_out_path)


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


def load_checkpoint_multigpu(model, weights):
	checkpoint = torch.load(weights)
	state_dict = checkpoint["state_dict"]
	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
		name = k[7:]
		new_state_dict[name] = v
	model.load_state_dict(new_state_dict)


def load_start_epoch(weights):
	checkpoint = torch.load(weights)
	epoch = checkpoint["epoch"]
	return epoch


def load_optim(optimizer, weights):
	checkpoint = torch.load(weights)
	optimizer.load_state_dict(checkpoint['optimizer'])
	for p in optimizer.param_groups:
		lr = p['lr']
	return lr


def get_arch(opt):
	from model import Uformer, UNet

	arch = opt.arch

	print('You choose ' + arch + '...')
	if arch == 'UNet':
		model_restoration = UNet(dim=opt.embed_dim)
	elif arch == 'Uformer':
		model_restoration = Uformer(img_size=opt.train_ps, embed_dim=opt.embed_dim, win_size=8,
									token_projection='linear', token_mlp='leff', modulator=True)
	elif arch == 'Drht':
		model_restoration = Uformer(img_size=opt.train_ps, embed_dim=32, win_size=8, token_projection='linear',
									token_mlp='leff', depths=[1, 2, 8, 8, 2, 8, 8, 2, 1], modulator=True,
									dd_in=opt.dd_in)
	else:
		raise Exception("Arch error!")

	return model_restoration


def myPSNR(tar_img, prd_img):
	imdff = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img, 0, 1)
	rmse = (imdff**2).mean().sqrt()

	ps = 20 * torch.log10(1 / rmse)
	return ps


def batch_PSNR(img1, img2, average=True):
	PSNR = []
	for im1, im2 in zip(img1, img2):
		psnr = myPSNR(im1, im2)
		PSNR.append(psnr)
	return sum(PSNR) / len(PSNR) if average else sum(PSNR)
