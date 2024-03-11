
import os
import argparse
import options
opt = options.Options().init(argparse.ArgumentParser(description='Image denoising')).parse_args()  # parser
print(opt)

import util
# from dataset.dataset_denoise import *

"""Set GPUs"""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
import torch
torch.backends.cudnn.benchmark = True

import torch.optim as optim
from torch.utils.data import DataLoader
import random
import time
import numpy as np
import datetime
from losses import CharbonnierLoss, Customlosskll1
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler
from datasetloader import get_training_data, get_validation_data

""" Logs dir """
log_dir = os.path.join(opt.save_dir, 'denoising', opt.dataset, opt.arch + opt.env)
if not os.path.exists(log_dir):
	os.makedirs(log_dir)

logname = os.path.join(log_dir, datetime.datetime.now().isoformat() + '.txt')
print("Now time is : ", datetime.datetime.now().isoformat())
result_dir = os.path.join(log_dir, 'results')
model_dir = os.path.join(log_dir, 'models')

os.makedirs(result_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

""" Set Seeds """
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

""" Model """
model_restoration = util.get_arch(opt)
tom = 40
assert tom == 40

with open(logname, 'a') as f:
	f.write(str(opt) + '\n')
	f.write(str(model_restoration) + '\n')

""" Optimizer """
start_epoch = 1
if opt.optimizer.lower() == 'adamw':
	optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999), eps=1e-8,
							weight_decay=opt.weight_decay)
else:
	raise Exception("Error optimizer...")

""" DataParallel """
model_restoration = torch.nn.DataParallel(model_restoration)
model_restoration.cuda()

""" Scheduler """
if opt.warmup:
	print("Using warmup and cosine strategy!")
	warmup_epochs = opt.warmup_epochs
	scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch - warmup_epochs, eta_min=1e-6)
	scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
										after_scheduler=scheduler_cosine)
	scheduler.step()
else:
	step = 50
	print("Using Step LR, plat={}!".format(step))
	scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
	scheduler.step()

""" Resume """
if opt.resume:
	path_chk_rest = opt.pretrain_weights
	print("Resume from " + path_chk_rest)
	util.load_checkpoint(model_restoration, path_chk_rest)
	start_epoch = util.load_start_epoch(path_chk_rest) + 1
	lr = util.load_optim(optimizer, path_chk_rest)

	# for p in optimizer.param_groups: p['lr'] = lr
	# warmup = False
	# new_lr = lr
	# print('------------------------------------------------------------------------------')
	# print(" == > Resuming Training with learning rate:", new_lr)
	# print('------------------------------------------------------------------------------')
	for i in range(1, start_epoch):
		scheduler.step()
	new_lr = scheduler.get_lr()[0]
	print('------------------------------------------------------------------------------')
	print(" == > Resuming Training with learning rate:", new_lr)
	print('------------------------------------------------------------------------------')

	# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-start_epoch+1, eta_min = 1e-6)

""" Loss """
criterion = CharbonnierLoss().cuda()
# criterion = nn.L1Loss().cuda()
criterionkll1 = Customlosskll1().cuda()
# criterionkll1.device =  torch.device("cuda")

""" DataLoader """
print(' == = > Loading datasets')
img_options_train = {'patch_size': opt.train_ps}

train_dataset = get_training_data(opt.train_dir, opt.upscale, opt.train_ps)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True,
		num_workers=opt.train_workers, pin_memory=False, drop_last=True)
val_dataset = get_validation_data(opt.val_dir, opt.upscale, opt.train_ps)
val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False,
		num_workers=opt.eval_workers, pin_memory=False, drop_last=False)

len_trainset = train_dataset.__len__()
len_valset = val_dataset.__len__()
print("Sizeof training set: ", len_trainset, ", sizeof validation set: ", len_valset)

""" validation """
with torch.no_grad():
	model_restoration.eval()
	psnr_dataset = []
	psnr_model_init = []
	for ii, data_val in enumerate((val_loader), 0):

		target = data_val[0].cuda()
		input_ = data_val[1].cuda()

		with torch.cuda.amp.autocast():
			restored, weightsl1, weights_kl = model_restoration(input_)

			# restored = torch.clamp(restored, 0, 1)
		psnr_dataset.append(util.batch_PSNR(input_, target, False).item())
		psnr_model_init.append(util.batch_PSNR(restored, target, False).item())
	psnr_dataset = sum(psnr_dataset) / len_valset
	psnr_model_init = sum(psnr_model_init) / len_valset
	print('Input & GT (PSNR) -->%.4f dB' % (psnr_dataset), ', Model_init & GT (PSNR) -->%.4f dB' % (psnr_model_init))

""" train """
print(' == = > Start Epoch {} End Epoch {}'.format(start_epoch, opt.nepoch))
best_psnr = 0
best_epoch = 0
best_iter = 0
eval_now = len(train_loader) // 64
print("\nEvaluation after every {} Iterations !!!\n".format(eval_now))

loss_scaler = NativeScaler()
torch.cuda.empty_cache()

for epoch in range(start_epoch, opt.nepoch + 1):
	epoch_start_time = time.time()
	epoch_loss = 0
	train_id = 1

	for i, data in enumerate(tqdm(train_loader), 0):
		# zero_grad
		optimizer.zero_grad()

		target = data[0].cuda()
		input_ = data[1].cuda()

		# if epoch>5:
		# 	target, input_ = utils.MixUp_AUG().aug(target, input_)
		with torch.cuda.amp.autocast():
			restored, weightsl1, weights_kl = model_restoration(input_)

		loss = criterionkll1(restored, target, weightsl1, weights_kl)
		loss_scaler(
				loss, optimizer, parameters=model_restoration.parameters(), create_graph=True)
		# [{"params":w1}, {"params":w2}, {"params":w3}, {"params":model_restoration.parameters()}]
		# loss_scaler(loss, optimizer, [{"params":w1}, {"params":w2}, {"params":w3},
		# 				{"params":model_restoration.parameters()}])
		epoch_loss += loss.item()

		"""Evaluation"""
		if (i + 1) % eval_now == 0 and i > 0:
			with torch.no_grad():
				model_restoration.eval()
				psnr_val_rgb = []
				ssim_val_rgb = []
				for ii, data_val in enumerate((val_loader), 0):
					target = data_val[0].cuda()
					input_ = data_val[1].cuda()
					filenames = data_val[2]
					with torch.cuda.amp.autocast():
						restored, weightsl1, weights_kl_row, weights_kl_col = model_restoration(input_)

					psnr_val_rgb.append(util.batch_PSNR(restored, target, False).item())
				psnr_val_rgb = sum(psnr_val_rgb) / len_valset
				if psnr_val_rgb > best_psnr:
					best_psnr = psnr_val_rgb
					best_epoch = epoch
					best_iter = i
					torch.save({'epoch': epoch,
								'state_dict': model_restoration.state_dict(),
								'optimizer': optimizer.state_dict()
								}, os.path.join(model_dir, "model_best.pth"))
				print("[Ep %d it %d\t PSNR Sino: %.4f\t] ----  [best_Ep_sino %d best_it_sino %d Best_PSNR%.4f] "
						% (epoch, i, psnr_val_rgb, best_epoch, best_iter, best_psnr))
				with open(logname, 'a') as f:
					f.write("[Ep %d it %d\t PSNR sino: %.4f\t] ----  [best_Ep_sino %d best_it_sino %d Best_PSNR_sino%.4f] "
						% (epoch, i, psnr_val_rgb, best_epoch, best_iter, best_psnr) + '\n')
				model_restoration.train()
				torch.cuda.empty_cache()
	scheduler.step()

	print("------------------------------------------------------------------")
	print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(
			epoch, time.time() - epoch_start_time, epoch_loss, scheduler.get_last_lr()[0]))
	print("------------------------------------------------------------------")
	with open(logname, 'a') as f:
		f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(
				epoch, time.time() - epoch_start_time, epoch_loss, scheduler.get_last_lr()[0]) + '\n')

	torch.save({'epoch': epoch,
				'state_dict': model_restoration.state_dict(),
				'optimizer': optimizer.state_dict()
				}, os.path.join(model_dir, "model_latest.pth"))

	if epoch % opt.checkpoint == 0:
		torch.save({'epoch': epoch,
					'state_dict': model_restoration.state_dict(),
					'optimizer': optimizer.state_dict()
					}, os.path.join(model_dir, "model_epoch_{}.pth".format(epoch)))
print("Now time is : ", datetime.datetime.now().isoformat())
