
import os
import torch
import sys
import argparse
import options
import util
import torch.nn as nn
import torch.optim as optim
import glob
from skimage.restoration import denoise_nl_means, estimate_sigma
import numpy as np
from empatches import EMPatches
import tifffile as tl
import cv2
from collections import OrderedDict

emp = EMPatches()
opt = options.Options().init(argparse.ArgumentParser(description='Image denoising')).parse_args()
model_restoration = util.get_arch(opt)
#path_chk_rest = opt.pretrain_weights 
#print("Resume from "+path_chk_rest)
#util.load_checkpoint(model_restoration,path_chk_rest) 

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
        
def minmaxscale(img):
    img=img.astype('float32')
    norm = (img - np.min(img)) / (np.max(img) - np.min(img))
    return norm
def remove_outlier(img):
    a_sigma_est = estimate_sigma(img, channel_axis=None, average_sigmas=True)
    asig=denoise_nl_means(img, patch_size=9, patch_distance=5,
                   fast_mode=True, sigma=0.001*a_sigma_est,
                  preserve_range=False, channel_axis=None)
    return asig
def preprocess(img,scale):
    img=minmaxscale(img)
    img=remove_outlier(img)
    m,n=img.shape
    img=cv2.resize(img,(n,int(m*scale)),interpolation=cv2.INTER_CUBIC)
    return img
def transforme(img):
    img=torch.Tensor(img)
    img=img.unsqueeze(0)
    img=img.unsqueeze(0)
    return img.cuda()

pathtosinograms='/gpuhome/aus79/data/seasct/'
allsinograms=glob.glob(pathtosinograms+'*.tif')
storepath='./restoredimages/'
if not os.path.exists(storepath):
    os.makedirs(storepath,exist_ok=True)
patch_size=128
upscale=4
path_to_trained_model=opt.save_dir+'denoising/custom/Drht_/models/model_best.pth'
#path_to_trained_model='./lgos/uformer/customzebra/denoising/custom/Drht_/models/model_best.pth'
model_restoration=load_checkpoint(model_restoration,path_to_trained_model)
model_restoration=model_restoration.cuda()

print("model loaded")
with torch.no_grad():
    model_restoration.eval()
    for item in allsinograms:
        if os.path.isfile(item):
            input_image=tl.imread(item)
            input_image=preprocess(input_image,upscale)
            patches,indices=emp.extract_patches(input_image, patchsize=patch_size, overlap=0.5)
            collect=[]
            p_len=len(patches)
            print("patched")
            for ind,patch in enumerate(patches):
                patch=transforme(patch)
                with torch.cuda.amp.autocast():
                    restored,_,_=model_restoration(patch)
                    print("restored patch ",str(ind),"/",str(p_len))
                collect.append(restored[0].squeeze(0).detach().cpu().numpy())
            print("collected")
            merged_img = emp.merge_patches(collect, indices, mode='avg')
            name=(item.split('/')[-1]).replace('.tif','_sr.tif')
            pt=storepath+name
            tl.imsave(pt,merged_img)
    print("finished")
