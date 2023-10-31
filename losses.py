
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch
import warnings
#from histotest import processhistogram
from torch import Tensor



def tv_loss(x, beta = 0.5, reg_coeff = 5):
    '''Calculates TV loss for an image `x`.
        
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta` 
    '''
    dh = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2)
    dw = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2)
    a,b,c,d=x.shape
    return reg_coeff*(torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta))/(a*b*c*d))

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class Customlosskll1(nn.Module):
    def __init__(self):
        super(Customlosskll1, self).__init__()
        self.kilo=nn.KLDivLoss(reduction='batchmean')
        self.simpleloss=nn.L1Loss(reduction='none')
        
    def __image_hist2d(self,image: torch.Tensor, min: float = 0., max: float = 1.,
                     n_bins: int = 100, bandwidth: float = -1.,
                     centers: torch.Tensor = torch.tensor([]), return_pdf: bool = False):
        """Function that estimates the histogram of the input image(s).

        The calculation uses triangular kernel density estimation.

        Args:
            x: Input tensor to compute the histogram with shape
            :math:`(H, W)`, :math:`(C, H, W)` or :math:`(B, C, H, W)`.
            min: Lower end of the interval (inclusive).
            max: Upper end of the interval (inclusive). Ignored when
            :attr:`centers` is specified.
            n_bins: The number of histogram bins. Ignored when
            :attr:`centers` is specified.
            bandwidth: Smoothing factor. If not specified or equal to -1,
            bandwidth = (max - min) / n_bins.
            centers: Centers of the bins with shape :math:`(n_bins,)`.
            If not specified or empty, it is calculated as centers of
            equal width bins of [min, max] range.
            return_pdf: If True, also return probability densities for
            each bin.

        Returns:
            Computed histogram of shape :math:`(bins)`, :math:`(C, bins)`,
            :math:`(B, C, bins)`.
            Computed probability densities of shape :math:`(bins)`, :math:`(C, bins)`,
            :math:`(B, C, bins)`, if return_pdf is ``True``. Tensor of zeros with shape
            of the histogram otherwise.
        """
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Input image type is not a torch.Tensor. Got {type(image)}.")

        if centers is not None and not isinstance(centers, torch.Tensor):
            raise TypeError(f"Bins' centers type is not a torch.Tensor. Got {type(centers)}.")

        if centers.numel() > 0 and centers.dim() != 1:
            raise ValueError(f"Bins' centers must be a torch.Tensor "
                             "of the shape (n_bins,). Got {values.shape}.")

        if not isinstance(min, float):
            raise TypeError(f'Type of lower end of the range is not a float. Got {type(min)}.')

        if not isinstance(max, float):
            raise TypeError(f"Type of upper end of the range is not a float. Got {type(min)}.")

        if not isinstance(n_bins, int):
            raise TypeError(f"Type of number of bins is not an int. Got {type(n_bins)}.")

        if bandwidth != -1 and not isinstance(bandwidth, float):
            raise TypeError(f"Bandwidth type is not a float. Got {type(bandwidth)}.")

        if not isinstance(return_pdf, bool):
            raise TypeError(f"Return_pdf type is not a bool. Got {type(return_pdf)}.")

        device = image.device

        if image.dim() == 4:
            batch_size, n_channels, height, width = image.size()
        elif image.dim() == 3:
            batch_size = 1
            n_channels, height, width = image.size()
        elif image.dim() == 2:
            height, width = image.size()
            batch_size, n_channels = 1, 1
        else:
            raise ValueError(f"Input values must be a of the shape BxCxHxW, "
                             f"CxHxW or HxW. Got {image.shape}.")

        if bandwidth == -1.:
            bandwidth = (max - min) / n_bins
        if centers.numel() == 0:
            centers = min + bandwidth * (torch.arange(n_bins, device=device).float() + 0.5)
        centers = centers.reshape(-1, 1, 1, 1, 1)
        u = abs(image.unsqueeze(0) - centers) / bandwidth
        mask = (u <= 1).float()
        hist = torch.sum(((1 - u) * mask), dim=(-2, -1)).permute(1, 2, 0)
        if return_pdf:
            normalization = torch.sum(hist, dim=-1).unsqueeze(0) + 1e-10
            pdf = hist / normalization
            return hist, pdf
        return hist, torch.zeros_like(hist, dtype=hist.dtype, device=device)
    def __fr(self,batch_t1,batch_t2):
        assert batch_t1.shape==batch_t2.shape
        filled_rows=torch.zeros(size=(batch_t1.shape[0],1,batch_t1.shape[-1]))#,device=("cuda"))
        for index,noise in enumerate(batch_t1):
            #filled_rows=torch.rand(size=(128,1))
            cleanref=batch_t2[index]
            c,r,w=noise.shape
            for row in range(0,r):
                cleanrow=cleanref[:,row,:]
                noiserow=noise[:,row,:]
                tn,pfnoise=self.__image_hist2d(noiserow,return_pdf=True)
                tc,pfclean=self.__image_hist2d(cleanrow,return_pdf=True)
                pfnoise[0,0,:]=pfnoise[0,0,:]+ 0.00001
                pfclean[0,0,:]=pfclean[0,0,:]+ 0.00001
                filled_rows[index,0,row]=self.kilo(pfnoise.log(),pfclean)
        return filled_rows   
                
               

    def __fc(self,batch_t1,batch_t2):
        assert batch_t1.shape==batch_t2.shape
        filled_rows=torch.zeros(size=(batch_t1.shape[0],1,batch_t1.shape[-1]))#,device=("cuda"))
        for index,noise in enumerate(batch_t1):
            #filled_rows=torch.rand(size=(128,1))
            cleanref=batch_t2[index]
            c,r,w=noise.shape
            for col in range(0,c):
                caleancol=cleanref[:,:,col]
                noisecol=noise[:,:,col]
                tn,pfnoise=self.__image_hist2d(noisecol,return_pdf=True)
                tc,pfclean=self.__image_hist2d(caleancol,return_pdf=True)
                pfnoise[0,0,:]=pfnoise[0,0,:]+ self.epsi#0.00001
                pfclean[0,0,:]=pfclean[0,0,:]+ self.epsi#0.00001
                filled_rows[index,0,col]=self.kilo(pfnoise.log(),pfclean)
        return filled_rows 

    def __processhistogram(self,inputs,targets):
        rowkls=self.__fr(inputs,targets)
        colkls=self.__fc(inputs,targets)
        
        return rowkls,colkls
    def __combiner(self,inp_img,tar_img,weight2,weight3):
        
        row_kls,col_kls=self.__processhistogram(inp_img, tar_img)
        #print("rowkls,colkls",row_kls.shape,col_kls.shape)
        row_kls=row_kls.cuda()*weight2+(row_kls.cuda()/weight2)
        #print(torch.mean(row_kls))
        col_kls=col_kls.cuda()*weight3+(col_kls.cuda()/weight3)
        #print(torch.mean(col_kls))
        full=(torch.mean(row_kls)+torch.mean(col_kls))/2
        return full

    def forward(self, inputo, target, we1, we2, we3):
        we2=we2.squeeze(-1)
        we3=we3.squeeze(-1)
        
        
        self.inputs=inputo
        self.epsi=0.000001
        self.targets=target
        
        self.weights1=we1+self.epsi
        self.weights2=we2+self.epsi
        self.weights3=we3+self.epsi
        self.parta=self.simpleloss(self.inputs,self.targets)
        self.parta=torch.mean((self.parta*self.weights1)+self.parta/self.weights1)
        self.partb=self.__combiner(self.inputs,self.targets,self.weights2,self.weights3)
        #print(self.parta,"a",self.partb,"b")
        full=self.parta+self.partb
        #print("a:",self.parta.item(),"b:",self.partb.item())
        #print(full.item())
        return full
    
    
    
    