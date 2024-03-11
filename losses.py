
import torch
import torch.nn as nn


def tv_loss(x, beta=0.5, reg_coeff=5):
    '''Calculates TV loss for an image `x`.

    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta`
    '''
    dh = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2)
    dw = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2)
    a, b, c, d = x.shape
    return reg_coeff(torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta)) / (a * b * c * d))


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
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


class Customlosskll1(nn.Module):
    def __init__(self):
        super(Customlosskll1, self).__init__()
        self.kilo = nn.KLDivLoss(reduction='none', log_target=True)
        self.simpleloss = nn.L1Loss(reduction='none')

    def differentiable_histogram(self, x, bins=255, min=0.0, max=1.0):

        if len(x.shape) == 4:
            n_samples, n_chns, _, _ = x.shape
        elif len(x.shape) == 2:
            n_samples, n_chns = 1, 1
        else:
            raise AssertionError('The dimension of input tensor should be 2 or 4.')

        hist_torch = torch.zeros(n_samples, n_chns, bins).to(x.device)
        delta = (max - min) / bins

        BIN_Table = torch.arange(start=0, end=bins, step=1) * delta

        for dim in range(1, bins - 1, 1):
            h_r = BIN_Table[dim].item()             # h_r
            h_r_sub_1 = BIN_Table[dim - 1].item()   # h_(r-1)
            h_r_plus_1 = BIN_Table[dim + 1].item()  # h_(r+1)

            mask_sub = ((h_r > x) & (x >= h_r_sub_1)).float()
            mask_plus = ((h_r_plus_1 > x) & (x >= h_r)).float()

            hist_torch[:, :, dim] += torch.sum(((x - h_r_sub_1) * mask_sub).view(n_samples, n_chns, -1), dim=-1)
            hist_torch[:, :, dim] += torch.sum(((h_r_plus_1 - x) * mask_plus).view(n_samples, n_chns, -1), dim=-1)
        histo = hist_torch / delta
        pdf = histo / torch.sum(histo)
        return pdf

    def __processhistogram(self, inputs, targets, imagesize):
        inputs = (inputs - torch.min(inputs)) / (torch.max(inputs) - torch.min(inputs))
        targets = (targets - torch.min(targets)) / (torch.max(targets) - torch.min(targets))
        pred_hist = self.differentiable_histogram(inputs, bins=imagesize)
        gt_hist = self.differentiable_histogram(targets, bins=imagesize)
        return pred_hist, gt_hist

    def __combiner(self, inp_img, tar_img, weight2, image_size):
        pred_hist, gt_hist = self.__processhistogram(inp_img, tar_img, image_size)
        kld = torch.abs(self.kilo(pred_hist, gt_hist))

        weight2 = weight2.squeeze(0)
        kld = kld * weight2 + (kld / weight2)

        return torch.mean(kld)

    def forward(self, inputo, target, we1, we2):
        we2 = we2.squeeze(-1)

        image_size = inputo.shape[3]
        self.inputs = inputo
        self.epsi = 0.000001
        self.targets = target

        self.weights1 = we1 + self.epsi
        self.weights2 = we2 + self.epsi
        self.parta = self.simpleloss(self.inputs, self.targets)
        self.parta = torch.mean((self.parta * self.weights1) + self.parta / self.weights1)
        self.partb = self.__combiner(self.inputs, self.targets, self.weights2, image_size)
        full = 4 * self.parta + 1 * self.partb
        return full
