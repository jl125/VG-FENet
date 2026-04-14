import torchvision
from skimage.color import deltaE_ciede2000
import numpy as np
from skimage import color

import torch
import torch.nn as nn




def rgb_to_lab(tensor):

    rgb_np = tensor.detach().permute(0, 2, 3, 1).cpu().numpy() 

    rgb_np = np.clip(rgb_np, 0, 1)

    lab_np = color.rgb2lab(rgb_np)

    lab_tensor = torch.from_numpy(lab_np).permute(0, 3, 1, 2).float().to(tensor.device)  
    return lab_tensor


def ciede2000_loss(y_true, y_pred):


    y_true_lab = rgb_to_lab(y_true)
    y_pred_lab = rgb_to_lab(y_pred)


    diffs = []

    for i in range(y_true_lab.size(0)):
        
        y_true_lab_sample = y_true_lab[i].detach().cpu().numpy()
        y_pred_lab_sample = y_pred_lab[i].detach().cpu().numpy()

        diff = deltaE_ciede2000(y_true_lab_sample, y_pred_lab_sample)
        diffs.append(diff)

    avg_diff = np.mean(diffs)


    return torch.tensor(avg_diff, device=y_true.device, dtype=torch.float32)

class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()
        self.ciede2000_loss_func = ciede2000_loss

    def forward(self, outputs, labels):
        ciede2000_loss_value = self.ciede2000_loss_func(labels, outputs)
        return ciede2000_loss_value


class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        # self.criterion = nn.L1Loss()
        self.criterion = nn.L1Loss(reduction='mean')
        self.criterion2 = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward2(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            # print(x_vgg[i].shape, y_vgg[i].shape)
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            # print(x_vgg[i].shape, y_vgg[i].shape)
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class IlluminationSmoothnessLoss():

    def __init__(self, l_optimized):

        if len(l_optimized.shape) != 4:
            raise ValueError(f"：{len(l_optimized.shape)}")
        if l_optimized.shape[1] != 1:
            raise ValueError(f"：{l_optimized.shape[1]}")

        self.l_optimized = l_optimized

    def __call__(self):

        _, _, H, W = self.l_optimized.shape

        grad_h = torch.abs(self.l_optimized[:, :, 1:, :] - self.l_optimized[:, :, :-1, :])
        grad_v = torch.abs(self.l_optimized[:, :, :, 1:] - self.l_optimized[:, :, :, :-1])

        total_grad = torch.sum(grad_h) + torch.sum(grad_v)
        return total_grad / (H * W)


class ReflectanceSparsityLoss:

    def __init__(self, r_optimized):

        if len(r_optimized.shape) != 4:
            raise ValueError(f"：{len(r_optimized.shape)}")

        self.r_optimized = r_optimized  

    def __call__(self):

        _, C, H, W = self.r_optimized.shape
        total_pixels = H * W * C
        total_l1 = torch.sum(torch.abs(self.r_optimized))
        return total_l1 / total_pixels
