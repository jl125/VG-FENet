import torch
import torch.nn as nn
from torch.fft import rfft2
from torchmetrics.image import StructuralSimilarityIndexMeasure

# 假设以下导入根据实际情况调整
from myloss import ColorLoss, VGGLoss



class EnhancedTotalLoss(nn.Module):
    def __init__(self, initial_epochs=5):
        super().__init__()
        # 基础损失组件
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()

        # 导入现有损失函数
        self.color_loss = ColorLoss()  # CIEDE2000颜色损失
        self.vgg_loss = VGGLoss()  # 感知损失

        # 初始权重设置
        self.initial_l1_weight = 1.0
        self.initial_color_weight = 0.2
        self.initial_vgg_weight = 0.5
        self.initial_fft_weight = 0.2
        self.initial_hsv_channel_weight = 0.2
        self.initial_ssim_weight = 0.1

        # 各通道权重
        self.h_weight = 0.3
        self.s_weight = 0.2
        self.v_weight = 0.7

        # 自适应权重相关参数
        self.initial_epochs = initial_epochs
        self.current_epoch = 0
        self.loss_history = []
        self.epsilon = 1e-8

    def set_epoch(self, epoch):
        """设置当前epoch"""
        self.current_epoch = epoch

    def _get_adaptive_weights(self, loss_values):
        """计算自适应权重"""
        l1_val, color_val, vgg_val, fft_val, hsv_val, ssim_val = loss_values

        # 计算损失的相对比例
        total = l1_val + color_val + vgg_val + fft_val + hsv_val + ssim_val + self.epsilon
        l1_ratio = l1_val / total
        color_ratio = color_val / total
        vgg_ratio = vgg_val / total
        fft_ratio = fft_val / total
        hsv_ratio = hsv_val / total
        ssim_ratio = ssim_val / total

        # 基于比例计算自适应权重
        l1_weight = torch.exp(-l1_ratio)
        color_weight = torch.exp(-color_ratio) * 0.5
        vgg_weight = torch.exp(-vgg_ratio) * 0.3
        fft_weight = torch.exp(-fft_ratio)
        hsv_weight = torch.exp(-hsv_ratio)
        ssim_weight = torch.exp(-ssim_ratio)

        # 权重归一化
        weights_sum = l1_weight + color_weight + vgg_weight + fft_weight + hsv_weight + ssim_weight
        initial_sum = (self.initial_l1_weight + self.initial_color_weight +
                       self.initial_vgg_weight + self.initial_fft_weight +
                       self.initial_hsv_channel_weight + self.initial_ssim_weight)

        scale = initial_sum / (weights_sum + self.epsilon)

        return [
            l1_weight * scale,
            color_weight * scale,
            vgg_weight * scale,
            fft_weight * scale,
            hsv_weight * scale,
            ssim_weight * scale
        ]

    def _fft_domain_loss(self, pred, target):
        """计算频域损失"""
        pred_gray = 0.299 * pred[:, 0:1, :, :] + 0.587 * pred[:, 1:2, :, :] + 0.114 * pred[:, 2:3, :, :]
        target_gray = 0.299 * target[:, 0:1, :, :] + 0.587 * target[:, 1:2, :, :] + 0.114 * target[:, 2:3, :, :]

        pred_fft = rfft2(pred_gray)
        target_fft = rfft2(target_gray)

        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)
        amp_loss = self.l1_loss(pred_amp, target_amp)

        pred_phase = torch.angle(pred_fft)
        target_phase = torch.angle(target_fft)
        phase_loss = 0.1 * self.l1_loss(pred_phase, target_phase)

        return amp_loss + phase_loss

    def _hsv_channel_loss(self, pred, target):
        """计算HSV通道损失"""
        pred_hsv = rgb_to_hsv(pred)
        target_hsv = rgb_to_hsv(target)

        h_pred, s_pred, v_pred = pred_hsv[:, 0:1, :, :], pred_hsv[:, 1:2, :, :], pred_hsv[:, 2:3, :, :]
        h_target, s_target, v_target = target_hsv[:, 0:1, :, :], target_hsv[:, 1:2, :, :], target_hsv[:, 2:3, :, :]

        h_loss = self.h_weight * self.l1_loss(h_pred, h_target)
        s_loss = self.s_weight * self.l1_loss(s_pred, s_target)
        v_loss = self.v_weight * self.l1_loss(v_pred, v_target)

        return h_loss + s_loss + v_loss

    def forward(self, pred, target):
        """计算综合损失"""
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)

        # 计算各损失分量
        l1 = self.l1_loss(pred, target)
        color = self.color_loss(pred, target)
        vgg = self.vgg_loss(pred, target)
        fft = self._fft_domain_loss(pred, target)
        hsv_loss = self._hsv_channel_loss(pred, target)
        ssim_val = self.ssim(pred, target)
        ssim_loss = 1 - ssim_val

        loss_values = [l1, color, vgg, fft, hsv_loss, ssim_loss]

        # 选择权重策略
        if self.current_epoch < self.initial_epochs:
            # 初始阶段使用固定权重（作为Python浮点数）
            weights = [
                self.initial_l1_weight,
                self.initial_color_weight,
                self.initial_vgg_weight,
                self.initial_fft_weight,
                self.initial_hsv_channel_weight,
                self.initial_ssim_weight
            ]
        else:
            # 自适应阶段使用张量权重
            weights = self._get_adaptive_weights([v.item() for v in loss_values])

        # 计算总损失
        total_loss = (
                weights[0] * l1 +
                weights[1] * color +
                weights[2] * vgg +
                weights[3] * fft +
                weights[4] * hsv_loss +
                weights[5] * ssim_loss
        )

        # 记录损失历史 - 修复了错误所在行
        # 对张量使用.item()，对普通浮点数直接使用
        recorded_weights = []
        for w in weights:
            if isinstance(w, torch.Tensor):
                recorded_weights.append(w.item())
            else:
                recorded_weights.append(w)

        self.loss_history.append({
            'l1': l1.item(),
            'color': color.item(),
            'vgg': vgg.item(),
            'fft': fft.item(),
            'hsv': hsv_loss.item(),
            'ssim': ssim_loss.item(),
            'weights': recorded_weights  # 使用修复后的权重列表
        })

        return total_loss
