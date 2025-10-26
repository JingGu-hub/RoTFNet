import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.MLP import MLP
from utils.utils import draw_curve

class TransformerEncoder(nn.Module):
    def __init__(self, input_dimension=1, d_model=128, nhead=4, numm_layers=1, dropout=0.1, batch_first=True):
        super(TransformerEncoder, self).__init__()

        new_input_length = math.ceil(d_model / nhead) * nhead
        padding_length = new_input_length - d_model

        self.seq_len = d_model
        self.new_input_length = new_input_length
        self.input_dimension = input_dimension
        self.nhead = nhead
        self.padding_length = int(padding_length)
        self.numm_layers = numm_layers

        self.transformer = nn.TransformerEncoderLayer(batch_first=True, d_model=self.new_input_length, nhead=self.nhead)
        self.norm = nn.LayerNorm([self.input_dimension, self.new_input_length])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        pd = (0, self.padding_length, 0, 0, 0, 0)  # pad -1 dim by padding_length behind
        x = F.pad(x, pd, "constant", 0)

        for i in range(self.numm_layers):
            x = self.transformer(x)
            x = F.relu(x)
            x = self.norm(x)
            x = self.dropout(x)
        x = x[:, :, :self.seq_len]

        return x


class feed_forward(nn.Module):
    def __init__(self, input_channel, dim, hidden_dim, dropout=0.1):
        super().__init__()

        self.norm = nn.LayerNorm([input_channel, dim])
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(self.norm(x))

class freq_aug_decoder(nn.Module):

    def __init__(self, input_channel, seq_len=128, low_freq_ratio=0.5, embed_size=64):
        super(freq_aug_decoder, self).__init__()

        self.input_channel = input_channel
        self.seq_len = seq_len
        self.embed_size = embed_size

        self.norm = nn.LayerNorm([input_channel, seq_len])

        rfft_len = torch.fft.rfft(torch.rand(1, 1, self.seq_len), norm='ortho').shape[-1]
        self.low_freq_k = int(rfft_len * low_freq_ratio)
        self.hight_freq_k = rfft_len - self.low_freq_k

        self.embed_size = rfft_len

        self.l1 = nn.Parameter(torch.randn(self.low_freq_k, self.embed_size, dtype=torch.cfloat))
        self.h1 = nn.Parameter(torch.randn(self.hight_freq_k, self.embed_size, dtype=torch.cfloat))
        self.lb1 = nn.Parameter(torch.randn(self.embed_size, dtype=torch.cfloat))
        self.hb1 = nn.Parameter(torch.randn(self.embed_size, dtype=torch.cfloat))

    def complex_compute(self, z, function=F.relu):
        if function == F.softmax:
            return torch.complex(function(z.real, dim=-1), function(z.imag, dim=-1))  # softmax on real and imaginary parts separately
        return torch.complex(function(z.real), function(z.imag))  # 可换成其他激活策略

    def attention(self, Q, K, V):
        d_k = Q.size(-1)  # head_dim
        scores = torch.matmul(Q, K.transpose(-2, -1)) / d_k ** 0.5  # (B, H, L, L)

        attn_weights = self.complex_compute(scores, F.softmax)  # (B, H, L, L)
        output = torch.matmul(attn_weights, V)  # (B, H, L, D)

        return output

    def gaussian_freq_conv(self, data, kernel_size=3):
        """
        对形状为 (B, C, F) 的频域张量进行高斯卷积，使用向量化操作加速
        """
        B, C, F = data.shape
        assert kernel_size <= F, "kernel_size 不能大于频域长度 F"

        # 构建高斯卷积核（汉宁窗）
        kernel = np.hanning(kernel_size)
        kernel /= kernel.sum()
        kernel_padded = np.zeros(F, dtype=np.float32)
        kernel_padded[:kernel_size] = kernel

        # 转换为 PyTorch tensor 并 FFT，形状为 (F,)
        kernel_padded_tensor = torch.tensor(kernel_padded, dtype=torch.float32, device=data.device)
        kernel_fft = torch.fft.fft(kernel_padded_tensor)  # (F,)

        # 向量化乘法：data (B, C, F) * kernel_fft (F,) → 广播为 (B, C, F)
        result = data * kernel_fft  # 广播自动完成

        return result

    def freq_aug(self, x_fft):
        # select k low freq components
        low_freq = x_fft[..., :self.low_freq_k]
        high_freq = x_fft[..., self.low_freq_k:]

        low_freq = self.complex_compute(torch.einsum('bdk,ke->bde', low_freq, self.l1) + self.lb1, function=F.relu)
        high_freq = self.complex_compute(torch.einsum('bdk,ke->bde', high_freq, self.h1) + self.hb1, function=F.relu)

        # low_freq = self.gaussian_freq_conv(low_freq)
        low_freq_x = self.attention(low_freq, low_freq, low_freq) + self.attention(low_freq, high_freq, low_freq)
        high_freq_x = self.attention(high_freq, high_freq, high_freq) + self.attention(high_freq, low_freq, high_freq)

        # combine low and high freq components
        x_filtered_fft = torch.cat([low_freq_x, high_freq_x[..., low_freq_x.shape[-1]:]], dim=-1)

        return x_filtered_fft

    def forward(self, x):
        x_fft = torch.fft.rfft(x, norm='ortho')  # [B, D, F], F = T//2 + 1
        x_filtered_fft = self.freq_aug(x_fft)
        freq_aug_x = torch.fft.irfft(x_filtered_fft, n=self.seq_len, norm='ortho')

        return freq_aug_x

class FAD(nn.Module):

    def __init__(self, input_channel, seq_len=128, full_seq_len=256, low_freq_ratio=0.5, embed_size=64, dropout=0.1, depth=1):
        super(FAD, self).__init__()

        self.seq_len = seq_len
        self.full_seq_len = full_seq_len
        self.low_freq_ratio = low_freq_ratio
        self.dropout = dropout

        self.layers = nn.ModuleList([])
        self.ln = nn.LayerNorm([input_channel, seq_len])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                freq_aug_decoder(input_channel=input_channel, seq_len=self.seq_len, low_freq_ratio=self.low_freq_ratio, embed_size=embed_size),
                feed_forward(input_channel, seq_len, int(seq_len / 2))
            ]))

        self.linear_change = nn.Linear(self.seq_len, self.seq_len)
        self.linear_up_sampling = nn.Sequential(
            torch.nn.Linear(self.seq_len, self.seq_len),
            torch.nn.Linear(self.seq_len, self.full_seq_len)
        )

    def forward(self, x):
        for freq_aug, ff in self.layers:
            x = freq_aug(x) + x
            x = ff(x) + x
        x = self.ln(x)

        freq_aug_x = self.linear_change(x)
        upsampling_aug_x = self.linear_up_sampling(freq_aug_x)

        return freq_aug_x, upsampling_aug_x

class NoiScaleModule(nn.Module):

    def __init__(self, args, seq_len, padding_length, downsample_scale=1, type='avgpool', low_freq_ratio=0.5):
        super(NoiScaleModule, self).__init__()

        self.input_channel = args.input_channel
        self.donwsample_scale = downsample_scale
        self.period = args.period
        self.padding_length = padding_length
        self.seq_len = seq_len
        self.full_seq_len = args.seq_len
        self.embed_size = args.embed_size
        self.low_freq_ratio = args.low_freq_ratio

        self.transformer = nn.TransformerEncoderLayer(batch_first=True, d_model=self.seq_len, nhead=args.num_heads)
        self.recons_transformer = nn.TransformerEncoderLayer(batch_first=True, d_model=self.seq_len, nhead=args.num_heads)

        self.down_sampling = self.downsample_torch
        self.up_sampling = F.interpolate
        self.linear_down_sampling = nn.Sequential(
            torch.nn.Linear(self.full_seq_len, self.seq_len),
            nn.GELU(),
            torch.nn.Linear(self.seq_len, self.seq_len)
        )

        self.norm1 = nn.LayerNorm([self.input_channel, self.seq_len])
        self.norm2 = nn.LayerNorm([self.input_channel, self.seq_len])
        self.dropout1 = nn.Dropout(args.dropout)
        self.dropout2 = nn.Dropout(args.dropout)

        self.fad = FAD(input_channel=self.input_channel, seq_len=self.seq_len, full_seq_len=self.full_seq_len,
                       low_freq_ratio=self.low_freq_ratio, embed_size=args.embed_size, depth=1)

    def downsample_torch(self, data):
        """
         Takes a batch of sequences with [batch size, channels, seq_len] and down-samples with sample
         rate k. hence, every k-th element of the original time series is kept.
        """
        last_one = 0
        if data.shape[2] % self.donwsample_scale > 0:
            last_one = 1

        new_length = int(np.floor(data.shape[2] / self.donwsample_scale)) + last_one
        output = torch.zeros((data.shape[0], data.shape[1], new_length)).cuda()
        output[:, :, range(new_length)] = data[:, :, [i * self.donwsample_scale for i in range(new_length)]]

        return output

    def forward(self, down_x, i=0, epoch=0):
        if i == 0:
            down_x = self.down_sampling(down_x)
            pd = (0, self.padding_length, 0, 0, 0, 0)  # pad -1 dim by padding_length behind
            down_x = F.pad(down_x, pd, "constant", 0)

        down_features = self.transformer(down_x)
        down_features = F.relu(down_features)
        down_features = self.norm1(down_features)
        down_features = self.dropout1(down_features)

        recons_aug_x, freq_aug_x = self.fad(down_x.clone())
        recons_features = self.transformer(recons_aug_x)
        recons_features = F.relu(recons_features)
        recons_features = self.norm1(recons_features)
        recons_features = self.dropout1(recons_features)

        # recons_features = self.recons_transformer(recons_aug_x)
        # recons_features = F.relu(recons_features)
        # recons_features = self.norm2(recons_features)
        # recons_features = self.dropout2(recons_features)

        # if epoch == 45:
        #     draw_curve(down_x.detach().cpu().numpy(), recons_aug_x.detach().cpu().numpy())

        if i == 0:
            res = down_features + recons_features
        else:
            res = down_x + down_features + recons_features

        return res, down_x, recons_aug_x, freq_aug_x

class NoiScaleFormer(nn.Module):

    def __init__(self, args, downsample_scale):
        super(NoiScaleFormer, self).__init__()

        self.input_channel = args.input_channel
        self.feature_size = args.feature_size
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads

        new_input_length = math.ceil(math.ceil(args.seq_len / downsample_scale) / self.num_heads) * self.num_heads
        padding_length = new_input_length - math.ceil(args.seq_len / downsample_scale)
        self.seq_len = new_input_length
        self.padding_length = int(padding_length)

        self.NoiScaleModule_list = nn.ModuleList([
            NoiScaleModule(args, seq_len=new_input_length, padding_length=self.padding_length, downsample_scale=downsample_scale, type='avgpool')
            for _ in range(self.num_layers)])

        self.feature_linear = nn.Linear(self.input_channel * new_input_length, self.feature_size)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)
        self.fc = MLP(input_size=self.feature_size, hidden_size=128, output_size=args.num_classes)
        # self.fc = nn.Linear(self.feature_size, args.num_classes)

        self.decoder = nn.Linear(self.feature_size, self.input_channel * new_input_length)

    def model_train(self):
        self.NoiScaleModule_list.eval()
        self.feature_linear.eval()
        self.fc.train()

    def pretrain_forward(self, x, epoch=0):
        down_x = None
        freq_aug_list, recons_aug_list = [], []
        for i in range(self.num_layers):
            x, res_x, recons_aug_x, freq_aug_x = self.NoiScaleModule_list[i](x, i, epoch=epoch)
            freq_aug_list.append(freq_aug_x)
            recons_aug_list.append(recons_aug_x)
            if down_x is None:
                down_x = res_x

        x = x.reshape(x.shape[0], -1)  # (B, embed_dim * T)
        output = self.feature_linear(x.float())

        reconst_x = self.decoder(output)
        reconst_x = reconst_x.reshape(reconst_x.shape[0], self.input_channel, self.seq_len)

        return down_x, reconst_x, recons_aug_list, freq_aug_list

    def train_forward(self, x):
        for i in range(self.num_layers):
            x, _, _, _ = self.NoiScaleModule_list[i](x, i)

        x = x.reshape(x.shape[0], -1)  # (B, embed_dim * T)
        output = self.feature_linear(x.float())
        logits = self.fc(output)

        return logits

    def forward(self, x, task='train', epoch=0):
        if task == 'pretrain':
            down_x, reconst_x, recons_aug_list, freq_aug_list = self.pretrain_forward(x, epoch=epoch)
            return down_x, reconst_x, recons_aug_list, freq_aug_list
        else:
            logits = self.train_forward(x)
            return logits
