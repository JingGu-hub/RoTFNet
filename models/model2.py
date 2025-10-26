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
        self.hight_freq_k = int(rfft_len * low_freq_ratio)
        self.low_freq_k = rfft_len - self.hight_freq_k

        self.l1 = nn.Parameter(torch.randn(self.hight_freq_k, self.embed_size, dtype=torch.cfloat))
        self.h1 = nn.Parameter(torch.randn(self.low_freq_k, self.embed_size, dtype=torch.cfloat))
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

    def freq_aug(self, x_fft):
        # select k low freq components
        low_freq = x_fft[..., :self.hight_freq_k]
        high_freq = x_fft[..., self.hight_freq_k:]

        low_freq = self.complex_compute(torch.einsum('bdk,ke->bde', low_freq, self.l1) + self.lb1, function=F.relu)
        high_freq = self.complex_compute(torch.einsum('bdk,ke->bde', high_freq, self.h1) + self.hb1, function=F.relu)

        low_freq_x = low_freq + self.attention(low_freq, low_freq, low_freq)  # + self.attention(low_freq, high_freq, low_freq)
        high_freq_x = high_freq + self.attention(high_freq, high_freq, high_freq)  # + self.attention(high_freq, low_freq, high_freq)

        # combine low and high freq components
        x_filtered_fft = torch.cat([low_freq_x, high_freq_x[..., low_freq_x.shape[-1]:]], dim=-1)

        return x_filtered_fft

    def forward(self, x):
        x_fft = torch.fft.rfft(x, norm='ortho')  # [B, D, F], F = T//2 + 1
        x_filtered_fft = self.freq_aug(x_fft)
        freq_aug_x = torch.fft.irfft(x_filtered_fft, n=self.seq_len, norm='ortho')

        return freq_aug_x

class FAD(nn.Module):

    def __init__(self, input_channel, seq_len=128, low_freq_ratio=0.5, embed_size=64, dropout=0.1, depth=1):
        super(FAD, self).__init__()

        self.seq_len = seq_len
        self.low_freq_ratio = low_freq_ratio
        self.dropout = dropout

        self.layers = nn.ModuleList([])
        self.ln = nn.LayerNorm([input_channel, seq_len])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                freq_aug_decoder(input_channel=input_channel, seq_len=self.seq_len, low_freq_ratio=self.low_freq_ratio, embed_size=embed_size),
                feed_forward(input_channel, seq_len, int(seq_len / 2))
            ]))

    def forward(self, x):
        for freq_aug, ff in self.layers:
            x = freq_aug(x) + x
            x = ff(x) + x
        x = self.ln(x)

        return x

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

        self.transformer = nn.TransformerEncoderLayer(batch_first=True, d_model=self.seq_len, nhead=args.num_heads)
        self.recons_transformer = nn.TransformerEncoderLayer(batch_first=True, d_model=self.seq_len, nhead=args.num_heads)

        self.low_freq_conv1d = nn.Conv1d(self.input_channel, self.input_channel, kernel_size=3, stride=1, padding=1)
        self.low_freq_transformer = TransformerEncoder(input_dimension=self.input_channel, d_model=self.seq_len)

        self.hight_freq_conv1d = nn.Conv1d(self.input_channel, self.input_channel, kernel_size=3, stride=1, padding=1)
        self.hight_freq_transformer = TransformerEncoder(input_dimension=self.input_channel, d_model=self.seq_len)
        self.linear_change = nn.Linear(self.seq_len, self.seq_len)

        self.down_sampling = self.downsample_torch
        self.up_sampling = F.interpolate
        self.linear_down_sampling = nn.Sequential(
            torch.nn.Linear(self.full_seq_len, self.seq_len),
            nn.GELU(),
            torch.nn.Linear(self.seq_len, self.seq_len)
        )
        self.linear_up_sampling = nn.Sequential(
            torch.nn.Linear(self.seq_len, self.seq_len),
            torch.nn.Linear(self.seq_len, self.full_seq_len)
        )

        self.norm1 = nn.LayerNorm([self.input_channel, self.seq_len])
        self.norm2 = nn.LayerNorm([self.input_channel, self.seq_len])
        self.dropout = nn.Dropout(args.dropout)

        rfft_len = torch.fft.rfft(torch.rand(1, 1, self.seq_len), norm='ortho').shape[-1]
        self.hight_freq_k = int(rfft_len * low_freq_ratio)
        self.low_freq_k = rfft_len - self.hight_freq_k

        self.l1 = nn.Parameter(torch.randn(self.hight_freq_k, self.embed_size, dtype=torch.cfloat))
        self.h1 = nn.Parameter(torch.randn(self.low_freq_k, self.embed_size, dtype=torch.cfloat))
        self.lb1 = nn.Parameter(torch.randn(self.embed_size, dtype=torch.cfloat))
        self.hb1 = nn.Parameter(torch.randn(self.embed_size, dtype=torch.cfloat))

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

    def fourier_transform_freqaug(self, x, low_freq_ratio=0.5):
        x_len = x.shape[-1]
        x_fft = torch.fft.rfft(x, norm='ortho')  # [B, D, F], F = T//2 + 1
        F_half = x_fft.shape[-1]  # total freq number
        k = int(F_half * low_freq_ratio)  # low freq number

        # select k low freq components
        low_freq = x_fft[..., :k]
        high_freq = x_fft[..., k:]

        # augment high freq components
        x_low_time = torch.fft.irfft(low_freq, n=x_len, norm='ortho')
        x_high_time = torch.fft.irfft(high_freq, n=x_len, norm='ortho')

        # convert to frequency domain
        # x_low_time = self.up_sampling(x_low_time, size=self.full_seq_len, mode='linear', align_corners=False)
        # x_high_time = self.up_sampling(x_high_time, size=self.full_seq_len, mode='linear', align_corners=False)

        # apply transformer on low and high freq components
        x_low_time = self.low_freq_conv1d(x_low_time)
        # x_low_time = self.low_freq_transformer(x_low_time)
        # x_high_time = self.hight_freq_conv1d(x_high_time)
        x_high_time = self.hight_freq_transformer(x_high_time)

        # convert back to time domain
        x_low_filtered = torch.fft.rfft(x_low_time, norm='ortho')
        x_high_filtered = torch.fft.rfft(x_high_time, norm='ortho')

        # combine low and high freq components
        x_filtered_fft = torch.cat([x_low_filtered, x_high_filtered[..., low_freq.shape[-1]:]], dim=-1)

        freq_aug_x = torch.fft.irfft(x_filtered_fft, n=self.seq_len, norm='ortho')
        freq_aug_x = self.linear_change(freq_aug_x)
        upsampling_aug_x = self.linear_up_sampling(freq_aug_x)
        # upsampling_aug_x = F.interpolate(freq_aug_x, size=self.seq_len, mode='linear', align_corners=False)
        # down_aug_x = self.linear_down_sampling(freq_aug_x)

        return freq_aug_x, upsampling_aug_x

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

    def ttf_freqaug(self, x):
        x_fft = torch.fft.rfft(x, norm='ortho')  # [B, D, F], F = T//2 + 1

        # select k low freq components
        low_freq = x_fft[..., :self.hight_freq_k]
        high_freq = x_fft[..., self.hight_freq_k:]

        low_freq = self.complex_compute(torch.einsum('bdk,ke->bde', low_freq, self.l1) + self.lb1, function=F.relu)
        high_freq = self.complex_compute(torch.einsum('bdk,ke->bde', high_freq, self.h1) + self.hb1, function=F.relu)

        low_freq_x = low_freq + self.attention(low_freq, low_freq, low_freq) # + self.attention(low_freq, high_freq, low_freq)
        high_freq_x = high_freq + self.attention(high_freq, high_freq, high_freq) # + self.attention(high_freq, low_freq, high_freq)

        # combine low and high freq components
        x_filtered_fft = torch.cat([low_freq_x, high_freq_x[..., low_freq_x.shape[-1]:]], dim=-1)
        freq_aug_x = torch.fft.irfft(x_filtered_fft, n=self.seq_len, norm='ortho')

        freq_aug_x = self.linear_change(freq_aug_x)
        upsampling_aug_x = self.linear_up_sampling(freq_aug_x)

        return freq_aug_x, upsampling_aug_x

    def forward(self, down_x, i=0, epoch=0):
        if i == 0:
            down_x = self.down_sampling(down_x)
            pd = (0, self.padding_length, 0, 0, 0, 0)  # pad -1 dim by padding_length behind
            down_x = F.pad(down_x, pd, "constant", 0)

        down_features = self.transformer(down_x)
        down_features = F.relu(down_features)
        down_features = self.norm1(down_features)
        down_features = self.dropout(down_features)

        recons_aug_x, freq_aug_x = self.ttf_freqaug(down_x.clone())
        recons_features = self.recons_transformer(recons_aug_x)
        recons_features = F.relu(recons_features)
        recons_features = self.norm2(recons_features)
        recons_features = self.dropout(recons_features)

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
