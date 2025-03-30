import torch
from torch import nn

from Time_Series_Library.layers.Transformer_EncDec import Encoder, EncoderLayer
from Time_Series_Library.layers.SelfAttention_Family import FullAttention, AttentionLayer
from Time_Series_Library.layers.Embed import PatchEmbedding, DataEmbedding, DataEmbedding_inverted
from Time_Series_Library.layers.PatchTST_layers import series_decomp

import ptwt
import pywt
import numpy as np


def Wavelet_for_Period(x, scale=1):
    scales = 2 ** np.linspace(-1, scale, 8)
    coeffs, freqs = ptwt.cwt(x, scales, "morl")
    return coeffs, freqs

def wavelet_denoising(data, wavelet, level):
    # 小波分解
    data = data.detach().cpu().numpy()
    coeffs = pywt.wavedec(data, wavelet, level=level)

    # 计算阈值
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(512))

    # 阈值处理
    coeffs[1:] = (pywt.threshold(i, value=uthresh, mode='soft') for i in coeffs[1:])

    # 重构信号
    return pywt.waverec(coeffs, wavelet)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class TimeFrequencyPredictor(nn.Module):
    def __init__(self, configs):
        super(TimeFrequencyPredictor, self).__init__()
        self.norm = nn.LayerNorm(normalized_shape=configs.d_model)
        self.fc = nn.Linear(configs.d_model, configs.pred_len)

    def forward(self, x):
        N, C, T = x.shape  # 32,11,512

        # 时间步归一化
        x = x.reshape(-1, T)
        x = self.norm(x)
        x = x.reshape(N, C, T)
        future_prediction = self.fc(x)  # (32, 11, 96)

        return future_prediction.permute(0, 2, 1)


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs, patch_len=16, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride
        kernel_size = configs.kernel_size
        # wavelet scale
        self.scale = configs.wavelet_scale
        # decomposition
        self.decomp_module = series_decomp(kernel_size)
        self.wavelet = TimeFrequencyPredictor(configs)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.enc_embedding_inverted = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed,
                                                             configs.freq,
                                                             configs.dropout)
        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)
        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )

        # Encoder
        self.encoder_trend = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.encoder_res = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        self.head_res = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)
        self.head_trend = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                      head_dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        _, _, N = x_enc.shape

        # do decomposition
        res_init, trend_init = self.decomp_module(x_enc)  # 32,96,7

        """
        PATCHTST PART
        """

        # do patching and embedding 对较为平稳的信号做PatchTST
        x_enc_trend = trend_init.permute(0, 2, 1)  # 32,7,96
        x_enc_res = res_init.permute(0, 2, 1)

        # u: [bs * nvars x patch_num x d_model]
        enc_out_trend, n_vars = self.patch_embedding(x_enc_trend)
        enc_out_res, n_vars = self.patch_embedding(x_enc_res)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out_trend, attns_trend = self.encoder_trend(enc_out_trend)
        enc_out_res, attns_res = self.encoder_res(enc_out_res)
        # z: [bs x nvars x patch_num x d_model]
        enc_out_trend = torch.reshape(enc_out_trend, (-1, n_vars, enc_out_trend.shape[-2], enc_out_trend.shape[-1]))
        enc_out_res = torch.reshape(enc_out_res, (-1, n_vars, enc_out_res.shape[-2], enc_out_res.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out_trend = enc_out_trend.permute(0, 1, 3, 2)
        enc_out_res = enc_out_res.permute(0, 1, 3, 2)

        # Decoder
        dec_out_trend = self.head_trend(enc_out_trend)  # z: [bs x nvars x target_window]
        dec_out_res = self.head_res(enc_out_res)
        dec_out_trend = dec_out_trend.permute(0, 2, 1)  # 32,96,7
        dec_out_res = dec_out_res.permute(0, 2, 1)  # 32,96,7

        """
        WAVELET PART
        """
        enc_out_wave_s = self.enc_embedding_inverted(res_init, x_mark_enc)
        enc_out_wave_t = self.enc_embedding_inverted(trend_init, x_mark_enc)  # 32,11,512

        coeffs_s = wavelet_denoising(enc_out_wave_s, 'db2', level=4)
        coeffs_t = wavelet_denoising(enc_out_wave_t, 'db2', level=4)
        coeffs_s = torch.from_numpy(coeffs_s).float()
        coeffs_t = torch.from_numpy(coeffs_t).float()
        if enc_out_wave_s.is_cuda:
            coeffs_s = coeffs_s.to('cuda')
        if enc_out_wave_t.is_cuda:
            coeffs_t = coeffs_t.to('cuda')
        # coeffs = Wavelet_for_Period(enc_out_wave, self.scale)[0].permute(1, 2, 0, 3).float()  # 32,11,7,512
        dec_out_wave_s = self.wavelet(coeffs_s)
        dec_out_wave_t = self.wavelet(coeffs_t)
        dec_out_wave = dec_out_wave_t + dec_out_wave_s

        alpha = torch.sigmoid(self.alpha)
        dec_out = alpha * (dec_out_trend + dec_out_res) + (1 - alpha) * dec_out_wave[:, -self.pred_len:, :N]

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):

        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

