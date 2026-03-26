import torch
import torch.nn.functional as F
import sys
import math

class AGCLoss_forward(torch.nn.Module):
    '''
        input is Al, Bl, channel = 1, range ~ [0, 255]
    '''

    def __init__(self,
                 sample_size=100, h=2, patch_size=7,
                 lambda_occ=0.05,
                 t = 1.):
        super(AGCLoss_forward, self).__init__()
        self.sample_size = sample_size
        self.patch_size = patch_size
        self.stride = max(patch_size // 2, 2)
        self.h = h
        self.occurrence_weight = lambda_occ
        self.t = t
    def feature_extraction(self, feature, sample_field=None):
        # Patch extraction - use patch as single feature
        if self.patch_size > 1:
            feature = batch_patch_extraction(feature, self.patch_size, self.stride) # patch windows values

        # Random sampling - random patches
        num_batch, num_channel = feature.shape[:2]
        if num_batch * feature.shape[-2] * feature.shape[-1] > self.sample_size ** 2:
            if isNone(sample_field):
                sample_field = torch.rand(
                    num_batch, self.sample_size, self.sample_size, 2, device=feature.device) * 2 - 1
            feature = F.grid_sample(feature, sample_field, mode='nearest')

        # Concatenate tensor
        sampled_feature = feature

        return sampled_feature, sample_field

    def calculate_distance(self, target_features, refer_features):
        # feature
        target_features, target_field = self.feature_extraction(target_features)
        refer_features, refer_field = self.feature_extraction(refer_features)
        d_total = compute_cosine_distance(target_features, refer_features) #! VGG dij loss

        # occurrence penalty
        min_idx_for_target = torch.min(d_total, dim=-1, keepdim=True)[1] # torch.min() return index,min_idx_for_target of the minimum value.
        use_occurrence = self.occurrence_weight > 0
        if use_occurrence:
            with torch.no_grad():
                omega = d_total.shape[1] / d_total.shape[2]
                occur = torch.zeros_like(d_total[:, 0, :]) # shape [1, source_size]
                indexs, counts = min_idx_for_target[0, :, 0].unique(return_counts=True) # Compute the occurrence frequencies of distinct elements.
                threshold = counts.max()*self.t #! t
                indexs_remove = indexs[counts <= threshold]
                counts_remove = counts[counts <= threshold]
                occur[:, indexs_remove] = counts_remove / omega #! occ loss
                occur = occur.view(1, 1, -1)
                d_total += occur * self.occurrence_weight

        return d_total

    def calculate_loss(self, d):
        # --------------------------------------------------
        # guided correspondence loss
        # --------------------------------------------------
        # calculate loss        # for each target feature, find closest refer feature
        d_min = torch.min(d, dim=-1, keepdim=True)[0]
        # convert to relative distance
        d_norm = d / (d_min + sys.float_info.epsilon)
        w = torch.exp((1 - d_norm) * self.h) #! wij
        A_ij = w / torch.sum(w, dim=-1, keepdim=True) #! CXij
        # texture loss per sample
        CX = torch.max(A_ij, dim=-1)[0] #! CX(i,NNi) NNi is the nearest neighbor of ti
        loss = -torch.log(CX).mean()

        return loss

    def forward(self, target_features, refer_features, args):
        d_total = self.calculate_distance(target_features, refer_features)
        loss = self.calculate_loss(d_total)
        return loss
 

def isNone(x):
    return type(x) is type(None)


def feature_normalize(feature_in):
    feature_norm = torch.norm(feature_in, 2, 1, keepdim=True) + sys.float_info.epsilon
    feature_in_norm = torch.div(feature_in, feature_norm)
    return feature_in_norm, feature_norm


def batch_patch_extraction(image_tensor, kernel_size, stride):
    """ [n, c, h, w] -> [n, np(num_patch), c, k, k] """
    n, c, h, w = image_tensor.shape
    h_out = math.floor((h - (kernel_size-1) - 1) / stride + 1)
    w_out = math.floor((w - (kernel_size-1) - 1) / stride + 1)
    unfold_tensor = F.unfold(image_tensor, kernel_size=kernel_size, stride=stride) # shape (batch_size, channels * kernel_size * kernel_size, h_out * w_out)
    unfold_tensor = unfold_tensor.contiguous().view(
        n, c * kernel_size * kernel_size, h_out, w_out)
    return unfold_tensor # shape (batch_size, channels, h_out, w_out)


def compute_cosine_distance(x, y):
    N, C, _, _ = x.size()

    # to normalized feature vectors
    y_mean = y.view(N, C, -1).mean(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
    x, x_norm = feature_normalize(x - y_mean)  # batch_size * feature_depth * feature_size * feature_size
    y, y_norm = feature_normalize(y - y_mean)  # batch_size * feature_depth * feature_size * feature_size
    x = x.view(N, C, -1)
    y = y.view(N, C, -1)

    # cosine distance = 1 - similarity
    x_permute = x.permute(0, 2, 1)  # batch_size * feature_size^2 * feature_depth

    # convert similarity to distance
    sim = torch.matmul(x_permute, y)
    dist = (1 - sim) / 2 # batch_size * feature_size^2 * feature_size^2

    return dist.clamp(min=0.) # if < 0., set 0.


def compute_l2_distance(x, y):
    N, C, Hx, Wx = x.size()
    _, _, Hy, Wy = y.size()
    x_vec = x.view(N, C, -1)
    y_vec = y.view(N, C, -1)
    x_s = torch.sum(x_vec ** 2, dim=1)
    y_s = torch.sum(y_vec ** 2, dim=1)

    A = y_vec.transpose(1, 2) @ x_vec
    dist = y_s.unsqueeze(2).expand_as(A) - 2 * A + x_s.unsqueeze(1).expand_as(A)
    dist = dist.transpose(1, 2).reshape(N, Hx*Wx, Hy*Wy)
    dist = dist.clamp(min=0.) / C

    return dist


import numpy as np
import matplotlib.pyplot as plt

def visualize_loss(GC_loss, DCT_loss):
    GC_loss_first_100 = GC_loss[:100]
    GC_loss_last_100 = GC_loss[100:]
    DCT_loss_first_100 = DCT_loss[:100]
    DCT_loss_last_100 = DCT_loss[100:]
    plt.figure(figsize=(6, 6))

    # 绘制前100个数据点
    plt.plot(GC_loss_first_100, label='GC(first 100 iterations)', color='green', linestyle='--', linewidth=2)
    plt.plot(DCT_loss_first_100, label='FreNTS(first 100 iterations)', color='red', linestyle='--', linewidth=2)

    # 绘制后100个数据点
    plt.plot(GC_loss_last_100, label='GC(last 100 iterations)', color='green', linewidth=2)
    plt.plot(DCT_loss_last_100, label='FreNTS(last 100 iterations)', color='red', linewidth=2)

    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.5)

    # 添加图例
    # plt.legend(fontsize='large')
    plt.legend(fontsize='large', loc='upper right', bbox_to_anchor=(1.2, 1))

    # 添加标题和标签，并调整字体大小
    plt.title('Similarity Loss at resolution=0.75', fontsize=18)
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Loss', fontsize=18)

    # 调整刻度字体大小
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()


if __name__ == '__main__':
    GC_loss = np.load('loss_path/GC_input/0.75_loss_plot_spatial.npy')
    DCT_loss = np.load('loss_path/DCT_input/0.75_loss_plot_spatial.npy')
    visualize_loss(GC_loss, DCT_loss)

# if __name__ == '__main__':
#     GC_loss = np.load('loss_path/GC_3/1_loss_plot_spatial.npy')
#     DCT_loss = np.load('loss_path/DCT_03/1_loss_plot_spatial.npy')
#     print(DCT_loss.shape)
#     print(DCT_loss.shape)
#     visualize_loss(GC_loss, DCT_loss)