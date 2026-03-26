import torch_dct as DCT
from torch import nn
import torch
import torch.nn.functional as F


def read_image2DCT(image):
    num_batchsize = image.shape[0]
    device = image.device

    m = image.shape[2]
    n = image.shape[3]
    m = (m // 8) * 8
    n = (n // 8) * 8

    image = image[:,:,:m,:n]
    image = image.reshape(num_batchsize, 3, m // 8, 8, n // 8, 8).permute(0, 2, 4, 1, 3, 5)
    DCT_x = DCT.dct_2d(image, norm='ortho')
    DCT_x = DCT_x.reshape(num_batchsize, m // 8, n // 8, -1).permute(0, 3, 1, 2)
    # DCT_x = torch.log(1 + torch.abs(DCT_x + 1e-8))
    DCT_x = torch.log(1 + torch.abs(DCT_x))

    return DCT_x


def dct_transform(patch):
    return DCT.dct_2d(patch, norm='ortho')



def calculate_DCT_image(image, DCT_size=8, DCT_step=1):
    num_batchsize, _, height, width = image.size()
    device = image.device

    unfolded_patches = image.unfold(2, DCT_size, DCT_step).unfold(3, DCT_size, DCT_step)
    num_batchsize, channel, num_patches_height, num_patches_width, _, _ = unfolded_patches.size()
    dct_image = dct_transform(unfolded_patches)

    dct_image = dct_image.reshape(num_batchsize, channel, num_patches_height, num_patches_width, -1).permute(0, 1, 4, 2, 3)

    dct_image = dct_image.reshape(num_batchsize, -1, num_patches_height, num_patches_width)

    dct_image = torch.log(1 + torch.abs(dct_image))
    return dct_image


class GradientCalculator(nn.Module):
    def __init__(self):
        super(GradientCalculator, self).__init__()
        self.conv_kernel_x = nn.Parameter(torch.tensor([[-1., 0., 1.]]).view(1, 1, 1, 3), requires_grad=False)
        self.conv_kernel_y = nn.Parameter(torch.tensor([[-1.], [0.], [1.]]).view(1, 1, 3, 1), requires_grad=False)

    def forward(self, img):
        conv_results = []
        for channel in range(img.shape[1]):
            conv_result_x = F.conv2d(img[:, channel:channel+1, :, :], self.conv_kernel_x, padding=(0, 1))
            conv_result_y = F.conv2d(img[:, channel:channel+1, :, :], self.conv_kernel_y, padding=(1, 0))
            conv_results.append(torch.sqrt(conv_result_x**2 + conv_result_y**2 + 1e-8))
            # conv_results.append(conv_result_x**2 + conv_result_y**2 + 1e-8)

        gradient_magnitude = torch.cat(conv_results, dim=1)
        return gradient_magnitude

if __name__ == '__mian__':
    radius = 21
    batch = 1
    size = 256