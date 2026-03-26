"""
"Neural Texture Systhesis in Frequency Domain"
The code is based on the work of Guided-Correspondence-Loss.
"""
import os
import utils
import argparse
from time import time
import numpy as np
import torch
import torch.nn.functional as F
from vgg_model import VGG19Model
from loss_fn import AGCLoss_forward
from DCT_transformer import read_image2DCT, GradientCalculator, calculate_DCT_image
import matplotlib.pyplot as plt

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_name = os.path.basename(args.image_path)[:-4]
    args.suffix = args.suffix + f'_resize({args.size})_out_size1({args.output_size[0]})_out_size2({args.output_size[1]})_patch_size({args.patch_size})_lambda_occ({args.lambda_occ})'
    args.output_folder = args.output_folder + '/' + image_name + args.suffix
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    SIZE = args.size
    OUTPUT_FILE = args.output_folder + '/' + args.output_name
    INPUT_FILE = args.image_path
    NB_ITER = args.base_iters + args.finetune_iters

    base_layers = args.base_layers
    fineTune_layers = args.finetune_layers
    scales = args.scales

    print('-------------------------Neural Texture Systhesis---------------------------')
    print(f'Launching texture synthesis from {INPUT_FILE} on size {SIZE} for {NB_ITER} steps. Output file: {OUTPUT_FILE}')
    print('  ************************Parameters information************************    ')
    print('input file:', INPUT_FILE)
    print('out file:', OUTPUT_FILE)
    print('output size:', args.output_size)
    print('h:', args.h)
    print('lambda_occ:', args.lambda_occ)
    print('t_1:', args.t_1)
    print('t_2:', args.t_2)
    print('lr:', args.lr)
    print('patch_size:', args.patch_size)
    print('use_DCT:', args.use_DCT)
    print('loss_save:', args.loss_save)

    print('--------------------------------------')

    # ---------------------------------------------------------------------------------
    # DATA - SOURCE AND TARGET (IMAGE)
    # ---------------------------------------------------------------------------------
    with torch.no_grad():
        # -----------------------------------------------
        # load source.
        # -----------------------------------------------
        refer_texture = utils.decode_image(INPUT_FILE, size=SIZE).to(device)
        utils.save_image(args.output_folder + '/refer-texture.jpg', refer_texture)

        # -----------------------------------------------
        # Initialize the target.
        # -----------------------------------------------
        output_size = args.output_size
        target_texture = torch.rand([1, 3, *output_size]).to(device)
        utils.save_image(args.output_folder + '/target-init.jpg', target_texture)

    # ---------------------------------------------------------------------------------
    # FIT FUNCTION
    # ---------------------------------------------------------------------------------
    # Modules setup
    extractor = VGG19Model(device, 3)  # customized_vgg_model in pytorch
    Gradient_extractor = GradientCalculator().to(device) # Gradient

    # spatial domain AGC loss
    spatial_loss_forward = AGCLoss_forward(
        h=args.h,
        patch_size=args.patch_size,
        lambda_occ=args.lambda_occ,
        t = args.t_1,
    ).to(device)

    def saptial_get_loss(target_feats, refer_feats, args, layers=[]):
        loss = 0
        base_loss = 0
        for layer in layers:
            temp_loss = spatial_loss_forward(target_feats[layer], refer_feats[layer], args)
            loss += temp_loss
        return loss

    # frequency domain AGC loss
    frequency_loss_forward = AGCLoss_forward(
        h=args.h,
        patch_size=args.patch_size,
        lambda_occ=args.lambda_occ,
        t = args.t_2
    ).to(device)

    def frequency_get_loss(target_feats, refer_feats, args, layers=[]):
        loss = 0
        base_loss = 0
        for layer in layers:
            temp_loss = frequency_loss_forward(target_feats[layer], refer_feats[layer], args)
            loss += temp_loss
        return loss
    # Optimization
    target_sizes = [[int(output_size[0] * scale), int(output_size[1] * scale)] for scale in scales]

    print('Start fitting...')
    start = time()
    # train in the frequency domain
    print('frequency domain.....')
    if args.use_DCT:
        for target_scale, target_size in zip(scales, target_sizes):
            target_texture = F.interpolate(target_texture, target_size).detach()
            optimizer = torch.optim.Adam([target_texture.requires_grad_()], lr=args.lr)
            refer_texture2 = F.interpolate(refer_texture, scale_factor=target_scale)
            refer_feats = [] 
            refer_texture_DCT = read_image2DCT(refer_texture2)
            # refer_texture_DCT = calculate_DCT_image(refer_texture2)
            refer_feats.append(refer_texture_DCT)
            # if args.use_Gradient:
            #     refer_texture_gradient = Gradient_extractor(refer_texture)
            #     refer_feats.append(refer_texture_gradient)
            for iter in range(NB_ITER):
                optimizer.zero_grad()
                target_texture.data.clamp_(0, 1)
                layers = [0]
                target_feats = []
                target_texture_DCT = read_image2DCT(target_texture)
                # target_texture_DCT = calculate_DCT_image(target_texture)
                target_feats.append(target_texture_DCT)
                # if args.use_Gradient:
                    # target_texture_gradient = Gradient_extractor(target_texture)
                    # target_feats.append(target_texture_gradient)
                loss = frequency_get_loss(target_feats, refer_feats, args=args, layers=layers)
                loss.backward()
                optimizer.step()

                if iter % args.save_freq == 0:
                    target_texture.data.clamp_(0, 1)
                    print(f'scale {target_scale} iter {iter + 1} loss {loss.item()}')
                    utils.save_image(args.output_folder + f'/step1-output-scale{target_scale}-iter{iter + 1}.jpg', target_texture)

        target_texture.data.clamp_(0, 1)
        output_file_temp =  args.output_folder + '/' + 'frequency.jpg'
        utils.save_image(output_file_temp, target_texture)


    # train in the spatial domain
    print('spatial domain.....')
    for target_scale, target_size in zip(scales, target_sizes):
        target_texture = F.interpolate(target_texture, target_size).detach()
        optimizer = torch.optim.Adam([target_texture.requires_grad_()], lr=args.lr)
        refer_texture2 = F.interpolate(refer_texture, scale_factor=target_scale)
        refer_feats = [feat.detach() for feat in extractor(refer_texture2)] # len=14  2+2+4+4+2
        for iter in range(NB_ITER):
            optimizer.zero_grad()
            target_texture.data.clamp_(0, 1)
            if iter >= args.base_iters:
                layers = fineTune_layers
            else:
                layers = base_layers
            target_feats = extractor(target_texture)
            loss = saptial_get_loss(target_feats, refer_feats, args=args, layers=layers)
            loss.backward()
            optimizer.step()
            if (iter+1) % args.save_freq == 0:
                target_texture.data.clamp_(0, 1)
                print(f'scale {target_scale} iter {iter + 1} loss {loss.item()}')
                utils.save_image(args.output_folder + f'/step2-output-scale{target_scale}-iter{iter + 1}.jpg', target_texture)

    target_texture.data.clamp_(0, 1)
    utils.save_image(OUTPUT_FILE, target_texture)
    end = time()
    print('Time: {:} minutes'.format((end - start) / 60.0))
            

if __name__ == '__main__':
    # parser
    parser = argparse.ArgumentParser(description='Neural Texture Systhesis')
    parser.add_argument('--suffix', default='', help='output suffix')
    parser.add_argument('--save_freq', default=100, type=int, help='save frequency')
    parser.add_argument('--base_iters', default=100, type=int, help='number of steps') # 300
    parser.add_argument('--finetune_iters', default=100, type=int, help='number of finetune steps') # 200

    parser.add_argument('--image_path', default='data/source/test01.png', type=str, help='data path')
    parser.add_argument('--output_folder', default='./outputs/texture_synthesis', type=str, help='output folder')
    parser.add_argument('--output_name', default='output.jpg', help='name of the output file')
    parser.add_argument('--output_size', type=int, nargs='+', default=[512, 512], help='output size')
    parser.add_argument('--size', default=256, type=int, help='resolution of the input texture (it will resize to this resolution)')
    parser.add_argument('--scales', type=float, nargs='+', default=[0.25, 0.5, 0.75, 1], help='multi-scale generation')
    parser.add_argument('--base_layers', type=int, nargs='+', default=[2, 4, 8], help='layers used in vgg')
    parser.add_argument('--finetune_layers', type=int, nargs='+', default=[0, 2, 4, 8], help='layers used in vgg')

    parser.add_argument('--use_DCT', action='store_false', help='Activate the DCT')
    parser.add_argument('--loss_save', action='store_true', help='Activate the loss save')
    # parser.add_argument('--use_Gradient', action='store_true', help='Activate the Gradient') 

    # The hyperparameters
    parser.add_argument('--h', type=float, default=2, help='h')
    parser.add_argument('--lambda_occ', type=float, default=0.05, help='lambda of occ')
    parser.add_argument('--t_1', type=float, default=0.6, help='t_1 is t of vgg feature, spatial domain')
    parser.add_argument('--t_2', type=float, default=0.1, help='t_2 is t of DCT, frequency domain')
    parser.add_argument('--lr', type=float, default=0.01, help='learing rate')
    parser.add_argument('--patch_size', type=int, default=7, help='patch size')

    args = parser.parse_args()

    train(args)
    