# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# @File    :   lpips.py
# @Time    :   2020/12/24 14:47:53
# @Author  :   yinpeng


import os
import torch
import torch.nn as nn
from torchvision import models


def normalize(x, eps=1e-10):
    return x * torch.rsqrt(torch.sum(x**2, dim=1, keepdim=True) + eps)


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = models.alexnet(pretrained=True).features
        self.channels = []
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                self.channels.append(layer.out_channels)

    def forward(self, x):
        fmaps = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                fmaps.append(x)
        return fmaps


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False))

    def forward(self, x):
        return self.main(x)


class LPIPS(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.alexnet = AlexNet()
        self.lpips_weights = nn.ModuleList()
        for channels in self.alexnet.channels:
            self.lpips_weights.append(Conv1x1(channels, 1))
        self._load_lpips_weights(device)
        # imagenet normalization for range [-1, 1]
        self.mu = torch.tensor([-0.03, -0.088, -0.188]).view(1, 3, 1, 1).to(device)
        self.sigma = torch.tensor([0.458, 0.448, 0.450]).view(1, 3, 1, 1).to(device)

    def _load_lpips_weights(self, device):
        own_state_dict = self.state_dict()
        state_dict = torch.load('lpips_weights.ckpt',map_location=device)
        for name, param in state_dict.items():
            if name in own_state_dict:
                own_state_dict[name].copy_(param)

    def forward(self, x, y):
        x = (x - self.mu) / self.sigma
        y = (y - self.mu) / self.sigma
        x_fmaps = self.alexnet(x)
        y_fmaps = self.alexnet(y)
        lpips_value = 0
        for x_fmap, y_fmap, conv1x1 in zip(x_fmaps, y_fmaps, self.lpips_weights):
            x_fmap = normalize(x_fmap)
            y_fmap = normalize(y_fmap)
            lpips_value += torch.mean(conv1x1((x_fmap - y_fmap)**2))
        return lpips_value


@torch.no_grad()
def calculate_lpips_given_images(args, group_of_images, device=None):
    lpips = LPIPS(device).eval().to(device)
    lpips_values = []
    num_rand_outputs = len(group_of_images)

    # calculate the average of pairwise distances among all random outputs
    for i in range(num_rand_outputs-1):
    # calculate the average of first and other image pair lpips distances
        # for j in range(i+1, num_rand_outputs):
            lpips_values.append(lpips(group_of_images[i].to(device), group_of_images[-1].to(device)))
    if args.max:
        lpips_value = torch.max(torch.stack(lpips_values, dim=0))
    else:
        lpips_value = torch.mean(torch.stack(lpips_values, dim=0))
    return lpips_value.item()

if __name__ == '__main__':
    from PIL import Image
    from torchvision import transforms
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--name', type=str, default='ours')
    parser.add_argument('--style',type=str, default='flat')
    parser.add_argument('--max', action='store_true')
    args = parser.parse_args()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(std=(0.5,0.5,0.5),mean=(0.5,0.5,0.5))])

    if args.style == 'flat':
        styles = [5, 45, 46, 85, 108, 113, 133, 142, 181]
    elif args.style == 'hand-drawn':
        styles = [1, 30, 44, 61, 82, 90, 99, 103, 130, 149, 162, 192]
    elif args.style == 'streak':
        styles = [18, 25, 83, 91, 107, 110, 210, 212]
    else:
        assert False, "arg style must in flat, hand-drawn, streak!"
    
    directory = f'../results/icons/{args.name}/single'
    mean_lpips_value = 0
    sum_lpips_value = 0
    count = 0
    for sty in os.listdir(directory):
        if int(sty[3:]) not in styles: continue
        print(sty)
        image_names = os.listdir(f'{directory}/{sty}')
        app_ids = [name.split('_')[0] for name in image_names]
        app_ids = list(set(app_ids))
        for app_id in app_ids:
            group_names = [im for im in image_names if im.startswith(app_id)]
            group_of_images = [transform(Image.open(f'{directory}/{sty}/{name}').convert('RGB')) for name in group_names]
            lpips_value = calculate_lpips_given_images(args, group_of_images, device=torch.device('cuda:4'))
            count += 1
            sum_lpips_value += lpips_value
            mean_lpips_value = sum_lpips_value/count

            print(lpips_value, mean_lpips_value)
    
    save_dict = dict(args._get_kwargs())
    save_dict['mLPIPS'] = mean_lpips_value
    with open('lpips.json', 'a') as f:
        f.write('\n')
        json.dump(save_dict, f)

# ours max [flat: 0.4061, hand-drawn: 0.2276, streak: 0.1904]
# ours min []