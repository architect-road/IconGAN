# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import enum
import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy

#----------------------------------------------------------------------------
def choose_gpu():
    "automatically choose the max memory GPU id"
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    with open('tmp', 'r') as f:
       memory_gpu = [int(x.split()[2]) for x in f.readlines()]
    max_free = np.argmax(memory_gpu)
    os.system('rm tmp')
    print("auto choose gpu %d free %.2f GB"%(max_free, memory_gpu[max_free]/1000.0))
    return str(max_free)

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

def save_image_grid(images, fname, gh_num, gw_num):
    """
        images: numpy images list
        fname: save output name
        gh_num: global height num
        gw_num: global width num
    """
    assert len(images) == gh_num * gw_num
    h,w,c = images[0].shape
    images = np.asarray(images) # [gh_num*gw_num, h, w, c]
    images = images.reshape(gh_num, gw_num, h, w, c)
    images = images.transpose(0, 2, 1, 3, 4) # [gh_num, h, gw_num, w, c]
    images = images.reshape(h*gh_num, w*gw_num, c)
    PIL.Image.fromarray(images, 'RGBA').save(fname)

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', default='training-runs/012-icons-both_split-fixed-p0.5_patch_sim_contra/network-snapshot-010886.pkl',help='Network pickle filename')
@click.option('--seeds', type=num_range, default='1,2', help='random seeds for generating')
@click.option('--styles', type=num_range, help='List of generated styles')
@click.option('--apps', type=num_range, help='List of generated apps')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', type=str, default='results/icons', help='Where to save the output images')
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    seeds: Optional[List[int]],
    styles: Optional[List[int]],
    apps: Optional[List[int]],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
):
    if styles is None:
        styles = [5, 45, 46] # flat
        styles += [1, 30, 44] # hand-drawn
        styles += [18, 25, 83] # streak
    if apps is None:
        apps =[0, 1, 2]
    # loading network
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device(f'cuda:0')
    with dnnlib.util.open_url(network_pkl) as f:
        network_dict = legacy.load_network_pkl(f)
        G = network_dict['G_ema'].to(device) 
    
    # set dataset
    dataset_kwargs = dnnlib.EasyDict(network_dict['training_set_kwargs'])
    dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    label_shape = dataset.label_shape
    labels_shapes = dataset.labels_shapes

    # make output directory
    os.makedirs(outdir, exist_ok=True)

    # load dataset images
    real_images = []
    print("saving dataset images...")
    sty_id_dict = dict() # label => [idx, ...]
    for idx in range(len(dataset)):
        sty_id = dataset.get_details(idx).raw_label[0]
        if sty_id not in sty_id_dict:
            sty_id_dict[sty_id] = []
        sty_id_dict[sty_id].append(idx)
    
    for sty_id in styles:
        for app_id in apps:
            the_app_ids = [dataset.get_details(idx).raw_label[1] for idx in sty_id_dict[sty_id]]
            if app_id in the_app_ids:
                idx = sty_id_dict[sty_id][the_app_ids.index(app_id)]
                real_images.append(dataset[idx][0].transpose(1,2,0))
                # PIL.Image.fromarray(dataset[idx][0].transpose(1,2,0), 'RGBA').save(f'{outdir}/real/sty{sty_id}/{app_id}.png')
            else:
                real_images.append(np.zeros_like(dataset[0][0]).transpose(1,2,0))
    save_image_grid(real_images, f'{outdir}/real_images.png', len(styles), len(apps))
    
    # Generate images.
    for j, seed in enumerate(seeds):
        print(f"generating seed{seed} images... [{j}/{len(seeds)}]")
        fake_images = []
        for i, sty_id in enumerate(styles):
            os.makedirs(f'{outdir}/single/sty{sty_id}', exist_ok=True)
            print(f"\tgenerating style{sty_id} images... [{i}/{len(styles)}]")
            for app_id in apps:
                label = torch.zeros([1,sum(labels_shapes)]).to(device)
                label[0][sty_id] = 1
                label[0][labels_shapes[0]+app_id] = 1
                z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
                img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                fake_images.append(img[0].cpu().numpy())
                PIL.Image.fromarray(img[0].cpu().numpy(), 'RGBA').save(f'{outdir}/single/sty{sty_id}/{app_id}_{seed}.png')
        save_image_grid(fake_images, f'{outdir}/fake_images_{seed}.png', len(styles), len(apps))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------