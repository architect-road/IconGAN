# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Frechet Inception Distance (FID) from the paper
"GANs trained by a two time-scale update rule converge to a local Nash
equilibrium". Matches the original implementation by Heusel et al. at
https://github.com/bioinf-jku/TTUR/blob/master/fid.py"""

import numpy as np
import json
import scipy.linalg
from . import metric_utils

#----------------------------------------------------------------------------

def compute_fid(opts, max_real, num_gen):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

    mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real).get_mean_cov()

    mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_generator(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen).get_mean_cov()

    if opts.rank != 0:
        return float('nan')

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fid)

def compute_mfid(opts, max_real, num_gen, only_sty=False):
    labels_shapes = opts.labels_shapes
    label_dim = sum(labels_shapes)
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

    # open flat, hand-drawn, streak theme id lists
    with open('metrics/flat_theme_id.json') as f:
        flat_theme_ids = json.load(f)
    with open('metrics/handdrawn_theme_id.json') as f:
        hand_theme_ids = json.load(f)
    with open('metrics/streak_theme_id.json') as f:
        streak_theme_ids = json.load(f)

    # get all features of real images and fake images
    real_x = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_all=True, max_items=max_real).get_all_all()

    fake_x = metric_utils.compute_feature_stats_for_generator(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=1, capture_all=True, max_items=num_gen).get_all_all()
    feature_dim = real_x.shape[1]-label_dim

    if opts.rank != 0:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
    
    # process all features conditioned on app or style
    real_features, real_labels = np.split(real_x,[feature_dim],axis=1)
    fake_features, fake_labels = np.split(fake_x,[feature_dim],axis=1)
    
    real_styf = [[],[],[]]
    # real_styf = [[] for _ in range(labels_shapes[0])]
    real_appf = [[] for _ in range(labels_shapes[1])]
    for feature, label in zip(real_features, real_labels):
        sty_label, app_label = np.split(label, [labels_shapes[0]])
        app_index = np.flatnonzero(app_label).item()
        sty_index = np.flatnonzero(sty_label).item()
        real_appf[app_index].append(feature)
        if sty_index in flat_theme_ids:
            real_styf[0].append(feature)
        elif sty_index in hand_theme_ids:
            real_styf[1].append(feature)
        else:
            real_styf[2].append(feature)
    fake_styf = [[],[],[]]
    # fake_styf = [[] for _ in range(labels_shapes[0])]
    fake_appf = [[] for _ in range(labels_shapes[1])]
    for feature, label in zip(fake_features, fake_labels):
        sty_label, app_label = np.split(label, [labels_shapes[0]])
        app_index = np.flatnonzero(app_label).item()
        sty_index = np.flatnonzero(sty_label).item()
        fake_appf[app_index].append(feature)
        if sty_index in flat_theme_ids:
            fake_styf[0].append(feature)
        elif sty_index in hand_theme_ids:
            fake_styf[1].append(feature)
        else:
            fake_styf[2].append(feature)

    # calculate fid score for every style
    sty_fids = []
    for i in range(len(real_styf)):
        real_raw_mean = np.zeros([feature_dim], dtype=np.float64)
        real_raw_cov = np.zeros([feature_dim, feature_dim], dtype=np.float64)
        for x in np.array_split(np.stack(real_styf[i]), len(real_styf[i])//64+1):
            x64 = x.astype(np.float64)
            real_raw_mean += x64.sum(axis=0)
            real_raw_cov += x64.T @ x64
        mu_real = real_raw_mean / len(real_styf[i])
        sigma_real = real_raw_cov / len(real_styf[i])
        sigma_real -= np.outer(mu_real, mu_real)
        
        fake_raw_mean = np.zeros([feature_dim], dtype=np.float64)
        fake_raw_cov = np.zeros([feature_dim, feature_dim], dtype=np.float64)
        for x in np.array_split(np.stack(fake_styf[i]), len(fake_styf[i])//64+1):
            x64 = x.astype(np.float64)
            fake_raw_mean += x64.sum(axis=0)
            fake_raw_cov += x64.T @ x64
        mu_fake = fake_raw_mean / len(fake_styf[i])
        sigma_fake = fake_raw_cov / len(fake_styf[i])
        sigma_fake -= np.outer(mu_fake, mu_fake)

        m = np.square(mu_fake - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_fake, sigma_real), disp=False)
        fid = np.real(m + np.trace(sigma_fake + sigma_real - s * 2))
        sty_fids.append(fid)
    
    if only_sty == True:
        sty_mfid = sum(sty_fids) / len(sty_fids)
        app_mfid = 0.0
        return sty_fids[0], sty_fids[1], sty_fids[2], sty_mfid, app_mfid

    # calculate fid score for every app
    app_fids = []
    for i in range(len(real_appf)):
        real_raw_mean = np.zeros([feature_dim], dtype=np.float64)
        real_raw_cov = np.zeros([feature_dim, feature_dim], dtype=np.float64)
        for x in np.array_split(np.stack(real_appf[i]), len(real_appf[i])//64+1):
            x64 = x.astype(np.float64)
            real_raw_mean += x64.sum(axis=0)
            real_raw_cov += x64.T @ x64
        mu_real = real_raw_mean / len(real_appf[i])
        sigma_real = real_raw_cov / len(real_appf[i])
        sigma_real -= np.outer(mu_real, mu_real)

        fake_raw_mean = np.zeros([feature_dim], dtype=np.float64)
        fake_raw_cov = np.zeros([feature_dim, feature_dim], dtype=np.float64)
        for x in np.array_split(np.stack(fake_appf[i]), len(fake_appf[i])//64+1):
            x64 = x.astype(np.float64)
            fake_raw_mean += x64.sum(axis=0)
            fake_raw_cov += x64.T @ x64
        mu_fake = fake_raw_mean / len(fake_appf[i])
        sigma_fake = fake_raw_cov / len(fake_appf[i])
        sigma_fake -= np.outer(mu_fake, mu_fake)

        m = np.square(mu_fake - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_fake, sigma_real), disp=False)
        fid = np.real(m + np.trace(sigma_fake + sigma_real - s * 2))
        app_fids.append(fid)
    
    sty_mfid = sum(sty_fids) / len(sty_fids)
    app_mfid = sum(app_fids) / len(app_fids)
    return sty_fids[0], sty_fids[1], sty_fids[2], sty_mfid, app_mfid

#----------------------------------------------------------------------------
