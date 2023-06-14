# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
import torch.nn.functional as F
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, labels_shapes, sim_loss_flag, contra_loss_flag, decouple_flag, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

# my loss,n: the calculated patches number
def patch_similarity_loss(x, n):
    # normalize patch feature
    x = torch.nn.functional.normalize(x, dim=1)
    # caculate the distance among patches
    perm = torch.randperm(x.size(2)*x.size(3)).to(x.device)
    x1 = x.reshape(x.size(0), x.size(1), -1) # [B, C, W*H]
    x1 = x1[:,:,perm[:n]] # [B, C, n]
    x2 = x1.transpose(1, 2) # [B, n, C]
    sim_matrix = torch.bmm(x2, x1) * (1 / np.sqrt(x.size(1))) # [B,n,n]
    mask_matrix = torch.eye(n).repeat([x.size(0),1,1]).to(x.device)
    sim_matrix = sim_matrix - mask_matrix * sim_matrix # remove diagonal elements
    sim_loss = sim_matrix.sum(dim=(1,2)) / (n*(n-1))
    return sim_loss

class GatherLayer(torch.autograd.Function):
    """
    This file is copied from
    https://github.com/open-mmlab/OpenSelfSup/blob/master/openselfsup/models/utils/gather_layer.py
    Gather tensors from all process, supporting backward propagation
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[torch.distributed.get_rank()]
        return grad_out

class ContrastiveLoss:
    def __init__(self, mode, thre=0.2, temp=2, patch_num=8, ddp=True):
        self.thre = thre
        self.ddp = ddp
        self.mode = mode
        self.temp = temp
        self.patch_num = patch_num

    def same_app_index(self, label, labels_shapes):
        same_app_list = [[] for _ in range(labels_shapes[1])]
        sty_label, app_label = label.split(labels_shapes, dim=1)
        for i in range(label.size(0)):
            index = torch.nonzero(app_label[i]).item()
            same_app_list[index].append(i)
        return same_app_list

    def same_sty_index(self, label, labels_shapes):
        same_sty_list = [[] for _ in range(labels_shapes[0])]
        sty_label, app_label = label.split(labels_shapes, dim=1)
        for i in range(label.size(0)):
            index = torch.nonzero(sty_label[i]).item()
            same_sty_list[index].append(i)
        return same_sty_list

    def compute(self, x, label, labels_shapes):
        if self.ddp:
            x = torch.cat(GatherLayer.apply(x), dim=0)
            label = torch.cat(GatherLayer.apply(label), dim=0)
        if x.ndim == 4:
            # use mean feature as feature of the image
            # x = x.mean(dim=(2,3))
            x = x[:,:,x.size(2)//2,x.size(3)//2]
        # feature normalization
        x_norm = F.normalize(x)
        # get indexes
        indexes_app = self.same_app_index(label, labels_shapes)
        indexes_sty = self.same_sty_index(label, labels_shapes)

        # # calculate contrastive loss for cross negative samples
        # if self.mode == 'app':
        #     indexes = indexes_app
        # else:
        #     indexes = indexes_sty
        # # alignment loss
        # align_loss = count_align = 0
        # for index in indexes:
        #     if len(index) > 1:
        #         loss_temp = torch.pdist(x_norm[index], p=2).pow(2)
        #         align_loss += F.relu(loss_temp - self.thre).mean()
        #         count_align += 1
        # align_loss = align_loss / count_align if count_align != 0 else 0 # 0 - 4
        # # uniformity loss
        # index = [i[0] for i in indexes if len(i) > 0]
        # loss_temp = torch.pdist(x_norm[index], p=2).pow(2)
        # uniformity_loss = loss_temp.mul(-self.temp).exp().mean().log()# -8 - 0

        # calculate contrastive loss
        align_loss = uniformity_loss = 0
        count_align = count_uniform = 0
        for index in indexes_app: # for the same app label
            # alignment loss
            if self.mode == 'app' and len(index) > 1:
                loss_temp = torch.pdist(x_norm[index], p=2).pow(2)
                align_loss += F.relu(loss_temp - self.thre).mean()
                count_align += 1
            # uniformity loss
            if self.mode == 'sty' and len(index) > 1:
                loss_temp = torch.pdist(x_norm[index], p=2).pow(2)
                uniformity_loss += loss_temp.mul(-self.temp).exp().mean().log()
                count_uniform += 1
        for index in indexes_sty: # for the same style label
            # alignment loss
            if self.mode == 'sty' and len(index) > 1:
                loss_temp = torch.pdist(x_norm[index], p=2).pow(2)
                align_loss += F.relu(loss_temp - self.thre).mean()
                count_align += 1
            # uniformity loss
            if self.mode == 'app' and len(index) > 1:
                loss_temp = torch.pdist(x_norm[index], p=2).pow(2)
                uniformity_loss += loss_temp.mul(-self.temp).exp().mean().log()
                count_uniform += 1
        # 0 - 4
        align_loss = align_loss / count_align if count_align != 0 else 0
        # -8 - 0
        uniformity_loss = uniformity_loss / count_uniform if count_uniform != 0 else 0
        # contra loss: [-0.5, 1]
        contra_loss = (align_loss + uniformity_loss+4) / 8
        return contra_loss

class Data2DataCrossEntropyLoss(torch.nn.Module):
    def __init__(self, temperature=0.75, m_p=0.98, DDP=True, mode='both'):
        super(Data2DataCrossEntropyLoss, self).__init__()
        self.temperature = temperature
        self.m_p = m_p
        self.DDP = DDP
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        self.mode = mode

    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix

    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def make_index_matrix(self, labels):
        labels1 = labels.clone().unsqueeze(0).repeat(labels.size(0), 1, 1)
        labels2 = labels.clone().unsqueeze(1).repeat(1, labels.size(0), 1)
        mask = torch.sum(torch.abs((labels1 - labels2)), dim=2) # [BxB]
        mask[mask>0] = 1
        return mask.type(torch.long)

    def remove_diag(self, M):
        h, w = M.shape
        assert h==w, "h and w should be same"
        mask = np.ones((h, w)) - np.eye(h)
        mask = torch.from_numpy(mask)
        mask = (mask).type(torch.bool)
        return M[mask].view(h, -1)

    def forward(self, embed, proxy, label, labels_shapes, **_):
        # If train a GAN throuh DDP, gather all data on the master rank
        if self.DDP:
            embed = torch.cat(GatherLayer.apply(embed), dim=0)
            proxy = torch.cat(GatherLayer.apply(proxy), dim=0)
            label = torch.cat(GatherLayer.apply(label), dim=0)
        if embed.ndim == 4:
            embed = embed[:,:,embed.size(2)//2,embed.size(3)//2]
        
        sty_c, app_c = label.split(labels_shapes, dim=1)
        if self.mode == 'both': label = label
        if self.mode == 'app': label = app_c
        if self.mode == 'sty': label = sty_c

        # calculate similarities between sample embeddings
        sim_matrix = self.calculate_similarity_matrix(embed, embed) + self.m_p - 1
        # remove diagonal terms
        sim_matrix = self.remove_diag(sim_matrix/self.temperature)
        # for numerical stability
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = F.relu(sim_matrix) - sim_max.detach()

        # calculate similarities between sample embeddings and the corresponding proxies
        smp2proxy = self.cosine_similarity(embed, proxy)
        # make false negative removal
        removal_fn = self.remove_diag(self.make_index_matrix(label))
        # apply the negative removal to the similarity matrix
        improved_sim_matrix = removal_fn*torch.exp(sim_matrix)

        # compute positive attraction term
        pos_attr = F.relu((self.m_p - smp2proxy)/self.temperature)
        # compute negative repulsion term
        neg_repul = torch.log(torch.exp(-pos_attr) + improved_sim_matrix.sum(dim=1))
        # compute data to data cross-entropy criterion
        criterion = pos_attr + neg_repul
        return criterion.mean()


class StyleGAN2Loss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D=None, D_app=None, D_sty=None,
                 C_app=None, C_sty=None, app_augment_pipe=None, sty_augment_pipe=None,
                 both_augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, 
                 pl_batch_shrink=2, pl_decay=0.01, pl_weight=2, num_gpus=4):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.D_app = D_app
        self.D_sty = D_sty
        self.C_app = C_app
        self.C_sty = C_sty
        self.app_augment_pipe = app_augment_pipe
        self.sty_augment_pipe = sty_augment_pipe
        self.both_augment_pipe = both_augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.app_contra_dis = ContrastiveLoss(mode='app', thre=0.25, ddp=(num_gpus>1))
        self.app_contra_gen = ContrastiveLoss(mode='app', thre=0.25, ddp=(num_gpus>1))
        self.sty_contra_dis = ContrastiveLoss(mode='sty', thre=0.1, ddp=(num_gpus>1))
        self.sty_contra_gen = ContrastiveLoss(mode='sty', thre=0.1, ddp=(num_gpus>1))
        self.d2d_cross_entropy_sty = Data2DataCrossEntropyLoss(mode='sty')
        self.d2d_cross_entropy_app = Data2DataCrossEntropyLoss(mode='app')
        self.d2d_cross_entropy_both = Data2DataCrossEntropyLoss(mode='both')

    def run_G(self, z, c, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws)
        return img, ws

    def run_D(self, img, c, labels_shapes, sync):
        if self.D is not None:
            if self.both_augment_pipe is not None:
                img = self.both_augment_pipe(img)
            with misc.ddp_sync(self.D, sync):
                (logits, x), cmap = self.D(img, c)
            return logits, x, cmap
        else:
            sty_c, app_c = c.split(labels_shapes, dim=1)
            img1 = img2 = img
            if self.app_augment_pipe is not None:
                img1 = self.app_augment_pipe(img)
            with misc.ddp_sync(self.D_app, sync):
                (logits_app, x_app), cmap_app = self.D_app(img1, app_c)
            if self.sty_augment_pipe is not None:
                img2 = self.sty_augment_pipe(img)
            with misc.ddp_sync(self.D_sty, sync):
                (logits_sty, x_sty), cmap_sty = self.D_sty(img2, sty_c)
            logits = logits_app + logits_sty
            return logits, x_app, x_sty, cmap_app, cmap_sty

    def run_C(self, x_sty, x_app, sync):
        with misc.ddp_sync(self.C_app, sync):
            if x_sty.ndim == 4:
                # x_sty = x_sty.mean(dim=(2,3))
                x_sty = x_sty[:,:,x_sty.size(2)//2,x_sty.size(3)//2]
            score_app = self.C_app(x_sty) # let C_app cannot classify x_sty
        with misc.ddp_sync(self.C_sty, sync):
            score_sty = self.C_sty(x_app) # let C_sty cannot classify x_app
        return score_app, score_sty

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, labels_shapes, sim_loss_flag, contra_loss_flag, decouple_flag, d2d_loss_flag, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth', 'Cboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Cmain = (phase == 'Cboth') and (decouple_flag != False)
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                if self.D is not None:
                    logits,x,cmap = self.run_D(gen_img, gen_c, labels_shapes, sync=False)
                    loss_adv = torch.nn.functional.softplus(-logits).mean()
                    if d2d_loss_flag:
                        loss_d2d = self.d2d_cross_entropy_both(x, cmap, gen_c, labels_shapes)
                    else:
                        loss_d2d = 0
                    loss_contra = 0
                else:
                    logits, x_app, x_sty, cmap_app, cmap_sty = self.run_D(gen_img, gen_c, labels_shapes, sync=False)
                    loss_sim = patch_similarity_loss(x_sty, 8) if sim_loss_flag else 0
                    loss_adv = torch.nn.functional.softplus(-(logits+loss_sim)).mean()
                    if d2d_loss_flag:
                        loss_d2d = self.d2d_cross_entropy_app(x_app, cmap_app, gen_c, labels_shapes) + \
                                   self.d2d_cross_entropy_sty(x_sty, cmap_sty, gen_c, labels_shapes)
                    else:
                        loss_d2d = 0
                    if contra_loss_flag:
                        loss_contra = self.app_contra_gen.compute(x_app, gen_c, labels_shapes) + \
                                      self.sty_contra_gen.compute(x_sty, gen_c, labels_shapes)
                    else:
                        loss_contra = 0
                training_stats.report('Loss/scores/fake_score', logits)
                training_stats.report('Loss/signs/fake_sign', logits.sign())
                loss_Gmain = loss_adv + loss_contra + loss_d2d
                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
                if self.D is not None:
                    logits, x, cmap = self.run_D(gen_img, gen_c, labels_shapes, sync=False)
                else:
                    logits, x_app, x_sty, cmap_app, cmap_sty = self.run_D(gen_img, gen_c, labels_shapes, sync=False)
                training_stats.report('Loss/scores/fake_score', logits)
                training_stats.report('Loss/signs/fake_sign', logits.sign())
                loss_Dgen = torch.nn.functional.softplus(logits).mean() # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                if self.D is not None:
                    logits, x, cmap = self.run_D(real_img_tmp, real_c, labels_shapes, sync=sync)
                    loss_adv = torch.nn.functional.softplus(-logits).mean()
                    if d2d_loss_flag:
                        loss_d2d = self.d2d_cross_entropy_both(x, cmap, real_c, labels_shapes)
                    else:
                        loss_d2d = 0
                    loss_cla = loss_contra = 0
                else:
                    logits, x_app, x_sty, cmap_app, cmap_sty = self.run_D(real_img_tmp, real_c, labels_shapes, sync=sync)
                    loss_sim = patch_similarity_loss(x_sty, 8) if sim_loss_flag else 0
                    loss_adv = torch.nn.functional.softplus(-(logits+loss_sim)).mean()
                    if d2d_loss_flag:
                        loss_d2d = self.d2d_cross_entropy_app(x_app, cmap_app, real_c, labels_shapes) + \
                                   self.d2d_cross_entropy_sty(x_sty, cmap_sty, real_c, labels_shapes)
                    else:
                        loss_d2d = 0
                    if decouple_flag:
                        # calculate classifer scores
                        score_app, score_sty = self.run_C(x_sty, x_app, sync=False)
                        loss_cla_app = score_app.std(dim=1).mean()
                        loss_cla_sty = score_sty.std(dim=1).mean()
                        loss_cla = (loss_cla_app + loss_cla_sty) / 2.0 # opposite direction
                    else:
                        loss_cla = 0
                    if contra_loss_flag:
                        loss_contra = self.app_contra_dis.compute(x_app, real_c, labels_shapes) + \
                                      self.sty_contra_dis.compute(x_sty, real_c, labels_shapes)
                    else:
                        loss_contra = 0
                training_stats.report('Loss/scores/real_score', logits)
                training_stats.report('Loss/signs/real_sign', logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = loss_adv + loss_contra + loss_cla + loss_d2d
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3]).mean()
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)
            with torch.autograd.profiler.record_function(name + '_backward'):
                (logits.mean() * 0 + loss_Dreal + loss_Dr1).mul(gain).backward()
        
        if do_Cmain:
            name = 'C_real'
            # calculate classifer scores
            logits, x_app, x_sty, cmap = self.run_D(real_img.detach(), real_c, labels_shapes, sync=False)
            score_app, score_sty = self.run_C(x_sty.detach(), x_app.detach(), sync=sync)
            sty_labels, app_labels = real_c.split(labels_shapes, dim=1)
            sty_ids = torch.nonzero(sty_labels)[:, 1]
            app_ids = torch.nonzero(app_labels)[:, 1]

            loss_cla_app = F.cross_entropy(score_app, app_ids)
            loss_cla_sty = F.cross_entropy(score_sty, sty_ids)
            loss_cla = (loss_cla_app + loss_cla_sty) / 2.0

            with torch.autograd.profiler.record_function(name + '_backward'):
                loss_cla.mul(gain).backward()

#----------------------------------------------------------------------------
