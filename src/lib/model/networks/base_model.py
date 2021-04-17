from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
import torch.nn.functional as F

from .DCNv2.dcn_v2 import DCN_TraDeS
import numpy as np
from ..utils import _sigmoid

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

BN_MOMENTUM = 0.1
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class BaseModel(nn.Module):
    def __init__(self, heads, head_convs, num_stacks, last_channel, opt=None):
        super(BaseModel, self).__init__()
        if opt is not None and opt.head_kernel != 3:
          print('Using head kernel:', opt.head_kernel)
          head_kernel = opt.head_kernel
        else:
          head_kernel = 3
        self.num_stacks = num_stacks

        if opt.trades:
            h = int(opt.output_h / 2)
            w = int(opt.output_w / 2)
            off_template_w = np.zeros((h, w, w), dtype=np.float32)
            off_template_h = np.zeros((h, w, h), dtype=np.float32)
            for ii in range(h):
                for jj in range(w):
                    for i in range(h):
                        off_template_h[ii, jj, i] = i - ii
                    for j in range(w):
                        off_template_w[ii, jj, j] = j - jj
            self.m = np.reshape(off_template_w, newshape=(h * w, w))[None, :, :] * 2
            self.v = np.reshape(off_template_h, newshape=(h * w, h))[None, :, :] * 2
            self.embed_dim = 128
            self.maxpool_stride2 = nn.MaxPool2d(2, stride=2)
            self.avgpool_stride4 = nn.AvgPool2d(4, stride=4)
            self.tempature = 5

            self.embedconv = nn.Sequential(
                nn.Conv2d(64, self.embed_dim, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(self.embed_dim, momentum=BN_MOMENTUM), nn.ReLU(inplace=True),
                nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(self.embed_dim, momentum=BN_MOMENTUM), nn.ReLU(inplace=True),
                nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1, stride=1, padding=0, bias=True))
            self._compute_chain_of_basic_blocks()
            self.attention_cur = nn.Conv2d(64, 1, kernel_size=(opt.deform_kernel_size, opt.deform_kernel_size), stride=1, dilation=(1, 1), padding=(1, 1), bias=True)
            self.attention_prev = nn.Conv2d(64, 1, kernel_size=(opt.deform_kernel_size, opt.deform_kernel_size), stride=1, dilation=(1, 1), padding=(1, 1), bias=True)
            self.conv_offset_w = nn.Conv2d(129, opt.deform_kernel_size * opt.deform_kernel_size, kernel_size=(opt.deform_kernel_size, opt.deform_kernel_size),
                                      stride=1, dilation=(1, 1), padding=(1, 1), bias=True)
            self.conv_offset_h = nn.Conv2d(129, opt.deform_kernel_size * opt.deform_kernel_size, kernel_size=(opt.deform_kernel_size, opt.deform_kernel_size),
                                      stride=1, dilation=(1, 1), padding=(1, 1), bias=True)
            self.dcn1_1 = DCN_TraDeS(64, 64, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=1)

        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            head_conv = head_convs[head]
            if len(head_conv) > 0:
                out = nn.Conv2d(head_conv[-1], classes,
                      kernel_size=1, stride=1, padding=0, bias=True)
                conv = nn.Conv2d(last_channel, head_conv[0],
                                 kernel_size=head_kernel,
                                 padding=head_kernel // 2, bias=True)
                convs = [conv]
                for k in range(1, len(head_conv)):
                    convs.append(nn.Conv2d(head_conv[k - 1], head_conv[k],
                                 kernel_size=1, bias=True))
                if len(convs) == 1:
                  fc = nn.Sequential(conv, nn.ReLU(inplace=True), out)
                elif len(convs) == 2:
                  fc = nn.Sequential(
                    convs[0], nn.ReLU(inplace=True),
                    convs[1], nn.ReLU(inplace=True), out)
                elif len(convs) == 3:
                  fc = nn.Sequential(
                      convs[0], nn.ReLU(inplace=True),
                      convs[1], nn.ReLU(inplace=True),
                      convs[2], nn.ReLU(inplace=True), out)
                elif len(convs) == 4:
                  fc = nn.Sequential(
                      convs[0], nn.ReLU(inplace=True),
                      convs[1], nn.ReLU(inplace=True),
                      convs[2], nn.ReLU(inplace=True),
                      convs[3], nn.ReLU(inplace=True), out)
                if head == "seg_feat":
                    fc = nn.Sequential(
                    nn.Conv2d(last_channel, head_conv[0],
                              kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(head_conv[0]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv[0], head_conv[0],
                              kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(head_conv[0]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv[0], classes,
                              kernel_size=1, padding=0, bias=False),
                    nn.BatchNorm2d(classes),
                    nn.ReLU(inplace=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(opt.prior_bias)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(last_channel, classes,
                    kernel_size=1, stride=1, padding=0, bias=True)
                if 'hm' in head:
                  fc.bias.data.fill_(opt.prior_bias)
                else:
                  fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def img2feats(self, x):
      raise NotImplementedError
    
    def imgpre2feats(self, x, pre_img=None, pre_hm=None):
      raise NotImplementedError

    def forward(self, x, pre_img=None, pre_hm=None, addtional_pre_imgs=None, addtional_pre_hms=None, inference_feats=None):
      cur_feat = self.img2feats(x)

      assert self.num_stacks == 1
      if self.opt.trades:
          feats, embedding, tracking_offset, dis_volume, h_volume_aux, w_volume_aux \
              = self.TraDeS(cur_feat, pre_img, pre_hm, addtional_pre_imgs, addtional_pre_hms, inference_feats)
      else:
          feats = [cur_feat[0]]

      out = []
      if self.opt.model_output_list:
        for s in range(self.num_stacks):
          z = []
          for head in sorted(self.heads):
              z.append(self.__getattr__(head)(feats[s]))
          out.append(z)
      else:
        for s in range(self.num_stacks):
          z = {}
          if self.opt.trades:
              z['embedding'] = embedding
              z['tracking_offset'] = tracking_offset
              if not self.opt.inference:
                  z['h_volume'] = dis_volume[0]
                  z['w_volume'] = dis_volume[1]
                  assert len(h_volume_aux) == self.opt.clip_len - 2
                  for temporal_id in range(2, self.opt.clip_len):
                      z['h_volume_prev{}'.format(temporal_id)] = h_volume_aux[temporal_id-2]
                      z['w_volume_prev{}'.format(temporal_id)] = w_volume_aux[temporal_id-2]
          for head in self.heads:
              z[head] = self.__getattr__(head)(feats[s])
          out.append(z)
      if self.opt.inference:
          return out, cur_feat[0].detach().cpu().numpy()
      else:
          return out

    def TraDeS(self, cur_feat, pre_img, pre_hm, addtional_pre_imgs, addtional_pre_hms, inference_feats):
        feat_list = []
        feat_list.append(cur_feat[0])  # current feature
        support_feats = []
        if self.opt.inference:
            for prev_feat in inference_feats:
                feat_list.append(torch.from_numpy(prev_feat).to(self.opt.device)[:, :, :, :])
            while len(feat_list) < self.opt.clip_len:  # only operate in the initial frame
                feat_list.append(cur_feat[0])

            for idx, feat_prev in enumerate(feat_list[1:]):
                pre_hm_i = addtional_pre_hms[idx]
                pre_hm_i = self.avgpool_stride4(pre_hm_i)
                support_feats.append(pre_hm_i * feat_prev)
        else:
            feat2 = self.img2feats_prev(pre_img)
            pre_hm_1 = self.avgpool_stride4(pre_hm)
            feat_list.append(feat2[0])
            support_feats.append(feat2[0] * pre_hm_1)
            for ff in range(len(addtional_pre_imgs) - 1):
                feats_ff = self.img2feats_prev(addtional_pre_imgs[ff])
                pre_hm_i = self.avgpool_stride4(addtional_pre_hms[ff])
                feat_list.append(feats_ff[0][:, :, :, :])
                support_feats.append(feats_ff[0][:, :, :, :]*pre_hm_i)

        return self.CVA_MFW(feat_list, support_feats)

    def CVA_MFW(self, feat_list, support_feats):
        prop_feats = []
        attentions = []
        h_max_for_loss_aux = []
        w_max_for_loss_aux = []
        feat_cur = feat_list[0]
        batch_size = feat_cur.shape[0]
        h_f = feat_cur.shape[2]
        w_f = feat_cur.shape[3]
        h_c = int(h_f / 2)
        w_c = int(w_f / 2)

        prop_feats.append(feat_cur)
        embedding = self.embedconv(feat_cur)
        embedding_prime = self.maxpool_stride2(embedding)
        # (B, 128, H, W) -> (B, H*W, 128):
        embedding_prime = embedding_prime.view(batch_size, self.embed_dim, -1).permute(0, 2, 1)
        attention_cur = self.attention_cur(feat_cur)
        attentions.append(attention_cur)
        for idx, feat_prev in enumerate(feat_list[1:]):
            # Sec. 4.1: Cost Volume based Association
            c_h, c_w, tracking_offset = self.CVA(embedding_prime, feat_prev, batch_size, h_c, w_c)

            # tracking offset output and CVA loss inputs
            if idx == 0:
                tracking_offset_output = tracking_offset
                h_max_for_loss = c_h
                w_max_for_loss = c_w
            else:
                h_max_for_loss_aux.append(c_h)
                w_max_for_loss_aux.append(c_w)

            # Sec. 4.2: Motion-guided Feature Warper
            prop_feat = self.MFW(support_feats[idx], tracking_offset, feat_cur, feat_prev, batch_size, h_f, w_f)
            prop_feats.append(prop_feat)
            attentions.append(self.attention_prev(prop_feat))

        attentions = torch.cat(attentions, dim=1)  # (B,T,H,W)
        adaptive_weights = F.softmax(attentions, dim=1)
        adaptive_weights = torch.split(adaptive_weights, 1, dim=1)  # 3*(B,1,H,W)
        # feature aggregation (MFW)
        enhanced_feat = 0
        for i in range(len(adaptive_weights)):
            enhanced_feat += adaptive_weights[i] * prop_feats[i]

        return [enhanced_feat], embedding, tracking_offset_output, [h_max_for_loss, w_max_for_loss], h_max_for_loss_aux, w_max_for_loss_aux

    def CVA(self, embedding_prime, feat_prev, batch_size, h_c, w_c):
        embedding_prev = self.embedconv(feat_prev)
        _embedding_prev = self.maxpool_stride2(embedding_prev)
        _embedding_prev = _embedding_prev.view(batch_size, self.embed_dim, -1)
        # Cost Volume Map
        c = torch.matmul(embedding_prime, _embedding_prev)  # (B, H*W/4, H*W/4)
        c = c.view(batch_size, h_c * w_c, h_c, w_c)  # (B, H*W, H, W)

        c_h = c.max(dim=3)[0]  # (B, H*W, H)
        c_w = c.max(dim=2)[0]  # (B, H*W, W)
        c_h_softmax = F.softmax(c_h * self.tempature, dim=2)
        c_w_softmax = F.softmax(c_w * self.tempature, dim=2)
        v = torch.tensor(self.v, device=self.opt.device)  # (1, H*W, H)
        m = torch.tensor(self.m, device=self.opt.device)
        off_h = torch.sum(c_h_softmax * v, dim=2, keepdim=True).permute(0, 2, 1)
        off_w = torch.sum(c_w_softmax * m, dim=2, keepdim=True).permute(0, 2, 1)
        off_h = off_h.view(batch_size, 1, h_c, w_c)
        off_w = off_w.view(batch_size, 1, h_c, w_c)
        off_h = nn.functional.interpolate(off_h, scale_factor=2)
        off_w = nn.functional.interpolate(off_w, scale_factor=2)

        tracking_offset = torch.cat((off_w, off_h), dim=1)

        return c_h, c_w, tracking_offset

    def MFW(self, support_feat, tracking_offset, feat_cur, feat_prev, batch_size, h_f, w_f):
        # deformable conv offset input
        off_deform = self.gamma(tracking_offset, feat_cur, feat_prev, batch_size, h_f, w_f)
        mask_deform = torch.tensor(np.ones((batch_size, 9, off_deform.shape[2], off_deform.shape[3]),
                                           dtype=np.float32)).to(self.opt.device)
        # feature propagation
        prop_feat = self.dcn1_1(support_feat, off_deform, mask_deform)

        return prop_feat

    def gamma(self, tracking_offset, feat_cur, feat_prev, batch_size, h_f, w_f):
        feat_diff = feat_cur - feat_prev
        feat_offs = self.offset_feats(feat_diff)
        feat_offs_h = torch.cat((tracking_offset[:, 1:2, :, :], feat_offs), dim=1)
        feat_offs_w = torch.cat((tracking_offset[:, 0:1, :, :], feat_offs), dim=1)

        off_h_deform = self.conv_offset_h(feat_offs_h)[:, :, None, :, :]  # (B, 9, H, W)
        off_w_deform = self.conv_offset_w(feat_offs_w)[:, :, None, :, :]
        off_deform = torch.cat((off_h_deform, off_w_deform), dim=2)  # (B, 9, 2, H, W)
        off_deform = off_deform.view(batch_size, 9 * 2, h_f, w_f)

        return off_deform

    def _compute_chain_of_basic_blocks(self):
        """
        "Learning Temporal Pose Estimation from Sparsely-Labeled Videos" (NeurIPS 2019)
        """
        num_blocks = 4
        block = BasicBlock
        in_ch = 128
        out_ch = 128
        stride = 1
        nc = 64
        ######
        downsample = nn.Sequential(
            nn.Conv2d(
                nc,
                in_ch,
                kernel_size=1, stride=stride, bias=False
            ),
            nn.BatchNorm2d(
                in_ch,
                momentum=BN_MOMENTUM
            ),
        )
        ##########3
        layers = []
        layers.append(
            block(
                nc,
                out_ch,
                stride,
                downsample
            )
        )
        for i in range(1, num_blocks):
            layers.append(
                block(
                    in_ch,
                    out_ch
                )
            )
        self.offset_feats = nn.Sequential(*layers)
        return
