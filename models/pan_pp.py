import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import time

from .backbone import build_backbone
from .neck import build_neck
from .head import build_head
from .utils import Conv_BN_ReLU


class PAN_PP(nn.Module):
    def __init__(self,
                 backbone,
                 neck,
                 detection_head,
                 recognition_head=None):
        super(PAN_PP, self).__init__()
        self.backbone = build_backbone(backbone)

        # 64, 128, 256, 512
        in_channels = neck.in_channels
        self.reduce_layer4 = Conv_BN_ReLU(in_channels[3], 128)
        self.reduce_layer3 = Conv_BN_ReLU(in_channels[2], 128)
        self.reduce_layer2 = Conv_BN_ReLU(in_channels[1], 128)
        self.reduce_layer1 = Conv_BN_ReLU(in_channels[0], 128)

        # self.cross_attention = Cross_Attention(512)

        self.fpem1 = build_neck(neck)
        self.fpem2 = build_neck(neck)

        self.det_head = build_head(detection_head)
        self.rec_head = None
        if recognition_head:
            self.rec_head = build_head(recognition_head)

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        # return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')
        return F.interpolate(x, size=(H // scale, W // scale), mode='bilinear')

    def forward(self,
                imgs,
                gt_texts=None,
                gt_kernels=None,
                training_masks=None,
                gt_instances=None,
                gt_bboxes=None,
                gt_words=None,
                word_masks=None,
                img_metas=None,
                cfg=None):
        outputs = dict()

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            start = time.time()

        # backbone
        f = self.backbone(imgs)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(
                backbone_time=time.time() - start
            ))
            start = time.time()

        # reduce channel
        f1 = self.reduce_layer1(f[0])
        f2 = self.reduce_layer2(f[1])
        f3 = self.reduce_layer3(f[2])
        f4 = self.reduce_layer4(f[3])

        # FPEM
        f1, f2, f3, f4 = self.fpem1(f1, f2, f3, f4)
        f1, f2, f3, f4 = self.fpem2(f1, f2, f3, f4)

        # FFM
        f2 = self._upsample(f2, f1.size())
        f3 = self._upsample(f3, f1.size())
        f4 = self._upsample(f4, f1.size())
        f = torch.cat((f1, f2, f3, f4), 1)

        # f = self.cross_attention(f)


        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(
                neck_time=time.time() - start
            ))
            start = time.time()

        # detection
        det_out = self.det_head(f)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(
                det_head_time=time.time() - start
            ))
            start = time.time()

        if self.training:
            det_out = self._upsample(det_out, imgs.size())
            loss_det = self.det_head.loss(det_out, gt_texts, gt_kernels, training_masks, gt_instances, gt_bboxes)
            outputs.update(loss_det)
        else:
            det_out = self._upsample(det_out, imgs.size(), cfg.test_cfg.scale)
            det_res = self.det_head.get_results(det_out, img_metas, cfg)
            outputs.update(det_res)

        if self.rec_head is not None:
            if self.training:
                if cfg.train_cfg.use_ex:
                    x_crops, gt_words = self.rec_head.extract_feature(
                        f, (imgs.size(2), imgs.size(3)),
                        gt_instances * gt_kernels[:, 0] * training_masks,
                        gt_bboxes, gt_words, word_masks)
                else:
                    x_crops, gt_words = self.rec_head.extract_feature(
                        f, (imgs.size(2), imgs.size(3)), gt_instances * training_masks, gt_bboxes, gt_words, word_masks)

                if x_crops is not None:
                    out_rec = self.rec_head(x_crops, gt_words)
                    loss_rec = self.rec_head.loss(out_rec, gt_words, reduce=False)
                else:
                    loss_rec = {
                        'loss_rec': f.new_full((1,), -1, dtype=torch.float32),
                        'acc_rec': f.new_full((1,), -1, dtype=torch.float32)
                    }
                outputs.update(loss_rec)
            else:
                if len(det_res['bboxes']) > 0:
                    x_crops, _ = self.rec_head.extract_feature(
                        f, (imgs.size(2), imgs.size(3)),
                        f.new_tensor(det_res['label'], dtype=torch.long).unsqueeze(0),
                        bboxes=f.new_tensor(det_res['bboxes_h'], dtype=torch.long),
                        unique_labels=det_res['instances'])
                    words, word_scores = self.rec_head.forward(x_crops)
                else:
                    words = []
                    word_scores = []

                if cfg.report_speed:
                    torch.cuda.synchronize()
                    outputs.update(dict(
                        rec_time=time.time() - start
                    ))
                outputs.update(dict(
                    words=words,
                    word_scores=word_scores,
                    label=''
                ))

        return outputs



class ConvBNLayer(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, act='relu', padding=0):
        super(ConvBNLayer, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.act = act
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.act != None:
            x = self.relu(x)
        return x


class Cross_Attention(nn.Module):
    def __init__(self, in_channels):
        super(Cross_Attention, self).__init__()
        self.theta_conv = ConvBNLayer(in_channels, in_channels, 1, 1, act='relu')
        self.phi_conv = ConvBNLayer(in_channels, in_channels, 1, 1, act='relu')
        self.g_conv = ConvBNLayer(in_channels, in_channels, 1, 1, act='relu')

        self.fh_weight_conv = ConvBNLayer(in_channels, in_channels, 1, 1, act=None)
        self.fh_sc_conv = ConvBNLayer(in_channels, in_channels, 1, 1, act=None)

        self.fv_weight_conv = ConvBNLayer(in_channels, in_channels, 1, 1, act=None)
        self.fv_sc_conv = ConvBNLayer(in_channels, in_channels, 1, 1, act=None)

        self.f_attn_conv = ConvBNLayer(in_channels * 2, in_channels, 1, 1, act='relu')

    def _cal_fweight(self, f, shape):
        f_theta, f_phi, f_g = f
        #flatten
        f_theta = f_theta.permute((0, 2, 3, 1))
        f_theta = torch.reshape(f_theta, [shape[0] * shape[1], shape[2], 512])
        f_phi = f_phi.permute((0, 2, 3, 1))
        f_phi = torch.reshape(f_phi, [shape[0] * shape[1], shape[2], 512])
        f_g = f_g.permute((0, 2, 3, 1))
        f_g = torch.reshape(f_g, [shape[0] * shape[1], shape[2], 512])
        #correlation
        f_attn = torch.matmul(f_theta, f_phi.permute((0, 2, 1)))
        #scale
        f_attn = f_attn / (128**0.5)
        f_attn = F.softmax(f_attn, dim=2)
        #weighted sum
        f_weight = torch.matmul(f_attn, f_g)
        f_weight = torch.reshape(f_weight, [shape[0], shape[1], shape[2], 512])
        return f_weight

    def forward(self, f_common):
        # f_shape = torch.shape(f_common)
        f_shape = f_common.shape
        # print('f_shape: ', f_shape)

        f_theta = self.theta_conv(f_common)
        f_phi = self.phi_conv(f_common)
        f_g = self.g_conv(f_common)

        ######## horizon ########
        fh_weight = self._cal_fweight([f_theta, f_phi, f_g],  [f_shape[0], f_shape[2], f_shape[3]])
        fh_weight = fh_weight.permute((0, 3, 1, 2))
        fh_weight = self.fh_weight_conv(fh_weight)
        #short cut
        fh_sc = self.fh_sc_conv(f_common)
        f_h = F.relu(fh_weight + fh_sc)

        ######## vertical ########
        fv_theta = f_theta.permute((0, 1, 3, 2))
        fv_phi = f_phi.permute((0, 1, 3, 2))
        fv_g = f_g.permute((0, 1, 3, 2))
        fv_weight = self._cal_fweight([fv_theta, fv_phi, fv_g], [f_shape[0], f_shape[3], f_shape[2]])
        fv_weight = fv_weight.permute((0, 3, 2, 1))
        fv_weight = self.fv_weight_conv(fv_weight)
        #short cut
        fv_sc = self.fv_sc_conv(f_common)
        f_v = F.relu(fv_weight + fv_sc)

        ######## merge ########
        f_attn = torch.cat([f_h, f_v], axis=1)
        f_attn = self.f_attn_conv(f_attn)
        return f_attn        

