# coding: utf-8

"""
@File     :model.py
@Author    :XieJing
@Date      :2021/8/23
@Copyright :AI
@Desc      :
"""
import torch
from linformer import Linformer
from torch import nn
import numpy as np
import timm
import torch.nn.functional as F

all_timm_models = ['adv_inception_v3', 'bat_resnext26ts', 'botnet26t_256', 'botnet50ts_256', 'cait_m36_384',
                   'cait_m48_448', 'cait_s24_224', 'cait_s24_384', 'cait_s36_384', 'cait_xs24_384', 'cait_xxs24_224',
                   'cait_xxs24_384', 'cait_xxs36_224', 'cait_xxs36_384', 'coat_lite_mini', 'coat_lite_small',
                   'coat_lite_tiny', 'coat_mini', 'coat_tiny', 'convit_base', 'convit_small', 'convit_tiny',
                   'cspdarknet53', 'cspdarknet53_iabn', 'cspresnet50', 'cspresnet50d', 'cspresnet50w', 'cspresnext50',
                   'cspresnext50_iabn', 'darknet53', 'deit_base_distilled_patch16_224',
                   'deit_base_distilled_patch16_384', 'deit_base_patch16_224', 'deit_base_patch16_384',
                   'deit_small_distilled_patch16_224', 'deit_small_patch16_224', 'deit_tiny_distilled_patch16_224',
                   'deit_tiny_patch16_224', 'densenet121', 'densenet121d', 'densenet161', 'densenet169', 'densenet201',
                   'densenet264', 'densenet264d_iabn', 'densenetblur121d', 'dla34', 'dla46_c', 'dla46x_c', 'dla60',
                   'dla60_res2net', 'dla60_res2next', 'dla60x', 'dla60x_c', 'dla102', 'dla102x', 'dla102x2', 'dla169',
                   'dm_nfnet_f0', 'dm_nfnet_f1', 'dm_nfnet_f2', 'dm_nfnet_f3', 'dm_nfnet_f4', 'dm_nfnet_f5',
                   'dm_nfnet_f6', 'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn107', 'dpn131', 'eca_botnext26ts_256',
                   'eca_efficientnet_b0', 'eca_halonext26ts', 'eca_lambda_resnext26ts', 'eca_nfnet_l0', 'eca_nfnet_l1',
                   'eca_nfnet_l2', 'eca_nfnet_l3', 'eca_swinnext26ts_256', 'eca_vovnet39b', 'ecaresnet26t',
                   'ecaresnet50d', 'ecaresnet50d_pruned', 'ecaresnet50t', 'ecaresnet101d', 'ecaresnet101d_pruned',
                   'ecaresnet200d', 'ecaresnet269d', 'ecaresnetlight', 'ecaresnext26t_32x4d', 'ecaresnext50t_32x4d',
                   'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b1_pruned', 'efficientnet_b2',
                   'efficientnet_b2_pruned', 'efficientnet_b2a', 'efficientnet_b3', 'efficientnet_b3_pruned',
                   'efficientnet_b3a', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
                   'efficientnet_b8', 'efficientnet_cc_b0_4e', 'efficientnet_cc_b0_8e', 'efficientnet_cc_b1_8e',
                   'efficientnet_el', 'efficientnet_el_pruned', 'efficientnet_em', 'efficientnet_es',
                   'efficientnet_es_pruned', 'efficientnet_l2', 'efficientnet_lite0', 'efficientnet_lite1',
                   'efficientnet_lite2', 'efficientnet_lite3', 'efficientnet_lite4', 'efficientnetv2_l',
                   'efficientnetv2_m', 'efficientnetv2_rw_m', 'efficientnetv2_rw_s', 'efficientnetv2_s',
                   'ens_adv_inception_resnet_v2', 'ese_vovnet19b_dw', 'ese_vovnet19b_slim', 'ese_vovnet19b_slim_dw',
                   'ese_vovnet39b', 'ese_vovnet39b_evos', 'ese_vovnet57b', 'ese_vovnet99b', 'ese_vovnet99b_iabn',
                   'fbnetc_100', 'fbnetv3_b', 'fbnetv3_d', 'fbnetv3_g', 'gc_efficientnet_b0', 'gcresnet50t',
                   'gcresnext26ts', 'geresnet50t', 'gernet_l', 'gernet_m', 'gernet_s', 'ghostnet_050', 'ghostnet_100',
                   'ghostnet_130', 'gluon_inception_v3', 'gluon_resnet18_v1b', 'gluon_resnet34_v1b',
                   'gluon_resnet50_v1b', 'gluon_resnet50_v1c', 'gluon_resnet50_v1d', 'gluon_resnet50_v1s',
                   'gluon_resnet101_v1b', 'gluon_resnet101_v1c', 'gluon_resnet101_v1d', 'gluon_resnet101_v1s',
                   'gluon_resnet152_v1b', 'gluon_resnet152_v1c', 'gluon_resnet152_v1d', 'gluon_resnet152_v1s',
                   'gluon_resnext50_32x4d', 'gluon_resnext101_32x4d', 'gluon_resnext101_64x4d', 'gluon_senet154',
                   'gluon_seresnext50_32x4d', 'gluon_seresnext101_32x4d', 'gluon_seresnext101_64x4d',
                   'gluon_xception65', 'gmixer_12_224', 'gmixer_24_224', 'gmlp_b16_224', 'gmlp_s16_224',
                   'gmlp_ti16_224', 'halonet26t', 'halonet50ts', 'halonet_h1', 'halonet_h1_c4c5', 'hardcorenas_a',
                   'hardcorenas_b', 'hardcorenas_c', 'hardcorenas_d', 'hardcorenas_e', 'hardcorenas_f', 'hrnet_w18',
                   'hrnet_w18_small', 'hrnet_w18_small_v2', 'hrnet_w30', 'hrnet_w32', 'hrnet_w40', 'hrnet_w44',
                   'hrnet_w48', 'hrnet_w64', 'ig_resnext101_32x8d', 'ig_resnext101_32x16d', 'ig_resnext101_32x32d',
                   'ig_resnext101_32x48d', 'inception_resnet_v2', 'inception_v3', 'inception_v4', 'lambda_resnet26t',
                   'lambda_resnet50t', 'legacy_senet154', 'legacy_seresnet18', 'legacy_seresnet34', 'legacy_seresnet50',
                   'legacy_seresnet101', 'legacy_seresnet152', 'legacy_seresnext26_32x4d', 'legacy_seresnext50_32x4d',
                   'legacy_seresnext101_32x4d', 'levit_128', 'levit_128s', 'levit_192', 'levit_256', 'levit_384',
                   'mixer_b16_224', 'mixer_b16_224_in21k', 'mixer_b16_224_miil', 'mixer_b16_224_miil_in21k',
                   'mixer_b32_224', 'mixer_l16_224', 'mixer_l16_224_in21k', 'mixer_l32_224', 'mixer_s16_224',
                   'mixer_s32_224', 'mixnet_l', 'mixnet_m', 'mixnet_s', 'mixnet_xl', 'mixnet_xxl', 'mnasnet_050',
                   'mnasnet_075', 'mnasnet_100', 'mnasnet_140', 'mnasnet_a1', 'mnasnet_b1', 'mnasnet_small',
                   'mobilenetv2_100', 'mobilenetv2_110d', 'mobilenetv2_120d', 'mobilenetv2_140',
                   'mobilenetv3_large_075', 'mobilenetv3_large_100', 'mobilenetv3_large_100_miil',
                   'mobilenetv3_large_100_miil_in21k', 'mobilenetv3_rw', 'mobilenetv3_small_075',
                   'mobilenetv3_small_100', 'nasnetalarge', 'nf_ecaresnet26', 'nf_ecaresnet50', 'nf_ecaresnet101',
                   'nf_regnet_b0', 'nf_regnet_b1', 'nf_regnet_b2', 'nf_regnet_b3', 'nf_regnet_b4', 'nf_regnet_b5',
                   'nf_resnet26', 'nf_resnet50', 'nf_resnet101', 'nf_seresnet26', 'nf_seresnet50', 'nf_seresnet101',
                   'nfnet_f0', 'nfnet_f0s', 'nfnet_f1', 'nfnet_f1s', 'nfnet_f2', 'nfnet_f2s', 'nfnet_f3', 'nfnet_f3s',
                   'nfnet_f4', 'nfnet_f4s', 'nfnet_f5', 'nfnet_f5s', 'nfnet_f6', 'nfnet_f6s', 'nfnet_f7', 'nfnet_f7s',
                   'nfnet_l0', 'pit_b_224', 'pit_b_distilled_224', 'pit_s_224', 'pit_s_distilled_224', 'pit_ti_224',
                   'pit_ti_distilled_224', 'pit_xs_224', 'pit_xs_distilled_224', 'pnasnet5large', 'rednet26t',
                   'rednet50ts', 'regnetx_002', 'regnetx_004', 'regnetx_006', 'regnetx_008', 'regnetx_016',
                   'regnetx_032', 'regnetx_040', 'regnetx_064', 'regnetx_080', 'regnetx_120', 'regnetx_160',
                   'regnetx_320', 'regnety_002', 'regnety_004', 'regnety_006', 'regnety_008', 'regnety_016',
                   'regnety_032', 'regnety_040', 'regnety_064', 'regnety_080', 'regnety_120', 'regnety_160',
                   'regnety_320', 'repvgg_a2', 'repvgg_b0', 'repvgg_b1', 'repvgg_b1g4', 'repvgg_b2', 'repvgg_b2g4',
                   'repvgg_b3', 'repvgg_b3g4', 'res2net50_14w_8s', 'res2net50_26w_4s', 'res2net50_26w_6s',
                   'res2net50_26w_8s', 'res2net50_48w_2s', 'res2net101_26w_4s', 'res2next50', 'resmlp_12_224',
                   'resmlp_12_distilled_224', 'resmlp_24_224', 'resmlp_24_distilled_224', 'resmlp_36_224',
                   'resmlp_36_distilled_224', 'resmlp_big_24_224', 'resmlp_big_24_224_in22ft1k',
                   'resmlp_big_24_distilled_224', 'resnest14d', 'resnest26d', 'resnest50d', 'resnest50d_1s4x24d',
                   'resnest50d_4s2x40d', 'resnest101e', 'resnest200e', 'resnest269e', 'resnet18', 'resnet18d',
                   'resnet26', 'resnet26d', 'resnet26t', 'resnet34', 'resnet34d', 'resnet50', 'resnet50d', 'resnet50t',
                   'resnet51q', 'resnet61q', 'resnet101', 'resnet101d', 'resnet152', 'resnet152d', 'resnet200',
                   'resnet200d', 'resnetblur18', 'resnetblur50', 'resnetrs50', 'resnetrs101', 'resnetrs152',
                   'resnetrs200', 'resnetrs270', 'resnetrs350', 'resnetrs420', 'resnetv2_50', 'resnetv2_50d',
                   'resnetv2_50t', 'resnetv2_50x1_bit_distilled', 'resnetv2_50x1_bitm', 'resnetv2_50x1_bitm_in21k',
                   'resnetv2_50x3_bitm', 'resnetv2_50x3_bitm_in21k', 'resnetv2_101', 'resnetv2_101d',
                   'resnetv2_101x1_bitm', 'resnetv2_101x1_bitm_in21k', 'resnetv2_101x3_bitm',
                   'resnetv2_101x3_bitm_in21k', 'resnetv2_152', 'resnetv2_152d', 'resnetv2_152x2_bit_teacher',
                   'resnetv2_152x2_bit_teacher_384', 'resnetv2_152x2_bitm', 'resnetv2_152x2_bitm_in21k',
                   'resnetv2_152x4_bitm', 'resnetv2_152x4_bitm_in21k', 'resnext50_32x4d', 'resnext50d_32x4d',
                   'resnext101_32x4d', 'resnext101_32x8d', 'resnext101_64x4d', 'rexnet_100', 'rexnet_130', 'rexnet_150',
                   'rexnet_200', 'rexnetr_100', 'rexnetr_130', 'rexnetr_150', 'rexnetr_200', 'selecsls42',
                   'selecsls42b', 'selecsls60', 'selecsls60b', 'selecsls84', 'semnasnet_050', 'semnasnet_075',
                   'semnasnet_100', 'semnasnet_140', 'senet154', 'seresnet18', 'seresnet34', 'seresnet50',
                   'seresnet50t', 'seresnet101', 'seresnet152', 'seresnet152d', 'seresnet200d', 'seresnet269d',
                   'seresnext26d_32x4d', 'seresnext26t_32x4d', 'seresnext26tn_32x4d', 'seresnext50_32x4d',
                   'seresnext101_32x4d', 'seresnext101_32x8d', 'skresnet18', 'skresnet34', 'skresnet50', 'skresnet50d',
                   'skresnext50_32x4d', 'spnasnet_100', 'ssl_resnet18', 'ssl_resnet50', 'ssl_resnext50_32x4d',
                   'ssl_resnext101_32x4d', 'ssl_resnext101_32x8d', 'ssl_resnext101_32x16d',
                   'swin_base_patch4_window7_224', 'swin_base_patch4_window7_224_in22k',
                   'swin_base_patch4_window12_384', 'swin_base_patch4_window12_384_in22k',
                   'swin_large_patch4_window7_224', 'swin_large_patch4_window7_224_in22k',
                   'swin_large_patch4_window12_384', 'swin_large_patch4_window12_384_in22k',
                   'swin_small_patch4_window7_224', 'swin_tiny_patch4_window7_224', 'swinnet26t_256', 'swinnet50ts_256',
                   'swsl_resnet18', 'swsl_resnet50', 'swsl_resnext50_32x4d', 'swsl_resnext101_32x4d',
                   'swsl_resnext101_32x8d', 'swsl_resnext101_32x16d', 'tf_efficientnet_b0', 'tf_efficientnet_b0_ap',
                   'tf_efficientnet_b0_ns', 'tf_efficientnet_b1', 'tf_efficientnet_b1_ap', 'tf_efficientnet_b1_ns',
                   'tf_efficientnet_b2', 'tf_efficientnet_b2_ap', 'tf_efficientnet_b2_ns', 'tf_efficientnet_b3',
                   'tf_efficientnet_b3_ap', 'tf_efficientnet_b3_ns', 'tf_efficientnet_b4', 'tf_efficientnet_b4_ap',
                   'tf_efficientnet_b4_ns', 'tf_efficientnet_b5', 'tf_efficientnet_b5_ap', 'tf_efficientnet_b5_ns',
                   'tf_efficientnet_b6', 'tf_efficientnet_b6_ap', 'tf_efficientnet_b6_ns', 'tf_efficientnet_b7',
                   'tf_efficientnet_b7_ap', 'tf_efficientnet_b7_ns', 'tf_efficientnet_b8', 'tf_efficientnet_b8_ap',
                   'tf_efficientnet_cc_b0_4e', 'tf_efficientnet_cc_b0_8e', 'tf_efficientnet_cc_b1_8e',
                   'tf_efficientnet_el', 'tf_efficientnet_em', 'tf_efficientnet_es', 'tf_efficientnet_l2_ns',
                   'tf_efficientnet_l2_ns_475', 'tf_efficientnet_lite0', 'tf_efficientnet_lite1',
                   'tf_efficientnet_lite2', 'tf_efficientnet_lite3', 'tf_efficientnet_lite4', 'tf_efficientnetv2_b0',
                   'tf_efficientnetv2_b1', 'tf_efficientnetv2_b2', 'tf_efficientnetv2_b3', 'tf_efficientnetv2_l',
                   'tf_efficientnetv2_l_in21ft1k', 'tf_efficientnetv2_l_in21k', 'tf_efficientnetv2_m',
                   'tf_efficientnetv2_m_in21ft1k', 'tf_efficientnetv2_m_in21k', 'tf_efficientnetv2_s',
                   'tf_efficientnetv2_s_in21ft1k', 'tf_efficientnetv2_s_in21k', 'tf_inception_v3', 'tf_mixnet_l',
                   'tf_mixnet_m', 'tf_mixnet_s', 'tf_mobilenetv3_large_075', 'tf_mobilenetv3_large_100',
                   'tf_mobilenetv3_large_minimal_100', 'tf_mobilenetv3_small_075', 'tf_mobilenetv3_small_100',
                   'tf_mobilenetv3_small_minimal_100', 'tnt_b_patch16_224', 'tnt_s_patch16_224', 'tresnet_l',
                   'tresnet_l_448', 'tresnet_m', 'tresnet_m_448', 'tresnet_m_miil_in21k', 'tresnet_xl',
                   'tresnet_xl_448', 'tv_densenet121', 'tv_resnet34', 'tv_resnet50', 'tv_resnet101', 'tv_resnet152',
                   'tv_resnext50_32x4d', 'twins_pcpvt_base', 'twins_pcpvt_large', 'twins_pcpvt_small', 'twins_svt_base',
                   'twins_svt_large', 'twins_svt_small', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
                   'vgg19', 'vgg19_bn', 'visformer_small', 'visformer_tiny', 'vit_base_patch16_224',
                   'vit_base_patch16_224_in21k', 'vit_base_patch16_224_miil', 'vit_base_patch16_224_miil_in21k',
                   'vit_base_patch16_384', 'vit_base_patch32_224', 'vit_base_patch32_224_in21k', 'vit_base_patch32_384',
                   'vit_base_r26_s32_224', 'vit_base_r50_s16_224', 'vit_base_r50_s16_224_in21k', 'vit_base_r50_s16_384',
                   'vit_base_resnet26d_224', 'vit_base_resnet50_224_in21k', 'vit_base_resnet50_384',
                   'vit_base_resnet50d_224', 'vit_huge_patch14_224_in21k', 'vit_large_patch16_224',
                   'vit_large_patch16_224_in21k', 'vit_large_patch16_384', 'vit_large_patch32_224',
                   'vit_large_patch32_224_in21k', 'vit_large_patch32_384', 'vit_large_r50_s32_224',
                   'vit_large_r50_s32_224_in21k', 'vit_large_r50_s32_384', 'vit_small_patch16_224',
                   'vit_small_patch16_224_in21k', 'vit_small_patch16_384', 'vit_small_patch32_224',
                   'vit_small_patch32_224_in21k', 'vit_small_patch32_384', 'vit_small_r26_s32_224',
                   'vit_small_r26_s32_224_in21k', 'vit_small_r26_s32_384', 'vit_small_resnet26d_224',
                   'vit_small_resnet50d_s16_224', 'vit_tiny_patch16_224', 'vit_tiny_patch16_224_in21k',
                   'vit_tiny_patch16_384', 'vit_tiny_r_s16_p8_224', 'vit_tiny_r_s16_p8_224_in21k',
                   'vit_tiny_r_s16_p8_384', 'vovnet39a', 'vovnet57a', 'wide_resnet50_2', 'wide_resnet101_2', 'xception',
                   'xception41', 'xception65', 'xception71']


def build_model(config):
    model_name = config['model_name'].lower()
    pretrained = config['pretrained']
    num_classes = config['num_classes']
    # if model_name in all_timm_models:
    if pretrained:
        model = timm.create_model(model_name=model_name, pretrained=pretrained, num_classes=num_classes)
        for param in model.parameters():
            param.require_grad = False
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        return model, model.fc.parameters()
        # return model, model.parameters()
    else:
        model = timm.create_model(model_name=model_name, pretrained=pretrained, num_classes=num_classes)
        model.apply(weights_init)
        return model, model.parameters()


class VGG16(nn.Module):
    def __init__(self, num_classes=4, init_weights=True):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 224*224*3     224*224*64
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 224*224*64
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes)
        )
        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


class AlexNet(nn.Module):
    def __init__(self, num_classes=4, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 56, 56]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[48, 28, 28]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),  # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 14, 14]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 7, 7]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def weights_init(m):
    # 1. 根据网络层的不同定义不同的初始化方式
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=4, aux_logits=False, init_weights=True):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 840 x 840
        x = self.conv1(x)
        # N x 64 x 420 x 420
        x = self.maxpool1(x)
        # N x 64 x 210 x 210
        x = self.conv2(x)
        # N x 64 x 210 x 210
        x = self.conv3(x)
        # N x 192 x 210 x 210
        x = self.maxpool2(x)

        # N x 192 x 105 x 105
        x = self.inception3a(x)
        # N x 256 x 105 x 105
        x = self.inception3b(x)
        # N x 480 x 105 x 105
        x = self.maxpool3(x)
        # N x 480 x 52 x 52
        x = self.inception4a(x)
        # N x 512 x 52 x 52
        if self.training and self.aux_logits:  # eval model lose this layer
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        if self.training and self.aux_logits:  # eval model lose this layer
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 26 x 26
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 26 x 26

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        if self.training and self.aux_logits:  # eval model lose this layer
            return x, aux2, aux1
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# inception结构
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)  # 保证输出大小等于输入大小
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)  # 保证输出大小等于输入大小
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


# 辅助分类器
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # output[batch, 128, 4, 4]

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = self.averagePool(x)
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x num_classes
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        self.sigmoid = torch.nn.Sigmoid()

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc1 = nn.Linear(512 * block.expansion, 1)
            self.fc2 = nn.Linear(512 * block.expansion, 1)
            self.fc3 = nn.Linear(512 * block.expansion, 1)
            self.fc4 = nn.Linear(512 * block.expansion, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        task1, task2, task3, task4 = x, x, x, x
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            task1 = self.sigmoid(self.fc1(x))
            task2 = self.sigmoid(self.fc2(x))
            task3 = self.sigmoid(self.fc3(x))
            task4 = self.sigmoid(self.fc4(x))

        return task1

