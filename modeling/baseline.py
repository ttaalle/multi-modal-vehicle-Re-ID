# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import pdb
import torch
from torch import nn

from .backbones.resnet import ResNet, BasicBlock, Bottleneck 
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck 
from .backbones.squeezenet import SqueezeNet,Fire
from .backbones.densenet import _DenseLayer, _DenseBlock, _Transition, DenseNet
from .backbones.mobilenet import ConvBNReLU, InvertedResidual, MobileNetV2
from .backbones.inception import Inception3, BasicConv2d

import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.nn.functional as F

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(Baseline, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.base  = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
            
        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 6, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 23, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)  
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=64, 
                              reduction=16,
                              dropout_p=0.2, 
                              last_stride=last_stride)
        elif model_name == 'squeezenet':
             self.in_planes = 512
             self.base = SqueezeNet()
        elif model_name == 'densenet':
             self.in_planes = 1024
             self.base = DenseNet()
        elif model_name == 'mobilenet':
             self.in_planes = 1280
             self.base = MobileNetV2()
        elif model_name == 'inception':
             self.in_planes = 2048
             self.base = Inception3()

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')
            
        self.conv1 = nn.Conv2d(self.in_planes, num_classes, kernel_size=1, bias=False)
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        
        #y1 = x[:,0:1,:,:]*0.299+x[:,1:2,:,:]*0.587+x[:,2:3,:,:]*0.114
        #y = x
        #y[:,0:1,:,:] = y1
        #y[:,1:2,:,:] = y1
        #y[:,2:3,:,:] = y1
       # a=torch.ones(64,2048,8,16)
       # a[:,:,3:5,6:10]=0
      #  pdb.set_trace()
      #  a = torch.ones(1,1,8,16)
       # a[:,:,2:5,2:14]=0
      #  global_feat1 = self.gap(self.base(x)*a.cuda())  # (b, 2048, 1, 1)
        #pdb.set_trace()
        global_feat1 = self.gap(self.base(x))
        global_feat1 = global_feat1.view(global_feat1.shape[0], -1)
       # global_feat1 = global_feat1.view(global_feat1.shape[0], -1)  # flatten to (bs, 2048)
        #pdb.set_trace()
        #global_feat2 = self.gap(self.base(y))  # (b, 2048, 1, 1)
        #global_feat2 = global_feat2.view(global_feat2.shape[0], -1)  # flatten to (bs, 2048)  
        


        #global_feat =  torch.cat((global_feat1,global_feat2),1)    
        #pdb.set_trace()
        #global_feat = global_feat1 * global_feat2
        #global_feat = global_feat1
        global_feat = global_feat1
        #global_feat = self.gap(self.base(y))  # (b, 2048, 1, 1)
        #global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            #feat = self.bottleneck(global_feat)  # normalize for angular softmax
             feat = global_feat
            
        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

class Baseline1(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(Baseline1, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.base  = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
            
        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 6, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 23, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)  
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=64, 
                              reduction=16,
                              dropout_p=0.2, 
                              last_stride=last_stride)
        elif model_name == 'squeezenet':
             self.in_planes = 512
             self.base = SqueezeNet()
        elif model_name == 'densenet':
             self.in_planes = 1024
             self.base = DenseNet()
        elif model_name == 'mobilenet':
             self.in_planes = 1280
             self.base = MobileNetV2()
        elif model_name == 'inception':
             self.in_planes = 2048
             self.base = Inception3()

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.conv1 = nn.Conv2d(self.in_planes, num_classes, kernel_size=1, bias=False)
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        
        #y1 = x[:,0:1,:,:]*0.299+x[:,1:2,:,:]*0.587+x[:,2:3,:,:]*0.114
        #y = x
        #y[:,0:1,:,:] = y1
        #y[:,1:2,:,:] = y1
        #y[:,2:3,:,:] = y1
        f = self.conv1(self.base(x))
        global_feat1 = self.gap(self.base(x))  # (b, 2048, 1, 1)
        global_feat1 = global_feat1.view(global_feat1.shape[0], -1)  # flatten to (bs, 2048)
        
        #global_feat2 = self.gap(self.base(y))  # (b, 2048, 1, 1)
        #global_feat2 = global_feat2.view(global_feat2.shape[0], -1)  # flatten to (bs, 2048)  
        


        #global_feat =  torch.cat((global_feat1,global_feat2),1)    
        #pdb.set_trace()
        #global_feat = global_feat1 * global_feat2
        global_feat = global_feat1
        #global_feat = self.gap(self.base(y))  # (b, 2048, 1, 1)
        #global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax
            #feat = global_feat

        return feat,f

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
            
class Baseline2(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(Baseline2, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.base  = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
            
        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 6, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 23, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)  
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=64, 
                              reduction=16,
                              dropout_p=0.2, 
                              last_stride=last_stride)
        elif model_name == 'squeezenet':
             self.in_planes = 512
             self.base = SqueezeNet()
        elif model_name == 'densenet':
             self.in_planes = 1024
             self.base = DenseNet()
        elif model_name == 'mobilenet':
             self.in_planes = 1280
             self.base = MobileNetV2()
        elif model_name == 'inception':
             self.in_planes = 2048
             self.base = Inception3()

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.conv1 = nn.Conv2d(self.in_planes, num_classes, kernel_size=1, bias=False)
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        
        #y1 = x[:,0:1,:,:]*0.299+x[:,1:2,:,:]*0.587+x[:,2:3,:,:]*0.114
        #y = x
        #y[:,0:1,:,:] = y1
        #y[:,1:2,:,:] = y1
        #y[:,2:3,:,:] = y1
        f = self.conv1(self.base(x))
        global_feat1 = self.gap(self.base(x))  # (b, 2048, 1, 1)
        global_feat1 = global_feat1.view(global_feat1.shape[0], -1)  # flatten to (bs, 2048)
        
        #global_feat2 = self.gap(self.base(y))  # (b, 2048, 1, 1)
        #global_feat2 = global_feat2.view(global_feat2.shape[0], -1)  # flatten to (bs, 2048)  
        


        #global_feat =  torch.cat((global_feat1,global_feat2),1)    
        #pdb.set_trace()
        #global_feat = global_feat1 * global_feat2
        global_feat = global_feat1
        #global_feat = self.gap(self.base(y))  # (b, 2048, 1, 1)
        #global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax
            #feat = global_feat

        return feat, f

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

class Baseline3(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(Baseline3, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.base  = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
            
        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 6, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 23, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)  
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=64, 
                              reduction=16,
                              dropout_p=0.2, 
                              last_stride=last_stride)
        elif model_name == 'squeezenet':
             self.in_planes = 512
             self.base = SqueezeNet()
        elif model_name == 'densenet':
             self.in_planes = 1024
             self.base = DenseNet()
        elif model_name == 'mobilenet':
             self.in_planes = 1280
             self.base = MobileNetV2()
        elif model_name == 'inception':
             self.in_planes = 2048
             self.base = Inception3()
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')
            
        self.conv1 = nn.Conv2d(self.in_planes, num_classes, kernel_size=1, bias=False) 
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        
        #y1 = x[:,0:1,:,:]*0.299+x[:,1:2,:,:]*0.587+x[:,2:3,:,:]*0.114
        #y = x
        #y[:,0:1,:,:] = y1
        #y[:,1:2,:,:] = y1
        #y[:,2:3,:,:] = y1
        f = self.conv1(self.base(x))
        global_feat1 = self.gap(self.base(x))  # (b, 2048, 1, 1)
        global_feat1 = global_feat1.view(global_feat1.shape[0], -1)  # flatten to (bs, 2048)
        
        #global_feat2 = self.gap(self.base(y))  # (b, 2048, 1, 1)
        #global_feat2 = global_feat2.view(global_feat2.shape[0], -1)  # flatten to (bs, 2048)  
        


        #global_feat =  torch.cat((global_feat1,global_feat2),1)    
        #pdb.set_trace()
        #global_feat = global_feat1 * global_feat2
        global_feat = global_feat1
        #global_feat = self.gap(self.base(y))  # (b, 2048, 1, 1)
        #global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax
            #feat = global_feat
        return feat,f

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        #pdb.set_trace()
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
            
class Baseline4(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(Baseline4, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
        elif model_name == 'resnet50':
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
        elif model_name == 'resnet101':
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
        elif model_name == 'resnet152':
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            
        elif model_name == 'se_resnet50':
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
        elif model_name == 'se_resnet101':
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
        elif model_name == 'se_resnet152':
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
        elif model_name == 'se_resnext50':
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
        elif model_name == 'se_resnext101':
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
        elif model_name == 'senet154':
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
        elif model_name == 'squeezenet':
            self.in_planes = 512
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
        elif model_name == 'densenet':
            self.in_planes = 1024
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
        elif model_name == 'mobilenet':
            self.in_planes = 1280
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
        elif model_name == 'inception':
            self.in_planes = 2048
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)         
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        self.classifier = nn.Linear(self.in_planes, self.num_classes)
        #self.classifier = nn.BatchNorm1d(self.classifier)
        # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
        # self.classifier.apply(weights_init_classifier)  # new add by luo
 


    def forward(self, x1, x2, x3):
    

        vs,f1 = self.visible_net(x1)
        nr,f2 = self.nir_net(x2)
        tr,f3 = self.thermal_net(x3)
        f1 = torch.sigmoid(f1)
        f2 = torch.sigmoid(f2)
        f3 = torch.sigmoid(f3)
        #pdb.set_trace()
        
        ff =((f2>f3).float()*f2 + (f3>f2).float()*f3)
        ff =((ff>f1).float()*ff + (f1>ff).float()*f1)
        fs1 = f1.sum(dim=[2,3])
        fs11 = fs1.sum(dim=1)
        fs11= torch.unsqueeze(fs11,1)
        
        fs2 = f2.sum(dim=[2,3])
        fs22 = fs2.sum(dim=1)
        fs22= torch.unsqueeze(fs22,1)
        
        fs3 = f3.sum(dim=[2,3])
        fs33 = fs3.sum(dim=1)
        fs33= torch.unsqueeze(fs33,1)
        
        ffs = ff.sum(dim=[2,3])
        ffss = ffs.sum(dim=1)
        ffss= torch.unsqueeze(ffss,1)
        
        fs111 = (fs1 / ffs) / ((fs2 / ffs) + (fs3 / ffs) + (fs1 / ffs))
        fs222 = (fs2 / ffs) / ((fs2 / ffs) + (fs3 / ffs) + (fs1 / ffs))
        fs333 = (fs3 / ffs) / ((fs2 / ffs) + (fs3 / ffs) + (fs1 / ffs))
        fs = (ffs/ffss) * fs222 * fs2 + (ffs/ffss) * fs333 * fs3 +  (ffs/ffss) * fs111 * fs1

        
        

        #pdb.set_trace()
        x =  torch.unsqueeze(((ffs/ffss) * fs222).sum(1),1)*nr + torch.unsqueeze(((ffs/ffss) * fs333).sum(1),1)*tr + torch.unsqueeze(((ffs/ffss) * fs111).sum(1),1)*vs
      #  x= vs + nr + tr
       # x =  torch.unsqueeze(((ffs/ffss) * fs222).sum(1),1)*nr + torch.unsqueeze(((ffs/ffss) * fs333).sum(1),1)*tr
        #x = torch.cat((vs, nr, tr), 0)
        

        if self.training:
            cls_score1 = self.classifier(vs)
            cls_score2 = self.classifier(nr)
            cls_score3 = self.classifier(tr)  
            cls_score4 = self.classifier(x)     
            return  cls_score1, vs, cls_score2, nr, cls_score3, tr, cls_score4, x  # global feature for triplet loss
        else:
            return x
            
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

class Baseline5(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(Baseline5, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
        elif model_name == 'resnet50':
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
        elif model_name == 'resnet101':
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
        elif model_name == 'resnet152':
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            
        elif model_name == 'se_resnet50':
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
        elif model_name == 'se_resnet101':
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
        elif model_name == 'se_resnet152':
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
        elif model_name == 'se_resnext50':
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
        elif model_name == 'se_resnext101':
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
        elif model_name == 'senet154':
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        self.classifier = nn.Linear(self.in_planes, self.num_classes)
        #self.classifier = nn.BatchNorm1d(self.classifier)
        # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
        # self.classifier.apply(weights_init_classifier)  # new add by luo
 


    def forward(self, x1, x2, x3):
    

        vs,f1 = self.visible_net(x1)
        nr,f2 = self.nir_net(x2)
        tr,f3 = self.thermal_net(x3)
        #x= vs + nr +tr 
        x= vs + nr
        #x = torch.cat((vs, nr, tr), 0)
        

        if self.training:
            cls_score1 = self.classifier(vs)
            cls_score2 = self.classifier(nr)
            cls_score3 = self.classifier(tr)       
            return cls_score1, vs, cls_score2, nr, cls_score3, tr  # global feature for triplet loss
        else:
            return x
            
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

class Baseline6(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(Baseline6, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
        elif model_name == 'resnet50':
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
           # self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
        elif model_name == 'resnet101':
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
        elif model_name == 'resnet152':
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            
        elif model_name == 'se_resnet50':
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
        elif model_name == 'se_resnet101':
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
        elif model_name == 'se_resnet152':
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
        elif model_name == 'se_resnext50':
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
        elif model_name == 'se_resnext101':
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
        elif model_name == 'senet154':
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
        elif model_name == 'squeezenet':
            self.in_planes = 512
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
           # self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
        elif model_name == 'densenet':
            self.in_planes = 1024
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
           # self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
        elif model_name == 'mobilenet':
            self.in_planes = 1280
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            #self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
        elif model_name == 'inception':
            self.in_planes = 2048
            self.visible_net = Baseline1(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            self.nir_net = Baseline2(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)
            #self.thermal_net = Baseline3(num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice)         
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        self.classifier = nn.Linear(self.in_planes, self.num_classes)
        #self.classifier = nn.BatchNorm1d(self.classifier)
        # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
        # self.classifier.apply(weights_init_classifier)  # new add by luo
 


    def forward(self, x1, x2, x3):
    

        vs,f1 = self.visible_net(x1)
        nr,f2 = self.nir_net(x2)
        #tr,f3 = self.thermal_net(x3)
        f1 = torch.sigmoid(f1)
        f2 = torch.sigmoid(f2)
       # f3 = torch.sigmoid(f3)
        #pdb.set_trace()
        
        ff =((f2>f1).float()*f2 + (f1>f2).float()*f1)
        fs1 = f1.sum(dim=[2,3])
        fs11 = fs1.sum(dim=1)
        fs11= torch.unsqueeze(fs11,1)
        
        fs2 = f2.sum(dim=[2,3])
        fs22 = fs2.sum(dim=1)
        fs22= torch.unsqueeze(fs22,1)
        
        
        ffs = ff.sum(dim=[2,3])
        ffss = ffs.sum(dim=1)
        ffss= torch.unsqueeze(ffss,1)
        
        fs111 = (fs1 / ffs) / ((fs2 / ffs)  + (fs1 / ffs))
        fs222 = (fs2 / ffs) / ((fs2 / ffs)  + (fs1 / ffs))
        fs = (ffs/ffss) * fs222 * fs2  +  (ffs/ffss) * fs111 * fs1

        
        

        #pdb.set_trace()
        x =  torch.unsqueeze(((ffs/ffss) * fs222).sum(1),1)*nr  + torch.unsqueeze(((ffs/ffss) * fs111).sum(1),1)*vs
      #  x= vs + nr + tr
       # x =  torch.unsqueeze(((ffs/ffss) * fs222).sum(1),1)*nr + torch.unsqueeze(((ffs/ffss) * fs333).sum(1),1)*tr
        #x = torch.cat((vs, nr, tr), 0)
        

        if self.training:
            cls_score1 = self.classifier(vs)
            cls_score2 = self.classifier(nr)  
            cls_score4 = fs  
            return  cls_score1, vs, cls_score2, nr, cls_score4, fs  # global feature for triplet loss
        else:
            return x
            
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])