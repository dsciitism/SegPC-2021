import argparse

args = argparse.ArgumentParser()
args.add_argument('--backbone',type=str,required=True,choices=["Original","Effb5","Transformer_Effb5"],help="The backbone to be used from the given choices")
args.add_argument('--saved_model_path',type=str,required=True,help="path to the saved model which will be loaded")
args.add_argument('--input_images_folder',type=str,required=True,help="path to the folder where images to inference on are kept")
args.add_argument('--save_path',type=str,required=True,help="path to the folder where the generated masks will be saved")

args = args.parse_args()

import os

if not os.path.exists("mmdetection/"):
    raise Exception("inference script is not in the same directory where mmdetection_preparation.sh was run")
else :
    os.chdir("mmdetection/")

import timm
from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer import *
import torch
import torch.nn as nn

import torch
import torch.nn as nn
from timm.models import create_model
from functools import partial
from PIL import Image

import torch.nn as nn
from mmdet.models.builder import BACKBONES

from torchvision.transforms import Resize
from einops import rearrange
from segmentation_models_pytorch.encoders import get_encoder
import torch.nn.functional as F

from mmdet.models import DetectoRS_ResNet
import os 

class Transformer_Encoder(VisionTransformer):
    def __init__(self, pretrained = False, pretrained_model = None, img_size=224, patch_size=16, in_chans=3, num_classes=1, embed_dim=768, depth=12,
                  num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                  drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm):

        super(Transformer_Encoder, self).__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=1000, embed_dim=embed_dim, depth=depth,
                  num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                  drop_path_rate=drop_path_rate, hybrid_backbone=hybrid_backbone, norm_layer=norm_layer)
        
        self.num_classes = 1
        self.dispatcher = {
            'vit_small_patch16_224': vit_small_patch16_224,
            'vit_base_patch16_224': vit_base_patch16_224,
            'vit_large_patch16_224': vit_large_patch16_224,
            'vit_base_patch16_384': vit_base_patch16_384,
            'vit_base_patch32_384': vit_base_patch32_384,
            'vit_large_patch16_384': vit_large_patch16_384,
            'vit_large_patch32_384': vit_large_patch32_384,
            'vit_large_patch16_224' : vit_large_patch16_224,
            'vit_large_patch32_384': vit_large_patch32_384,
            'vit_small_resnet26d_224': vit_small_resnet26d_224,
            'vit_small_resnet50d_s3_224': vit_small_resnet50d_s3_224,
            'vit_base_resnet26d_224' : vit_base_resnet26d_224,
            'vit_base_resnet50d_224' : vit_base_resnet50d_224,
        }
        self.pretrained_model = pretrained_model
        self.pretrained = pretrained
        if pretrained:
            self.load_weights()
        self.head = nn.Identity()
        self.encoder_out = [1,2,3,4,5]

        

    def forward_features(self, x):

        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        features = []

        for i,blk in enumerate(self.blocks,1):
            x = blk(x)
            if i in self.encoder_out:
                features.append(x)

        for i in range(len(features)):
            features[i] = self.norm(features[i])

        return features

    def forward(self, x):

        features = self.forward_features(x)
        return features
    
    def load_weights(self):
        model = None
        try:
            model = self.dispatcher[self.pretrained_model](pretrained=True)
        except:
            print('could not not load model')
        if model == None:
            return
        # try:
        self.load_state_dict(model.state_dict())
        print("successfully loaded weights!!!")
        
        # except:
        #     print("Could not load weights. Parameters should match!!!")

@BACKBONES.register_module()
class Custom_Backbone(nn.Module):
    def __init__(self, **kwargs):

        super().__init__()
        self.num_classes = 1
        self.emb_dim = 768
        self.pretrained = True
        self.pretrained_trans_model = 'vit_base_patch16_384'
        self.patch_size = 16

        self.encoder_name = 'timm-efficientnet-b5'
        self.in_channels = 3
        self.encoder_depth = 5
        self.encoder_weights = 'noisy-student'
        
        self.conv_encoder = get_encoder(self.encoder_name,
                in_channels=self.in_channels,
                depth=self.encoder_depth,
                weights=self.encoder_weights)
        self.conv_encoder.num_classes = 1
        
        self.conv_channels = self.conv_encoder.out_channels
        
        if "Transformer" in args.backbone:
          self.transformer = Transformer_Encoder(pretrained = True, img_size = 384, pretrained_model = self.pretrained_trans_model, patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias = True)
          self.conv_final = nn.ModuleList(
              [nn.Conv2d(self.conv_channels[i],self.emb_dim,3,stride = 2, padding = 1) for i in range(1,len(self.conv_channels))]
          )
          self.names = ["p"+str(i+2) for i in range(5)]
          self.resize =  Resize((384,384))
          self.Wq = nn.Linear(self.emb_dim, self.emb_dim, bias = False)
          self.Wk = nn.Linear(self.emb_dim, self.emb_dim, bias = False)
          self.output_conv = nn.ModuleList([nn.Conv2d(self.emb_dim,2**(8+i),1,stride=1,padding = 0) for i in range(4)])
        
        else :
          self.output_conv = nn.ModuleList([nn.Conv2d(self.conv_channels[i+2],2**(8+i),1,stride=1,padding = 0) for i in range(4)])
        

    def forward(self, image):
    
      conv_features = list(self.conv_encoder(image))

      if "Transformer" in args.backbone:
        conv_features = conv_features[1:]
        for i in range(len(self.conv_final)):
            conv_features[i]  = self.conv_final[i](conv_features[i])        
        exp_shape = [i.shape for i in conv_features]
        transformer_features = self.transformer(F.interpolate(image,(384,384)))
        features = self.project(conv_features, transformer_features)
        features = self.emb2img(features, exp_shape)
        features = features[:-1]
        features = [self.output_conv[i](features[i]) for i in range(len(features))]
        features.insert(0,image)

      else :
        conv_features = conv_features[2:]
        features = conv_features
        features = [self.output_conv[i](features[i]) for i in range(len(features))]
        features.insert(0,image)

      return features
    
    def project(self, conv_features, transformer_features):

        features = []

        for i in range(len(conv_features)):

            t = transformer_features[i]
            x = rearrange(conv_features[i], 'b c h w -> b (h w) c') 
            xwq = self.Wq(x)
            twk = self.Wk(t)
            twk_T = rearrange(twk, 'b l c -> b c l')
            A = torch.einsum('bij,bjk->bik', xwq, twk_T).softmax(dim = -1)
            x += torch.einsum('bij,bjk->bik', A, t)
            features.append(x)

        return features
    
    def emb2img(self, features, exp_shape):

        for i, x in enumerate(features):
            B, P, E = x.shape             #(batch_size, latent_dim, emb_dim)
            x = x.transpose(1,2).reshape(B, E, exp_shape[i][2], exp_shape[i][3])
            features[i] = x

        return features
    
    def init_weights(self, pretrained=None):
      pass
    

from mmcv import Config
cfg = Config.fromfile('./configs/detectors/detectors_htc_r50_1x_coco.py')
    
from mmdet.apis import set_random_seed

cfg.dataset_type = 'CocoDataset'

cfg.model.roi_head.bbox_head[0].num_classes = 1
cfg.model.roi_head.bbox_head[1].num_classes = 1
cfg.model.roi_head.bbox_head[2].num_classes = 1
cfg.model.roi_head.mask_head[0].num_classes = 1
cfg.model.roi_head.mask_head[1].num_classes = 1
cfg.model.roi_head.mask_head[2].num_classes = 1
cfg.model.roi_head.semantic_head.num_classes = 1

cfg.data.train.type = 'CocoDataset'
cfg.data.train.classes = ['cell']
cfg.data.val.type = 'CocoDataset'
cfg.data.val.classes = ['cell']

if "Original" not in args.backbone:
    cfg.model.backbone['type'] = 'Custom_Backbone'
cfg.data['samples_per_gpu']=1
cfg.load_from = args.saved_model_path

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import copy
import os.path as osp

import mmcv
import numpy as np

model = init_detector(cfg,cfg.load_from, device='cuda:0')

import matplotlib.image as img
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import os
from tqdm import tqdm
import glob

test_root = args.input_images_folder
test_ids = os.listdir(test_root)
model.cfg = cfg

for it in tqdm(range(len(test_ids))):
  id = test_ids[it]
  image = mmcv.imread(test_root+"/"+id)
  
  result = inference_detector(model,image)
  count = 1
  
  print(len(result[1][0]))

  for i,mask in enumerate(result[1][0]):
    if mask.sum()<500:
      continue
    img.imsave(args.save_path+"/"+id[:-4]+"_{}".format(count)+".bmp",mask)
    count+=1
