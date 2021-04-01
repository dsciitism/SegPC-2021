import argparse

args = argparse.ArgumentParser()
args.add_argument('--backbone',type=str,required=True,choices=["Original","Effb5","Transformer_Effb5"],help="The backbone to be used from the given choices")
args.add_argument('--saved_model_path',type=str,required=True,help="path to the saved model which will be loaded")
args.add_argument('--input_images_folder',type=str,required=True,help="path to the folder where images to inference on are kept")
args.add_argument('--save_path',type=str,required=True,help="path to the folder where the generated masks will be saved")

args = args.parse_args()

import os
import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model,build_resnet_backbone,build_backbone
from detectron2.structures import ImageList
from detectron2.structures import Instances
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from torch import nn



import timm
from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer import *
import torch
import torch.nn as nn



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

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from segmentation_models_pytorch.encoders import get_encoder
import torch
import torch.nn as nn

@BACKBONE_REGISTRY.register()
class Effb5(Backbone):
  def __init__(self, cfg, input_shape):
    super().__init__()

    encoder_name = 'timm-efficientnet-b5'
    in_channels = 3
    encoder_depth = 5
    encoder_weights = 'noisy-student'
    self.encoder = get_encoder(encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights)
    self.channels = self.encoder.out_channels
    self.conv = nn.ModuleList(
        [nn.Conv2d(self.channels[i],256,3,stride = 2, padding = 1) for i in range(len(self.channels))]
    )

    self.names = ["p"+str(i+1) for i in range(6)]
  def forward(self, image):
    
    features = self.encoder(image)
    out = {self.names[i]: self.conv[i](features[i]) for i in range(1, len(features))}

    return out

  def output_shape(self):
    out_shape = {self.names[i]: ShapeSpec(channels =256, stride = 2**(i+1)) for i in range(1, len(self.names))}

    return out_shape
    
    
from torchvision.transforms import Resize
from einops import rearrange

@BACKBONE_REGISTRY.register()
class Transformer_Effb5(Backbone):
    def __init__(self, cfg, input_shape):

        super().__init__()
        self.emb_dim = 768
        self.pretrained = True
        self.pretrained_trans_model = 'vit_base_patch16_384'
        self.patch_size = 16
        
        self.transformer = Transformer_Encoder(pretrained = True, img_size = 384, pretrained_model = self.pretrained_trans_model, patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias = True)
        self.encoder_name = 'timm-efficientnet-b5'
        self.in_channels = 3
        self.encoder_depth = 5
        self.encoder_weights = 'noisy-student'
        
        self.conv_encoder = get_encoder(self.encoder_name,
                in_channels=self.in_channels,
                depth=self.encoder_depth,
                weights=self.encoder_weights)
        
        self.conv_channels = self.conv_encoder.out_channels
        self.conv_final = nn.ModuleList(
            [nn.Conv2d(self.conv_channels[i],self.emb_dim,3,stride = 2, padding = 1) for i in range(1,len(self.conv_channels))]
        )
        self.names = ["p"+str(i+2) for i in range(5)]
        self.resize =  Resize((384,384))
        self.Wq = nn.Linear(self.emb_dim, self.emb_dim, bias = False)
        self.Wk = nn.Linear(self.emb_dim, self.emb_dim, bias = False)
        
        


    def forward(self, image):
    
      conv_features = self.conv_encoder(image)
      # print("initial shape of conv features:")
      # print([i.shape for i in conv_features])
      conv_features = conv_features[1:]
      for i in range(len(self.conv_final)):
          conv_features[i]  = self.conv_final[i](conv_features[i])


      # print("final shape of conv features:")
      exp_shape = [i.shape for i in conv_features]
      # print(exp_shape)
      
      

      transformer_features = self.transformer(self.resize(image))
      # print("shape of transformer features:")
      # print([i.shape for i in transformer_features])
      
      # _ , l, e = transformer_features[0].shape
      # _ , e, h, w = conv_features[0].shape

      features = self.project(conv_features, transformer_features)
      features = self.emb2img(features, exp_shape)

      # print("shape of final features:")
      # print([i.shape for i in features])
      out = {self.names[i]: features[i] for i in range(len(features))}

      return out
    
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
    
    def output_shape(self):
      out_shape = {self.names[i]: ShapeSpec(channels =768, stride = 2**(i+2)) for i in range(len(self.names))}

      return out_shape
    
    
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"))

if 'Original' not in args.backbone:
  cfg.MODEL.BACKBONE.NAME = args.backbone

cfg.DATALOADER.NUM_WORKERS = 4

cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.GAMMA = 0.05

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1   


cfg.MODEL.WEIGHTS = args.saved_model_path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.001
predictor = DefaultPredictor(cfg)

img_root = args.input_images_folder
pred_root = args.save_path
names = os.listdir(img_root)
thresh = 0
var= 1

res_size=(1080,1440)

for name in names:
    print(var)
    var+=1
    print(name)
    index = name[:-4]
    
    im = cv2.imread(img_root+name)
    orig_shape = im.shape[0:2]
    im = cv2.resize(im, res_size[::-1],interpolation=cv2.INTER_NEAREST)

    outputs = predictor(im)
    scores = outputs['instances'].to('cpu').scores.numpy()
    pred_masks = outputs['instances'].to('cpu').pred_masks.numpy()
    count = 1
    for i in range(len(scores)):
        
        if scores[i]>=thresh:
           tmp_mask = pred_masks[i].astype('uint8')
          
           tmp_mask = 255*tmp_mask
           tmp_mask = cv2.resize(tmp_mask, orig_shape[::-1],interpolation=cv2.INTER_NEAREST)
           cv2.imwrite(pred_root+index+'_'+str(count)+'.bmp', tmp_mask)
           count+=1
    if count==1:
        tmp_mask = np.zeros(res_size)
        cv2.imwrite(pred_root+index+'_1.bmp', tmp_mask)
        print('blank mask saved')
    print(count-1,'masks saved')






