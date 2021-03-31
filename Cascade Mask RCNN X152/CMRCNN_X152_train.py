import argparse

args = argparse.ArgumentParser()
args.add_argument('--backbone',type=str,required=True,choices=["Original","Effb5","Transformer_Effb5"],help="The backbone to be used from the given choices")
args.add_argument('--train_data_root',type=str,required=True,help="path to training data root folder")
args.add_argument('--training_json_path',type=str,required=True,help="path to the training json file in COCO format")

args.add_argument('--val_data_root',type=str,required=True,help="path to validation data root folder")
args.add_argument('--validation_json_path',type=str,required=True,help="path to validation json file in COCO format")

args.add_argument('--work_dir',type=str,required=True,help="path to the folder where models and logs will be saved")
args.add_argument('--iterations',type=int,default=20000)
args.add_argument('--batch_size',type=int,default=16)

args = args.parse_args()

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



from detectron2.data.datasets import register_coco_instances
register_coco_instances("SegPC_train", {}, args.training_json_path, args.train_data_root)
register_coco_instances("SegPC_val", {}, args.validation_json_path, args.val_data_root)


train_meta = MetadataCatalog.get('SegPC_train')
val_meta = MetadataCatalog.get('SegPC_val')

train_dicts = DatasetCatalog.get("SegPC_train")
val_dicts = DatasetCatalog.get("SegPC_val")


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
cfg.DATASETS.TRAIN = ("SegPC_train",)
cfg.DATASETS.TEST = ("SegPC_val",)

cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml")  
cfg.SOLVER.IMS_PER_BATCH = args.batch_size
cfg.SOLVER.BASE_LR = 0.02/8
cfg.SOLVER.LR_SCHEDULER_NAME = 'WarmupCosineLR'

cfg.SOLVER.WARMUP_ITERS = 1500
cfg.SOLVER.MAX_ITER = args.iterations 
cfg.SOLVER.STEPS = (1000, 1500)
cfg.SOLVER.GAMMA = 0.05
cfg.SOLVER.CHECKPOINT_PERIOD = 1000

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

cfg.TEST.EVAL_PERIOD = 250

if 'Original' not in args.backbone:
  cfg.MODEL.BACKBONE.NAME = args.backbone
  
cfg.CUDNN_BENCHMARK = True
cfg.OUTPUT_DIR = args.work_dir



from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)
    
    
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()



  

  


