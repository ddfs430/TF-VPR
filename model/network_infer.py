import os
import torch
import logging
import torchvision
from torch import nn
from os.path import join
from transformers import ViTModel, DeiTModel
#from google_drive_downloader import GoogleDriveDownloader as gdd

from model.cct import cct_14_7x2_384
from model.normalization import L2Norm
from model.TF_Cross_attn import TF_CAM

from torch.nn import Module, Linear, Dropout, LayerNorm, Identity
import torch.nn.functional as F
import math
from model.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

import faiss
from torch.nn.parameter import Parameter
from fast_pytorch_kmeans import KMeans
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm

# Pretrained models on Google Landmarks v2 and Places 365
PRETRAINED_MODELS = {
    'resnet18_places'  : '1DnEQXhmPxtBUrRc81nAvT8z17bk-GBj5',
    'resnet50_places'  : '1zsY4mN4jJ-AsmV3h4hjbT72CBfJsgSGC',
    'resnet101_places' : '1E1ibXQcg7qkmmmyYgmwMTh7Xf1cDNQXa',
    'vgg16_places'     : '1UWl1uz6rZ6Nqmp1K5z3GHAIZJmDh4bDu',
    'resnet18_gldv2'   : '1wkUeUXFXuPHuEvGTXVpuP5BMB-JJ1xke',
    'resnet50_gldv2'   : '1UDUv6mszlXNC1lv6McLdeBNMq9-kaA70',
    'resnet101_gldv2'  : '1apiRxMJpDlV0XmKlC5Na_Drg2jtGL-uE',
    'vgg16_gldv2'      : '10Ov9JdO7gbyz6mB5x0v_VSAUMj91Ta4o'
}

# Training-Free Self-Attention Module
class TF_SAM(nn.Module):
    def __init__(self, temperature=1):
        super(FCM_plus_StandardAttention, self).__init__()
        self.temperature = temperature

    def forward(self, feats, w=0.8):
        B, N, C = feats.shape
        feats = feats.reshape(B * N, C)
        feats = F.normalize(feats, p=2, dim=-1)
        
        scores = torch.matmul(feats, feats.transpose(-1, -2)) / self.temperature
        
        attn_weights = nn.Softmax(dim=-1)(scores)

        feats_att = torch.matmul(attn_weights, feats)

        feats_fcm = feats * w + feats_att * (1 - w)
        
        feats_fcm = feats_fcm.reshape(B, N ,C)
        feats_norm = F.normalize(feats_fcm, p=2, dim=-1)

        return feats_norm


# p=1  -->  Avg_pooling , p=999999  -->  Max_pooling 
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, work_with_tokens=False):
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
        self.work_with_tokens=work_with_tokens
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps, work_with_tokens=self.work_with_tokens)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

def gem(x, p=3, eps=1e-6, work_with_tokens=False):
    if work_with_tokens:
        x = x.permute(0, 2, 1)
        # unseqeeze to maintain compatibility with Flatten
        return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1))).pow(1./p).unsqueeze(3)
    else:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class Flatten(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): assert x.shape[2] == x.shape[3] == 1; return x[:,:,0,0]

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)

class Infer_Model(nn.Module):
    def __init__(self, args, pretrained_foundation=False, foundation_model_path=None):
        super().__init__()
        self.backbone = get_backbone(args, pretrained_foundation, foundation_model_path)
        self.arch_name = args.backbone
        if args.fc_output_dim != None:
            args.features_dim = args.fc_output_dim

        self.agg_GeM = nn.Sequential(L2Norm(), GeM(work_with_tokens=None), Flatten())
        self.agg_avg = nn.Sequential(L2Norm(), GeM(p=1, work_with_tokens=None), Flatten())

        self.agg_cam = TF_CAM(
                                clusters_num=args.clusters_num, 
                                dim=args.features_dim,
                                work_with_tokens=args.work_with_tokens)

        self.agg_sam = TF_SAM()

    def forward(self, x, do_GeM=False):
        x = self.backbone(x)
        if self.arch_name.startswith("vit"):
            B,P,D = x.last_hidden_state.shape
            W = H = int(math.sqrt(P-1))
            x1 = x.last_hidden_state[:, 1:, :].view(B,W,H,D).permute(0, 3, 1, 2) 
            x = x1
        elif self.arch_name.startswith("cct"):
            B,P,D = x.shape
            x = x.view(-1,24,24,384)
            x = x.permute(0, 3, 1, 2) 
        elif self.arch_name.startswith("dinov2"):
            B,P,D = x["x_prenorm"].shape
            W = H = int(math.sqrt(P-1))
            x0 = x["x_norm_clstoken"]
            x1 = x["x_norm_patchtokens"].view(B,W,H,D).permute(0, 3, 1, 2) 
            x2 = self.agg_GeM(x1)

            
            # # TF_SAM --> Training-Free Self-Attention Module
            x1 = self.agg_sam(x["x_norm_patchtokens"])
            x1 = x1.view(B,W,H,D).permute(0, 3, 1, 2)

            if do_GeM:
                # Experiment 1: TF_SAM + GeM
                # # GeM: Generalized average pooling
                x10,x11,x12,x13 = self.agg_GeM(x1[:,:,0:12,0:12]),self.agg_GeM(x1[:,:,0:12,12:]),self.agg_GeM(x1[:,:,12:,0:12]),self.agg_GeM(x1[:,:,12:,12:])
                x20,x21,x22,x23,x24,x25,x26,x27,x28 = self.agg_GeM(x1[:,:,0:8,0:8]),self.agg_GeM(x1[:,:,0:8,8:16]),self.agg_GeM(x1[:,:,0:8,16:]),\
                                                self.agg_GeM(x1[:,:,8:16,0:8]),self.agg_GeM(x1[:,:,8:16,8:16]),self.agg_GeM(x1[:,:,8:16,16:]),\
                                                self.agg_GeM(x1[:,:,16:,0:8]),self.agg_GeM(x1[:,:,16:,8:16]),self.agg_GeM(x1[:,:,16:,16:])
                x = [i.unsqueeze(1) for i in [x0,x10,x11,x12,x13,x20,x21,x22,x23,x24,x25,x26,x27,x28]]
                x = torch.cat(x,dim=1)
            else:
                # Experiment 2: TF_SAM + TF_CAM
                # TF_CAM --> Training-Free Cross-Attention Module
                x = self.agg_cam(x1)
        else:
            x = x

        x = x.view(B,-1,D)
        x = F.normalize(x, p=2, dim=2)  # intra-normalization
        x = x.view(B, -1)               # Flatten
        x = F.normalize(x, p=2, dim=1)  # L2 normalize
        return x
        
def get_pretrained_model(args):
    if args.pretrain == 'places':  num_classes = 365
    elif args.pretrain == 'gldv2':  num_classes = 512
    
    if args.backbone.startswith("resnet18"):
        model = torchvision.models.resnet18(num_classes=num_classes)
    elif args.backbone.startswith("resnet50"):
        model = torchvision.models.resnet50(num_classes=num_classes)
    elif args.backbone.startswith("resnet101"):
        model = torchvision.models.resnet101(num_classes=num_classes)
    elif args.backbone.startswith("vgg16"):
        model = torchvision.models.vgg16(num_classes=num_classes)
    
    if args.backbone.startswith('resnet'):
        model_name = args.backbone.split('conv')[0] + "_" + args.pretrain
    else:
        model_name = args.backbone + "_" + args.pretrain
    file_path = join("data", "pretrained_nets", model_name +".pth")
    
    if not os.path.exists(file_path):
        gdd.download_file_from_google_drive(file_id=PRETRAINED_MODELS[model_name], dest_path=file_path)
    state_dict = torch.load(file_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model

def get_backbone(args, pretrained_foundation, foundation_model_path):
    # The aggregation layer works differently based on the type of architecture
    args.work_with_tokens = False
    if args.backbone.startswith("resnet"):
        if args.pretrain in ['places', 'gldv2']:
            backbone = get_pretrained_model(args)
        elif args.backbone.startswith("resnet18"):
            backbone = torchvision.models.resnet18(pretrained=True)
        elif args.backbone.startswith("resnet50"):
            backbone = torchvision.models.resnet50(pretrained=True)
        elif args.backbone.startswith("resnet101"):
            backbone = torchvision.models.resnet101(pretrained=True)
        for name, child in backbone.named_children():
            # Freeze layers before conv_3
            if name == "layer3":
                break
            for params in child.parameters():
                params.requires_grad = False
        if args.backbone.endswith("conv4"):
            logging.debug(f"Train only conv4_x of the resnet{args.backbone.split('conv')[0]} (remove conv5_x), freeze the previous ones")
            layers = list(backbone.children())[:-3]
        elif args.backbone.endswith("conv5"):
            logging.debug(f"Train only conv4_x and conv5_x of the resnet{args.backbone.split('conv')[0]}, freeze the previous ones")
            layers = list(backbone.children())[:-2]
    elif args.backbone == "vgg16":
        if args.pretrain in ['places', 'gldv2']:
            backbone = get_pretrained_model(args)
        else:
            backbone = torchvision.models.vgg16(pretrained=True)
        layers = list(backbone.features.children())[:-2]
        for l in layers[:-5]:
            for p in l.parameters(): p.requires_grad = False
        logging.debug("Train last layers of the vgg16, freeze the previous ones")
    elif args.backbone == "alexnet":
        backbone = torchvision.models.alexnet(pretrained=True)
        layers = list(backbone.features.children())[:-2]
        for l in layers[:5]:
            for p in l.parameters(): p.requires_grad = False
        logging.debug("Train last layers of the alexnet, freeze the previous ones")
    elif args.backbone.startswith("cct"):
        if args.backbone.startswith("cct384"):
            backbone = cct_14_7x2_384(pretrained=True, progress=True, aggregation=args.aggregation)
        if args.trunc_te:
            logging.debug(f"Truncate CCT at transformers encoder {args.trunc_te}")
            backbone.classifier.blocks = torch.nn.ModuleList(backbone.classifier.blocks[:args.trunc_te].children())
        if args.freeze_te:
            logging.debug(f"Freeze all the layers up to tranformer encoder {args.freeze_te}")
            for p in backbone.parameters():
                p.requires_grad = False
            for name, child in backbone.classifier.blocks.named_children():
                if int(name) > args.freeze_te:
                    for params in child.parameters():
                        params.requires_grad = True
        args.features_dim = 384
        return backbone
    elif args.backbone.startswith("vit"):
        backbone = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        # if args.resize[0] == 224:
        #     backbone = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        # elif args.resize[0] == 384:
        #     backbone = ViTModel.from_pretrained('google/vit-base-patch16-384')
        # else:
        #     raise ValueError('Image size for ViT must be either 224 or 384')

        if args.trunc_te:
            logging.debug(f"Truncate ViT at transformers encoder {args.trunc_te}")
            backbone.encoder.layer = backbone.encoder.layer[:args.trunc_te]
        if args.freeze_te:
            logging.debug(f"Freeze all the layers up to tranformer encoder {args.freeze_te+1}")
            for p in backbone.parameters():
                p.requires_grad = False
            for name, child in backbone.encoder.layer.named_children():
                if int(name) > args.freeze_te:
                    for params in child.parameters():
                        params.requires_grad = True
        args.features_dim = 768
        return backbone

    # dinov2 vit B/14 768
    elif args.backbone.startswith("dinov2"):
        backbone = vit_base(patch_size=14,img_size=518,init_values=1,block_chunks=0)  
        if pretrained_foundation:
            assert foundation_model_path is not None, "Please specify foundation model path."
            model_dict = backbone.state_dict()
            state_dict = torch.load(foundation_model_path)
            model_dict.update(state_dict.items())
            backbone.load_state_dict(model_dict)

        if args.freeze_te:
            for p in backbone.parameters():
                p.requires_grad = False
            for name, child in backbone.blocks.named_children():
                if int(name) >= args.freeze_te:
                    for params in child.parameters():
                        params.requires_grad = True

        args.features_dim = 768 #1024
        return backbone
    
    backbone = torch.nn.Sequential(*layers)
    args.features_dim = get_output_channels_dim(backbone)  # Dinamically obtain number of channels in output
    return backbone

def get_output_channels_dim(model):
    """Return the number of channels in the output of a model."""
    return model(torch.ones([1, 3, 224, 224])).shape[1]

