
import math
import torch
import faiss
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, SubsetRandomSampler

class TF_GAM(nn.Module):
    def __init__(self, temperature=1):
        super(TF_GAM, self).__init__()
        self.temperature = temperature

    def forward(self, feats, node=8, Lambda=0.8):
        feats = F.normalize(feats, p=2, dim=-1)
        scores = torch.matmul(feats, feats.transpose(-2, -1)) / self.temperature
        top_k_values, top_k_indices = torch.topk(scores, k=node, dim=-1)
        
        result = torch.full_like(scores, float("-inf"))
        result.scatter_(-1, top_k_indices, top_k_values)

        attn_weights = nn.Softmax(dim=-1)(result)
        feats_att = torch.matmul(attn_weights, feats)

        feats_gam = feats * Lambda + feats_att * (1 - Lambda)
        feats_norm = F.normalize(feats_gam, p=2, dim=-1)
        return feats_norm

class TF_CAM(nn.Module):
    def __init__(self, clusters_num=64, dim=768, normalize_input=True, work_with_tokens=False):
        super().__init__()
        self.clusters_num = clusters_num
        self.dim = dim
        self.alpha = 0
        self.normalize_input = normalize_input
        self.work_with_tokens = work_with_tokens
        if work_with_tokens:
            self.conv = nn.Conv1d(dim, clusters_num, kernel_size=1, bias=False)
        else:
            self.conv = nn.Conv2d(dim, clusters_num, kernel_size=(1, 1), bias=False)

    def init_params(self, centroids, descriptors):
        centroids_assign = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        dots = np.dot(centroids_assign, descriptors.T)
        dots.sort(0)
        dots = dots[::-1, :]

        self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
        if self.work_with_tokens:
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha * centroids_assign).unsqueeze(2))
        else:
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha * centroids_assign).unsqueeze(2).unsqueeze(3))
        self.conv.bias = None

    def forward(self, x):
        if self.work_with_tokens:
            x = x.permute(0, 2, 1)
            N, D, _ = x.shape[:]
        else:
            N, D, H, W = x.shape[:]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # Across descriptor dim
        x_flatten = x.view(N, D, -1)
        attention = self.conv(x).view(N, self.clusters_num, -1)
        attention = F.softmax(attention, dim=1)
        attn_out = torch.zeros([N, self.clusters_num, D], dtype=x_flatten.dtype, device=x_flatten.device)

        # Cross Attention
        for D in range(self.clusters_num):  # Slower than non-looped, but lower memory usage
            xx = x_flatten.unsqueeze(0).permute(1, 0, 2, 3)
            xx = xx * attention[:,D:D+1,:].unsqueeze(2)
            attn_out[:,D:D+1,:] = xx.sum(dim=-1)

        attn_out = F.normalize(attn_out, p=2, dim=2)  # intra-normalization
        attn_out = attn_out.view(N, -1)  # Flatten
        attn_out = F.normalize(attn_out, p=2, dim=1)  # L2 normalize
        return attn_out

    def Init_Cluster(self, args, cluster_ds, model):
        backbone = model.backbone
        descs_num_per_image = 100
        
        if len(cluster_ds) < 5000:
            if args.backbone.startswith("dinov2"): descs_num_per_image= 24*24
            if args.backbone.startswith("vitb16_224"): descs_num_per_image= 14*14
            if args.backbone.startswith("vitb16_384"): descs_num_per_image= 24*24
            if args.backbone.startswith("alexnet"): descs_num_per_image= 20*20
            if args.backbone.startswith("vgg16"): descs_num_per_image= 21*21
            if args.backbone.startswith("resnet101conv5"): descs_num_per_image= 11*11
            if args.backbone.startswith("resnet50conv5"): descs_num_per_image= 11*11
            if args.backbone.startswith("cct384"): descs_num_per_image= 24*24

        num_samples = len(cluster_ds)
        descriptors_num = num_samples * descs_num_per_image
        random_sampler = SubsetRandomSampler(np.random.choice(len(cluster_ds), num_samples, replace=False))
        random_dl = DataLoader(dataset=cluster_ds, num_workers=args.num_workers,
                                batch_size=args.infer_batch_size, sampler=random_sampler)
        with torch.no_grad():
            backbone = backbone.eval()
            logging.debug("Extracting features to initialize Cluster layer")
            descriptors = np.zeros(shape=(descriptors_num, args.features_dim), dtype=np.float32)
            for iteration, (inputs, _) in enumerate(tqdm(random_dl, ncols=100)):
                inputs = inputs.to(args.device)
                outputs = backbone(inputs)

                if args.backbone.startswith("dinov2"):
                    B,P,D = outputs["x_prenorm"].shape
                    W = H = int(math.sqrt(P-1))
                    outputs = outputs["x_norm_patchtokens"].view(B,W,H,D).permute(0, 3, 1, 2)

                elif args.backbone.startswith("cct"):
                    outputs = outputs.view(-1,24,24,384).permute(0, 3, 1, 2)
                
                elif args.backbone.startswith("vit"):                 
                    B,P,D = outputs.last_hidden_state.shape
                    W = H = int(math.sqrt(P-1))
                    outputs = outputs.last_hidden_state[:, 1:, :].view(B,W,H,D).permute(0, 3, 1, 2)                 
                else:
                    outputs = outputs

                norm_outputs = F.normalize(outputs, p=2, dim=1)
                image_descriptors = norm_outputs.view(norm_outputs.shape[0], args.features_dim, -1).permute(0, 2, 1)
                image_descriptors = image_descriptors.cpu().numpy()
                batchix = iteration * args.infer_batch_size * descs_num_per_image
                for ix in range(image_descriptors.shape[0]):
                    sample = np.random.choice(image_descriptors.shape[1], descs_num_per_image, replace=False)
                    startix = batchix + ix * descs_num_per_image
                    descriptors[startix:startix + descs_num_per_image, :] = image_descriptors[ix, sample, :]
        kmeans = faiss.Kmeans(args.features_dim, self.clusters_num, niter=100, verbose=False)
        kmeans.train(descriptors)
        logging.debug(f"All clusters shape: {kmeans.centroids.shape}")
        self.init_params(kmeans.centroids, descriptors)
        self = self.to(args.device)

