import os
import torch
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmarking Visual Geolocalization", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #------------------------------------------      core set          --------------------------------------------------
    parser.add_argument('--num_clusters', type=int, default=16, help="Number of clusters for clustering layer.")
    parser.add_argument("--foundation_model_path", type=str, 
                        default="/data/users/model_weight/DINO_V2/dinov2_vitb14_pretrain.pth",
                        help="Path to load foundation model checkpoint.")
    parser.add_argument("--backbone", type=str, default="dinov2",
                        choices=["alexnet", "vgg16", "resnet18conv4", "resnet18conv5", 
                                "resnet50conv4", "resnet50conv5", "resnet101conv4", "resnet101conv5",
                                "cct384", "vitb16_224", "vitb16_384", "dinov2", "swin"])
    parser.add_argument('--resize', type=int, default=[336, 336], nargs=2, help="Resizing shape for images (HxW).")
    parser.add_argument("--eval_dataset_name", type=str, 
                        default="msls", 
                        choices=["msls", "nordland", "pitts30k", "pitts250k", "sped",
                                 "san_francisco", "st_lucia", "tokyo247","amstertime", "eynsham",
                                 "svox", "svox_night", "svox_overcast", "svox_rain", "svox_snow", "svox_sun"])
    parser.add_argument("--eval_datasets_folder", type=str, default='/data/datasets/vpr', help="Path with all datasets")
    parser.add_argument('--test_method', type=str, default="hard_resize",
                        choices=["hard_resize", "single_query", "central_crop", "five_crops", "nearest_crop", "maj_voting"],
                        help="This includes pre/post-processing methods and prediction refinement")
    parser.add_argument("--infer_batch_size", type=int, default=50, help="Batch size for inference (caching and testing)")
    parser.add_argument("--mode", type=int, default="TF_VPR",
                        choices=["TF_VPR", "GeM_CLS", "CLS", "Mean", "GeM_Mean", "CLS_Mean"],
                        help="Mode choice for inference (caching and testing)")
    parser.add_argument("--seed", type=int, default=1)
    #---------------------------------------------------------------------------------------------------------------------
    # Other parameters
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=16, help="num_workers for all dataloaders")
    parser.add_argument("--majority_weight", type=float, default=0.01, 
                        help="only for majority voting, scale factor, the higher it is the more importance is given to agreement")
    parser.add_argument("--efficient_ram_testing", action='store_true', help="_")
    parser.add_argument("--val_positive_dist_threshold", type=int, default=25, help="_")
    parser.add_argument("--train_positives_dist_threshold", type=int, default=10, help="_")
    parser.add_argument('--recall_values', type=int, default=[1, 5, 10], nargs="+",
                        help="Recalls to be computed, such as R@5.")
    # Data augmentation parameters
    parser.add_argument("--brightness", type=float, default=None, help="_")
    parser.add_argument("--contrast", type=float, default=None, help="_")
    parser.add_argument("--saturation", type=float, default=None, help="_")
    parser.add_argument("--hue", type=float, default=None, help="_")
    parser.add_argument("--rand_perspective", type=float, default=None, help="_")
    parser.add_argument("--horizontal_flip", action='store_true', help="_")
    parser.add_argument("--random_resized_crop", type=float, default=None, help="_")
    parser.add_argument("--random_rotation", type=float, default=None, help="_")
    # Paths parameters
    parser.add_argument("--pca_dataset_folder", type=str, default=None,
                        help="Path with images to be used to compute PCA (ie: pitts30k/images/train")
    parser.add_argument("--save_dir", type=str, default="default",
                        help="Folder name of the current run (saved in ./logs/)")
    args = parser.parse_args()
    
    if args.eval_datasets_folder == None:
        try:
            args.eval_datasets_folder = os.environ['DATASETS_FOLDER']
        except KeyError:
            raise Exception("You should set the parameter --datasets_folder or export " +
                            "the DATASETS_FOLDER environment variable as such \n" +
                            "export DATASETS_FOLDER=../datasets_vg/datasets")
    
    if args.queries_per_epoch % args.cache_refresh_rate != 0:
        raise ValueError("Ensure that queries_per_epoch is divisible by cache_refresh_rate, " +
                         f"because {args.queries_per_epoch} is not divisible by {args.cache_refresh_rate}")
    
    if torch.cuda.device_count() >= 2 and args.criterion in ['sare_joint', "sare_ind"]:
        raise NotImplementedError("SARE losses are not implemented for multiple GPUs, " +
                                  f"but you're using {torch.cuda.device_count()} GPUs and {args.criterion} loss.")
    
    if args.mining == "msls_weighted" and args.dataset_name != "msls":
        raise ValueError("msls_weighted mining can only be applied to msls dataset, but you're using it on {args.dataset_name}")
    
    if args.off_the_shelf in ["radenovic_sfm", "radenovic_gldv1", "naver"]:
        if args.backbone not in ["resnet50conv5", "resnet101conv5"] or args.aggregation != "gem" or args.fc_output_dim != 2048:
            raise ValueError("Off-the-shelf models are trained only with ResNet-50/101 + GeM + FC 2048")
    
    if args.pca_dim != None and args.pca_dataset_folder == None:
        raise ValueError("Please specify --pca_dataset_folder when using pca")
    
    return args

