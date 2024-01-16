import argparse
import logging
import torch
from segmesh.utils import ConfigLoader
from eval.s3dis_eval import S3DISEvaluator
from eval.scannet_eval import ScanNetEvaluator
from segmesh.networks.scene_seg import PicassoNetII
from eval.transform_texture import S3DISTransformTexture, ScannetTransformTexture
from segmesh.utils import S3DISAlign as s3dis_align_fn
from segmesh.utils import ScannetAlign as scannet_align_fn

# Setup logging
logging_format = "%(asctime)s - [%(filename)s:%(lineno)s] - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=logging_format)
logger = logging.getLogger(__name__)

def load_model(config, device):
    NUM_CLASSES = len(config['classnames'])
    model = PicassoNetII(num_class=NUM_CLASSES, stride=[4, 3, 3, 2, 2], spharm_L=config['degree'], use_height=True).to(device)
    checkpoint = torch.load(config['ckpt_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def main():
    parser = argparse.ArgumentParser(description="Dataset Evaluator")
    parser.add_argument('--config', required=True, help='path to config file')
    parser.add_argument('--dataset', required=True, choices=['s3dis', 'scannet'], help='dataset to evaluate')
    args = parser.parse_args()

    # Load configuration
    config_loader = ConfigLoader(args.config)
    config = config_loader.load_config()
    device = torch.device(f"cuda:{config['gpu']}" if torch.cuda.is_available() else "cpu")
    model = load_model(config, device)

    # Choose the evaluator based on the dataset
    if args.dataset == 's3dis':
        voxel_grid, alpha, beta = (3, 5, 3)
        transform = S3DISTransformTexture(config, voxel_size=voxel_grid, alpha=alpha, beta=beta, align_fn=s3dis_align_fn())
        s3dis_evaluator = S3DISEvaluator(model=model, device=device, transform_fn=transform, class_names=config['classnames'], loss_fn=torch.nn.CrossEntropyLoss(), data_dir=config['data_dir'], test_fold=config['test_fold'])
        logger.info("Evaluating S3DIS dataset...")
        s3dis_evaluator.evaluate()
    elif args.dataset == 'scannet':
        transform = ScannetTransformTexture(config, align_fn=scannet_align_fn(), raw_mesh_dir=config['raw_mesh_dir'])
        scannet_evaluator = ScanNetEvaluator(model=model, device=device, transform_fn=transform, class_names=config['classnames'], loss_fn=torch.nn.CrossEntropyLoss(), data_dir=config['data_dir'])
        logger.info("Evaluating ScanNet dataset...")
        scannet_evaluator.evaluate()

if __name__ == "__main__":
    main()