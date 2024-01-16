import warnings
import os
import logging
import torch
import numpy as np
from glob import glob
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from segmesh.networks.scene_seg import PicassoNetII
from segmesh.datasets import SceneDataset, CustomCollateSceneDataset
from torch.utils.data import DataLoader
from fit import MyFit
from segmesh.utils import S3DISAugment as augment_fn
from segmesh.utils import S3DISAlign as align_fn
from segmesh.utils import ConfigLoader
from typing import Dict

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup logging
logging_format = "%(asctime)s - [%(filename)s:%(lineno)s] - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=logging_format)
logger = logging.getLogger(__name__)

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class BaseTrainer:
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler, loss_fn: torch.nn.Module,
             device: torch.device, dataset_handler, config: Dict):
        """Initialize the BaseTrainer.

        Args:
            model: The neural network model.
            optimizer: The optimizer for training.
            loss_fn: The loss function.
            device: The device (CPU/GPU) to use.
            dataset_handler: Handler for data loading.
            config: Configuration dictionary.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.dataset_handler = dataset_handler
        self.config = config
        self.writer = None
        self.ckpt_epoch = 0
        self.fout = None
        self.setup_logging_and_checkpoint()
        self.scheduler = scheduler

    def setup_logging_and_checkpoint(self) -> None:
        """Set up logging and checkpoints."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.write_folder = f'runs_scenes/s3dis_render_{timestamp}'
        if self.config['ckpt_path']:
            self.write_folder = os.path.dirname(self.config['ckpt_path'])
            self.load_model_from_checkpoint()
        else:
            self.ckpt_epoch = 0
        logger.info(f"Write folder: {self.write_folder}")
        self.setup_summary_writer()
        self.fout = open(os.path.join(self.write_folder, 'log_train.txt'), 'a')
        self.fout.write(str(self.config) + '\n')
        self.backup_training_procedure()

    def load_model_from_checkpoint(self) -> None:
        """Load model from a given checkpoint."""
        try:
            checkpoint = torch.load(self.config['ckpt_path'])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.ckpt_epoch = checkpoint['epoch'] + 1
            logger.info(f"Loaded model from epoch {self.ckpt_epoch}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")

    def setup_summary_writer(self) -> None:
        """Set up the summary writer for TensorBoard."""
        self.writer = SummaryWriter(self.write_folder)

    def backup_training_procedure(self) -> None:
        """Backup the training procedure."""
        os.system(f'cp {__file__} {self.write_folder}')  # Backup of train procedure
        os.system(f'cp segmesh/mesh/layers.py {self.write_folder}')
        os.system(f'cp segmesh/networks/scene_seg.py {self.write_folder}')

    def train(self) -> None:
        """Training loop."""
        # hyperparameter settings
        MAX_EPOCHS = self.config['max_epochs']
        classnames = self.config['classnames']
        train_loader = self.dataset_handler.get_train_loader()
        test_loader = self.dataset_handler.get_test_loader()
        fit = MyFit(model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
                    writer=self.writer, loss=self.loss_fn, device=self.device, fout=self.fout)
        fit(self.ckpt_epoch, MAX_EPOCHS, train_loader, test_loader, self.write_folder,
            report_iou=True, class_names=classnames)

    def evaluate(self, test_loader) -> None:
        """Evaluation loop."""
        pass

class DatasetHandler:
    def __init__(self, config: Dict):
        """Initialize the DatasetHandler.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.train_loader = None
        self.test_loader = None
        self.setup_data()

    def setup_data(self) -> None:
        """Set up data loaders."""
        Meshes, Labels = {"train": [], "test": []}, {"train": [], "test": []}
        area_ids = [*range(1, 7)]
        area_ids.remove(self.config['test_fold'])
        for id in area_ids:
            Meshes['train'] += glob(f"{self.config['data_dir']}/Area_{id}/*.h5")
        Labels['train'] = [filename.replace('.h5', '.txt') for filename in Meshes['train']]
        Meshes['test'] = glob(f"{self.config['data_dir']}/Area_{self.config['test_fold']}/*.h5")
        Labels['test'] = [filename.replace('.h5', '.txt') for filename in Meshes['test']]
        repeat = np.ceil((600 * self.config['batch_size']) / len(Meshes['train'])).astype('int32')
        
        # Use SceneDataset for s3dis dataset
        self.train_loader = DataLoader(
            SceneDataset(Labels['train'] * repeat, Meshes['train'] * repeat, transform=augment_fn(0.5), normalize=align_fn()),
            shuffle=True, batch_size=1, num_workers=6, collate_fn=CustomCollateSceneDataset(batch_size=self.config['batch_size'], max_nv=self.config['max_nv'])
        )
        self.test_loader = DataLoader(
            SceneDataset(Labels['test'], Meshes['test'], transform=None, normalize=align_fn()),
            shuffle=False, batch_size=1, num_workers=0, collate_fn=CustomCollateSceneDataset(batch_size=1, max_nv=self.config['max_nv'])
        )

    def get_train_loader(self) -> DataLoader:
        """Get the training data loader."""
        return self.train_loader

    def get_test_loader(self) -> DataLoader:
        """Get the test data loader."""
        return self.test_loader

def main():
    """Main function to set up and start training."""
    # Load configuration from YAML file
    config_loader = ConfigLoader("./config/s3dis.yaml")
    config = config_loader.load_config()

    logger.info(config)

    # Setup CUDA device
    device = torch.device(f"cuda:{config['gpu']}" if torch.cuda.is_available() else "cpu")

    # Setup model, optimizer, loss function
    NUM_CLASSES = len(config['classnames'])
    model = PicassoNetII(num_class=NUM_CLASSES, stride=[4, 3, 3, 2, 2], spharm_L=config['degree'], use_height=True).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['gamma'])

    # Setup data handler
    dataset_handler = DatasetHandler(config)
    trainer = BaseTrainer(model=model, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn, device=device, dataset_handler=dataset_handler, config=config)

    # Start training
    logger.info("Starting training...")
    trainer.train()

if __name__ == '__main__':
    main()