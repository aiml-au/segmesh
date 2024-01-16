import warnings
import os
import logging
import torch
from glob import glob
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from segmesh.networks.shape_cls import PicassoNetII
from segmesh.datasets import ShapeDataset
from segmesh.datasets import CustomCollateShapeDataset as collate_fn
from torch.utils.data import DataLoader
from fit import MyFit
from segmesh.utils import ShapeNetCore as augment_fn
from segmesh.utils import Normalize as normalize_fn
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
        self.write_folder = f'runs_shapes/shapenetcore_CAD_{timestamp}'  # Updated for shapenetcore
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

    def train(self) -> None:
        """Training loop."""
        MAX_EPOCHS = self.config['max_epochs']
        train_loader = self.dataset_handler.get_train_loader()
        test_loader = self.dataset_handler.get_test_loader()
        fit = MyFit(model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
                    writer=self.writer, loss=self.loss_fn, device=self.device, fout=self.fout)
        fit(self.ckpt_epoch, MAX_EPOCHS, train_loader, test_loader, self.write_folder)

    def backup_training_procedure(self) -> None:
        """Backup the training procedure."""
        os.system(f'cp {__file__} {self.write_folder}')  # Backup of train procedure
        os.system(f'cp picasso/networks/shape_cls.py {self.write_folder}')  # Updated for shapenetcore

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
        """Set up data loaders for the shapenetcore dataset."""
        Meshes, Labels = {}, {}

        train_files = [line.rstrip() for line in open(os.path.join(self.config['data_dir'], 'train_files.txt'))]
        val_files = [line.rstrip() for line in open(os.path.join(self.config['data_dir'], 'val_files.txt'))]
        test_files = [line.rstrip() for line in open(os.path.join(self.config['data_dir'], 'test_files.txt'))]

        Meshes['train'] = [f"{self.config['data_dir']}/{filename}/models/mesh.obj" for filename in train_files]
        Meshes['val'] = [f"{self.config['data_dir']}/{filename}/models/mesh.obj" for filename in val_files]
        Meshes['test'] = [f"{self.config['data_dir']}/{filename}/models/mesh.obj" for filename in test_files]
        Labels['train'] = [f"{self.config['data_dir']}/{filename}/models/label.txt" for filename in train_files]
        Labels['val'] = [f"{self.config['data_dir']}/{filename}/models/label.txt" for filename in val_files]
        Labels['test'] = [f"{self.config['data_dir']}/{filename}/models/label.txt" for filename in test_files]

        Meshes['train'] = Meshes['train'] + Meshes['val']
        Labels['train'] = Labels['train'] + Labels['val']

        # build training set dataloader
        trainSet = ShapeDataset(Labels['train'], Meshes['train'], transform=augment_fn(prob=0.5), normalize=normalize_fn())
        self.train_loader = DataLoader(trainSet, batch_size=self.config['batch_size'], shuffle=True, num_workers=8, collate_fn=collate_fn())
        
        # build validation set dataloader
        testSet = ShapeDataset(Labels['test'], Meshes['test'], transform=None, normalize=normalize_fn())
        self.test_loader = DataLoader(testSet, batch_size=6, shuffle=False, num_workers=0, collate_fn=collate_fn())

    def get_train_loader(self) -> DataLoader:
        """Get the training data loader."""
        return self.train_loader

    def get_test_loader(self) -> DataLoader:
        """Get the test data loader."""
        return self.test_loader

def main():
    """Main function to set up and start training for the shapenetcore dataset."""
    # Load configuration from YAML file
    config_loader = ConfigLoader("./config/shapenetcore.yaml")
    config = config_loader.load_config()

    # Setup CUDA device
    device = torch.device(f"cuda:{config['gpu']}" if torch.cuda.is_available() else "cpu")

    # Determine number of classes based on dataset
    NUM_CLASSES = config['num_classes']

    # Setup model, optimizer, loss function
    model = PicassoNetII(num_class=NUM_CLASSES, stride=[3,2,2,2], spharm_L=config['degree'], use_height=False).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['gamma'])

    # Setup data handler
    dataset_handler = DatasetHandler(config)
    trainer = BaseTrainer(model=model, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn, device=device, dataset_handler=dataset_handler, config=config)

    # Start training
    logger.info("Starting training for the shapenetcore dataset...")
    trainer.train()

if __name__ == '__main__':
    main()