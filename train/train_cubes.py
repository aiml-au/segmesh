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
from segmesh.utils import CubesAugment as augment_fn
from segmesh.utils import NormalizeCubes as normalize_fn
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
                 device: torch.device, dataset_handler, config: dict):
        """Initialize the BaseTrainer for cubes dataset.

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
        """Set up logging and checkpoints for cubes dataset."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.write_folder = f'runs_shapes/cubes_CAD_{timestamp}'
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
        """Backup the training procedure for cubes dataset."""
        os.system(f'cp {__file__} {self.write_folder}')  # Backup of train procedure
        os.system(f'cp picasso/networks/shape_cls.py {self.write_folder}')

    def train(self) -> None:
        """Training loop for cubes dataset."""
        MAX_EPOCHS = self.config['max_epochs']
        #classnames = ['cube_class%d' % i for i in range(len(self.config['classnames']))]  # Assuming 22 classes for cubes
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
    def __init__(self, config : Dict):
        self.data_dir = config['data_dir']
        self.batch_size = config['batch_size']
        self.train_mesh_files = []
        self.train_label_files = []
        self.test_mesh_files = []
        self.test_label_files = []
        #self.classnames = ['cube_class%d' % i for i in range(22)]  # Assuming 22 classes for cubes
        self.classnames = config['classnames']
        self._prepare_data_paths()

    def _prepare_data_paths(self):
        for clsname in self.classnames:
            train_list = glob(f"{self.data_dir}/{clsname}/train/*.obj")
            self.train_mesh_files += train_list
            self.train_label_files += [f"{self.data_dir}/{clsname}/label.txt"] * len(train_list)

            test_list = glob(f"{self.data_dir}/{clsname}/test/*.obj")
            self.test_mesh_files += test_list
            self.test_label_files += [f"{self.data_dir}/{clsname}/label.txt"] * len(test_list)

    def get_train_loader(self):
        repeat = self.batch_size * 500 // len(self.train_mesh_files) + 1
        train_set = ShapeDataset(self.train_label_files * repeat, self.train_mesh_files * repeat,
                                      transform=augment_fn(prob=0.5), normalize=normalize_fn())
        return DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=3,
                          collate_fn=collate_fn())

    def get_test_loader(self):
        test_set = ShapeDataset(self.test_label_files, self.test_mesh_files, transform=None, normalize=normalize_fn())
        return DataLoader(test_set, batch_size=16, shuffle=False, num_workers=0,
                          collate_fn=collate_fn())


def main():
    """Main function to set up and start training for the cubes dataset."""
    # Load configuration from YAML file
    # Assuming a new configuration file for the cubes dataset
    config_loader = ConfigLoader("./config/cubes.yaml")
    config = config_loader.load_config()

    # Setup CUDA device
    device = torch.device(f"cuda:{config['gpu']}" if torch.cuda.is_available() else "cpu")

    # Number of classes for the Cubes dataset
    NUM_CLASSES = len(config['classnames'])

    # Setup model, optimizer, loss function
    model = PicassoNetII(num_class=NUM_CLASSES, spharm_L=config['degree'], use_height=False).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['gamma'])

    # Setup data handler
    dataset_handler = DatasetHandler(config)
    trainer = BaseTrainer(model=model, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn, device=device, dataset_handler=dataset_handler, config=config)

    # Start training
    logger.info("Starting training for the cubes dataset...")
    trainer.train()

if __name__ == '__main__':
    main()
