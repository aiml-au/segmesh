import warnings
import os
import logging
import torch
from glob import glob
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from segmesh.networks.shape_seg import PicassoNetII
from segmesh.datasets import CustomMeshDataset
from segmesh.datasets import CustomCollateShapeDataset as collate_fn
from torch.utils.data import DataLoader
from segmesh.utils import FaustMatchAugment as augment_fn
from segmesh.utils import Normalize as normalize_fn
from segmesh.utils import ConfigLoader
from fit import MyFit
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
        """Initialize the BaseTrainer for the FAUST dataset.

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
        self.scheduler = scheduler
        self.setup_logging_and_checkpoint()

    def setup_logging_and_checkpoint(self) -> None:
        """Set up logging and checkpoints for the FAUST dataset."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.write_folder = f'runs_faust/FAUST_{timestamp}'
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
        """Backup the training procedure for the FAUST dataset."""
        os.system(f'cp {__file__} {self.write_folder}')  # Backup of train procedure
        os.system(f'cp picasso/networks/shape_seg.py {self.write_folder}')

    def train(self) -> None:
        """Training loop for the FAUST dataset."""
        MAX_EPOCHS = self.config['max_epochs']
        train_loader = self.dataset_handler.get_train_loader()
        test_loader = self.dataset_handler.get_test_loader()
        fit = MyFit(model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
                    writer=self.writer, loss=self.loss_fn, device=self.device, fout=self.fout)
        fit(self.ckpt_epoch, MAX_EPOCHS, train_loader, test_loader, self.write_folder,
        report_iou=False, class_names=classnames)

    def evaluate(self, test_loader) -> None:
        """Evaluation loop."""
        pass

class DatasetHandler:
    def __init__(self, config: Dict):
        self.data_dir = config['data_dir']
        self.batch_size = config['batch_size']
        self.train_mesh_files = []
        self.train_label_files = []
        self.test_mesh_files = []
        self.test_label_files = []
        self._prepare_data_paths()

    def _prepare_data_paths(self):
        mesh_files = glob(f'{self.data_dir}/*.ply')
        mesh_files.sort()
        label_file = f'{os.path.dirname(self.data_dir)}/match_labels.txt'
        self.train_mesh_files = mesh_files[:80]
        self.test_mesh_files = mesh_files[80:]
        self.train_label_files = [label_file] * len(self.train_mesh_files)
        self.test_label_files = [label_file] * len(self.test_mesh_files)

    def get_train_loader(self):
        repeat = self.batch_size * 500 // len(self.train_mesh_files) + 1
        train_set = ShapeDataset(self.train_label_files * repeat, self.train_mesh_files * repeat,
                                      transform=augment_fn(0.5), normalize=normalize_fn())
        return DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=4,
                          collate_fn=train_set.collate_fn)

    def get_test_loader(self):
        test_set = ShapeDataset(self.test_label_files, self.test_mesh_files,
                                     transform=None, normalize=normalize_fn())
        return DataLoader(test_set, batch_size=1, shuffle=False,
                          collate_fn=test_set.collate_fn)

def main():
    """Main function to set up and start training for the faust match dataset."""
    # Load configuration from YAML file
    # Assuming a new configuration file for the faust match dataset
    config_loader = ConfigLoader("./config/faust_match.yaml")
    config = config_loader.load_config()

    # Setup CUDA device
    device = torch.device(f"cuda:{config['gpu']}" if torch.cuda.is_available() else "cpu")

    # Number of classes for the Faust Match dataset
    NUM_CLASSES = len(config['classnames'])
    classnames = [str(i) for i in range(NUM_CLASSES)]
    # Setup model, optimizer, loss function
    model = PicassoNetII(num_class=NUM_CLASSES, stride=[3, 2, 2, 2], pred_facet=False, spharm_L=config['degree'],
                         use_height=False).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['gamma'])

    # Setup data handler
    dataset_handler = DatasetHandler(config)
    trainer = BaseTrainer(model=model, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn, device=device, dataset_handler=dataset_handler, config=config)

    # Start training
    logger.info("Starting training for the faust match dataset...")
    trainer.train()

if __name__ == '__main__':
    main()
