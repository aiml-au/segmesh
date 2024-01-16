import torch
import numpy as np
import open3d as o3d
from segmesh.networks.shape_cls import PicassoNetII
from segmesh.utils import ConfigLoader
from segmesh.utils import NormalizeCubes as normalize_fn_cubes
from segmesh.utils import Normalize as normalize_fn_
from typing import Tuple, Optional
import logging
import argparse
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MeshInferenceForShapeClassPrediction:
    """
    This class handles inference for shape classification using the PicassoNetII model.

    Attributes:
        model (torch.nn.Module): The PicassoNetII model for inference.
        device (torch.device): Device (CPU/GPU) to perform computation.
        preprocessor (HumanSegPreprocess): Preprocessing utility for human segmentation.
        normalize (Function): Normalization function for mesh data.
    """

    def __init__(self, model_path: str, device: str, config: dict, normalize_fn, stride: list = None):
        """
        Args:
            model_path (str): Path to the trained model checkpoint.
            device (str): Device type (e.g., 'cpu', 'cuda:0').
            config (dict): Configuration dictionary.
            normalize_fn (callable): Function to normalize the mesh data.
            stride (list, optional): Stride configuration for the model. Defaults to None.
        """
        self.device = torch.device(device)
        self.model = PicassoNetII(num_class=config['num_classes'], spharm_L=config['degree'], use_height=False, stride=stride if stride else None).to(self.device)
        self.load_model(model_path)
        self.normalize = normalize_fn()

    def load_model(self, model_path: str) -> None:
        """
        Load the model from the specified checkpoint.

        Args:
            model_path (str): Path to the model checkpoint.
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def preprocess_data(self, mesh_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Preprocess the mesh data from the specified path.

        Args:
            mesh_path (str): Path to the mesh file.

        Returns:
            Tuple containing vertices, faces, nv (number of vertices), and mf (number of faces).
        """
        try:
            raw_mesh = o3d.io.read_triangle_mesh(mesh_path)
        except Exception as e:
            raise ValueError(f"Error reading mesh from {mesh_path}: {e}")

        vertex = torch.tensor(np.asarray(raw_mesh.vertices), dtype=torch.float).to(self.device)
        face = torch.tensor(np.asarray(raw_mesh.triangles), dtype=torch.int).to(self.device)

        if self.normalize:
            vertex = self.normalize(vertex)

        nv = torch.tensor([vertex.shape[0]], dtype=torch.int).to(self.device)
        mf = torch.tensor([face.shape[0]], dtype=torch.int).to(self.device)
        return vertex, face, nv, mf

    def inference(self, mesh_path: str) -> int:
        """
        Perform inference on a single mesh and return the predicted class.

        Args:
            mesh_path (str): Path to the mesh file.

        Returns:
            int: The predicted class index.

        Raises:
            Exception: If an error occurs during inference.
        """
        try:
            # Preprocess the mesh to get vertices, faces, and other required tensors
            vertices, faces, nv, mf = self.preprocess_data(mesh_path)

            # Ensure model and data are on the same device
            self.model.to(self.device)

            with torch.no_grad():
                # Perform model inference
                pred_logits = self.model(vertices, faces, nv, mf)
                # Get the predicted class index
                predicted_class = torch.argmax(pred_logits, dim=-1).item()  # Assuming the last dimension holds class scores

            logging.info("Inference completed for {}. Predicted Class: {}".format(mesh_path, predicted_class))
            return predicted_class
        except Exception as e:
            logging.error("Inference failed for {}: {}".format(mesh_path, e))
            raise e

    def eval(self, mesh_path: str, label_path: Optional[str]) -> None:
        """
        Evaluate the inference result against a ground truth label.

        Args:
            mesh_path (str): Path to the mesh file.
            label_path (Optional[str]): Path to the label file. If None, skips the evaluation against ground truth.

        Raises:
            Exception: If an error occurs during evaluation.
        """
        try:
            predicted_class = self.inference(mesh_path)
            if os.path.exists(label_path):
                ground_truth_class = self.read_class_from_label_file(label_path)
                if predicted_class == ground_truth_class:
                    logging.info("Correct prediction for {}".format(mesh_path))
                else:
                    logging.warning("Incorrect prediction for {}: Predicted {} but was {}".format(mesh_path, predicted_class, ground_truth_class))
            else:
                logging.warning("Label file not found for evaluation: {}".format(label_path))
        except Exception as e:
            logging.error("Evaluation failed for {}: {}".format(mesh_path, e))
            raise e

    def read_class_from_label_file(self, label_path: str) -> int:
        """
        Read the class index from a label file.

        Args:
            label_path (str): Path to the label file.

        Returns:
            int: Class index read from the label file.
        """
        try:
            with open(label_path, 'r') as file:
                class_index = int(file.readline().strip())
                return class_index
        except Exception as e:
            raise ValueError(f"Error reading class index from {label_path}: {e}")

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Mesh Inference for Shape Classification")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--mesh_file', type=str, required=True, help='Path to the input mesh file (.obj)')
    parser.add_argument('--label_file', type=str, required=True, help='Path to the label file')
    parser.add_argument('--config_file', type=str, default='./config/cubes.yaml', help='Path to the configuration file')

    # Parse arguments
    args = parser.parse_args()

    # Load configuration from YAML file
    config_loader = ConfigLoader(args.config_file)
    config = config_loader.load_config()
    print(config)
    
    # Setup CUDA device
    device = torch.device(f"cuda:{config['gpu']}" if torch.cuda.is_available() else "cpu")
    config['num_classes'] = len(config['classnames']) if 'classnames' in config else None
    stride = len(config['stride']) if 'stride' in config else None

    # If you are not infering on the cubes dataset samples, replace `normalize_fn_cubes` with `normalize_fn_`
    inference = MeshInferenceForShapeClassPrediction(args.model_path, device, config, normalize_fn_cubes, stride=stride)
    inference.eval(args.mesh_file, args.label_file)