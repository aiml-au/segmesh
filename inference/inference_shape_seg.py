import torch
import numpy as np
import open3d as o3d
from segmesh.networks.shape_seg import PicassoNetII
from segmesh.utils import ConfigLoader
from segmesh.utils import Normalize as normalize_fn
from typing import Tuple, Optional, List
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MeshInferenceForShapeSegPrediction:
    def __init__(self, model_path: str, device: str, config: dict):
        """
        Initialize the MeshInferenceForShapeSegPrediction class.

        Args:
            model_path (str): Path to the saved model.
            device (str): Device to run the model on ('cpu' or 'cuda').
            config (dict): Configuration dictionary containing model settings.

        Raises:
            ValueError: If the device is not recognized.
        """
        try:
            self.device = device
            self.model = PicassoNetII(num_class=config['num_classes'], 
                                    pred_facet=True, 
                                    spharm_L=config['degree'], 
                                    use_height=False).to(self.device)
            self.load_model(model_path)
            self.normalize = normalize_fn()
            logging.info("Model initialized successfully on device: {}".format(device))
        except Exception as e:
            logging.error("Model initialization failed: {}".format(e))
            raise e

    def load_model(self, model_path: str):
        """
        Load the trained model from a specified path.

        Args:
            model_path (str): Path to the trained model file.

        Raises:
            IOError: If the model file cannot be loaded.
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            logging.info("Model loaded successfully from {}".format(model_path))
        except IOError as e:
            logging.error("Failed to load model from {}: {}".format(model_path, e))
            raise e
    
    def preprocess_data(self, mesh_path: str, label_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Preprocess mesh data and labels for model inference.

        Args:
            mesh_path (str): Path to the mesh file.
            label_path (str): Path to the label file.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
            Vertices, faces, number of vertices (nv), number of faces (mf), and labels as tensors.

        Raises:
            ValueError: If there is an error reading the mesh or label data.
        """
        try:
            raw_mesh = o3d.io.read_triangle_mesh(mesh_path)
            label = np.loadtxt(label_path, dtype=np.int32)

            vertex = torch.tensor(np.asarray(raw_mesh.vertices), dtype=torch.float).to(self.device)
            face = torch.tensor(np.asarray(raw_mesh.triangles), dtype=torch.int).to(self.device)
            label = torch.tensor(label, dtype=torch.long).to(self.device)

            if face.max() >= vertex.shape[0]:
                raise ValueError("Face indices are out of bounds of vertex indices.")
            
            if self.normalize:
                vertex = self.normalize(vertex)

            nv = torch.tensor([vertex.shape[0]], dtype=torch.int).to(self.device)
            mf = torch.tensor([face.shape[0]], dtype=torch.int).to(self.device)
            
            logging.info("Data preprocessed successfully for mesh: {}".format(mesh_path))
            return vertex, face, nv, mf, label
        except Exception as e:
            logging.error("Preprocessing failed for mesh {}: {}".format(mesh_path, e))
            raise e

    def perform_inference(self, vertices: torch.Tensor, faces: torch.Tensor, nv: torch.Tensor, mf: torch.Tensor) -> np.ndarray:
        """
        Perform inference on the provided mesh data.

        Args:
            vertices (torch.Tensor): The vertices of the mesh.
            faces (torch.Tensor): The faces of the mesh.
            nv (torch.Tensor): The number of vertices.
            mf (torch.Tensor): The number of faces.

        Returns:
            np.ndarray: Predicted labels for each vertex.

        Raises:
            RuntimeError: If inference fails.
        """
        try:
            self.model.to(self.device)
            with torch.no_grad():
                pred_logits = self.model(vertices, faces, nv, mf)
                pred_labels = torch.argmax(pred_logits, dim=-1)
                logging.info("Inference performed successfully.")
                return pred_labels.squeeze(0).cpu().numpy()
        except RuntimeError as e:
            logging.error("Inference failed: {}".format(e))
            raise e

    def infer_and_visualize(self, mesh_path: str, label_path: str):
        """
        Perform inference on a mesh and visualize the results.

        Args:
            mesh_path (str): Path to the mesh file.
            label_path (str): Path to the label file.

        Raises:
            RuntimeError: If visualization fails.
        """
        try:
            vertices, faces, nv, mf, label = self.preprocess_data(mesh_path, label_path)
            predictions = self.perform_inference(vertices, faces, nv, mf)
            self.visualize_results(mesh_path, predictions, label)
        except RuntimeError as e:
            logging.error("Visualization failed for mesh {}: {}".format(mesh_path, e))
            raise e
            
    def visualize_results(self, mesh_path: str, predictions: np.ndarray, ground_truth: Optional[np.ndarray] = None):
        """
        Visualize the prediction results on a mesh.

        Args:
            mesh_path (str): Path to the mesh file.
            predictions (np.ndarray): Predicted labels for the mesh.
            ground_truth (Optional[np.ndarray]): Ground truth labels for the mesh.

        Raises:
            RuntimeError: If the mesh cannot be visualized.
        """
        try:
            original_mesh = o3d.io.read_triangle_mesh(mesh_path)
            mesh_pred = o3d.geometry.TriangleMesh()
            mesh_pred.vertices = original_mesh.vertices
            mesh_pred.triangles = original_mesh.triangles
            colors_pred = np.array([self.get_color(label) for label in predictions])
            mesh_pred.vertex_colors = o3d.utility.Vector3dVector(colors_pred)
            # Translate the prediction mesh to the right
            bbox = original_mesh.get_axis_aligned_bounding_box()
            translation_distance = bbox.get_max_bound()[0] - bbox.get_min_bound()[0]
            mesh_pred.translate([translation_distance, 0, 0], relative=True)
            # mesh_pred.compute_vertex_normals()
            # mesh_pred.paint_uniform_color([1, 0.206, 0])
            if ground_truth is not None:
                mesh_gt = o3d.geometry.TriangleMesh()
                mesh_gt.vertices = original_mesh.vertices
                mesh_gt.triangles = original_mesh.triangles
                colors_gt = np.array([self.get_color(label) for label in ground_truth])
                mesh_gt.vertex_colors = o3d.utility.Vector3dVector(colors_gt)
                o3d.visualization.draw_geometries([mesh_gt, mesh_pred], window_name="Ground Truth vs Prediction")
            else:
                o3d.visualization.draw_geometries([mesh_pred], window_name="Prediction")
            logging.info("Mesh visualization complete for {}".format(mesh_path))
        except RuntimeError as e:
            logging.error("Failed to visualize mesh {}: {}".format(mesh_path, e))
            raise e

    def get_color(self, label: int) -> List[float]:
        """
        Get the color corresponding to a label.

        Args:
            label (int): The label index.

        Returns:
            List[float]: RGB color values for the label.

        Note:
            The method uses a predefined color map. Unknown labels are assigned a default color.
        """
        color_map = {
            0: [1, 1, 1],  # White for background or label 0
            1: [1, 0, 0],  # Red for label 1
            2: [0, 1, 0],  # Green for label 2
            3: [0, 0, 1],  # Blue for label 3
            # Add more colors as needed
        }
        return color_map.get(label, [0.5, 0.5, 0.5])  # Default gray color for unknown labels

"""
Sample command to run this script
python inference_shape_seg.py --dataset human --model_path './runs_shapes/human_20230922_121456/model_epoch_100' --mesh_file './data/human_seg/test/shrec__10.obj' --label_file './data/human_seg/face_label/shrec__10.txt'
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on different datasets.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (e.g., 'human', 'coseg_vases', 'coseg_aliens').")
    parser.add_argument("--model_path", type=str, required=True, help="Epoch number of the model to use.")
    parser.add_argument("--mesh_file", type=str, required=True, help="Path to the mesh file for inference.")
    parser.add_argument("--label_file", type=str, required=False, help="Path to the label file (if applicable).")

    args = parser.parse_args()

    # Config file should be present in the config directory
    config_file = f'./config/{args.dataset}.yaml'

    # Load configuration
    config_loader = ConfigLoader(config_file)
    config = config_loader.load_config()

    # Setup CUDA device
    device = torch.device(f"cuda:{config['gpu']}" if torch.cuda.is_available() else "cpu")

    # Number of classes (optional, depends on how your config is structured)
    config['num_classes'] = len(config['classnames']) if 'classnames' in config else config['num_classes']

    # Setup model
    inference = MeshInferenceForShapeSegPrediction(args.model_path, device, config)
    inference.infer_and_visualize(args.mesh_file, args.label_file)