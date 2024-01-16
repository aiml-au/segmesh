import argparse
import torch
import h5py
import numpy as np
from segmesh.utils import ConfigLoader
from segmesh.networks.scene_seg import PicassoNetII
from eval.transform_texture import S3DISTransformTexture, ScannetTransformTexture
from segmesh.utils import S3DISAlign as s3dis_align_fn
from segmesh.utils import ScannetAlign as scannet_align_fn
import matplotlib.pyplot as plt
import logging
import open3d as o3d
import open3d.visualization.gui as gui
from typing import Optional, Tuple, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(model_path: str, config: dict, device: torch.device) -> PicassoNetII:
    """
    Load the trained scene segmentation model.

    Args:
        model_path (str): Path to the trained model checkpoint.
        config (dict): Configuration dictionary with model and dataset details.
        device (torch.device): The device to load the model on.

    Returns:
        PicassoNetII: The loaded model.
    """
    NUM_CLASSES = len(config['classnames'])
    model = PicassoNetII(num_class=NUM_CLASSES, stride=[4, 3, 3, 2, 2], spharm_L=config['degree'], use_height=True).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

class MeshInferenceForSceneSegmentation:
    def __init__(self, model: PicassoNetII, device: torch.device, transform_fn) -> None:
        """
        Initialize the scene inference class.

        Args:
            model (PicassoNetII): Trained model for scene segmentation.
            device (torch.device): Device to perform inference on.
            transform_fn: Function to transform the input data for inference.
        """
        self.model = model.to(device)
        self.device = device
        self.transform = transform_fn

    def load_and_preprocess(self, mesh_file: str, label_file: str) -> Optional[Tuple[torch.Tensor, ...]]:
        """
        Load and preprocess data from given paths.

        Args:
            mesh_file (str): Path to the mesh file.
            label_file (str): Path to the label file.

        Returns:
            Optional[Tuple[torch.Tensor, ...]]: Preprocessed data or None if an error occurs.
        """
        try:
            # Apply transformation directly on file paths
            if self.transform:
                vertex, face, nv, mf, texture, bcoeff, kt, dense_label, nn_idx = self.transform(mesh_file, label_file)
            return vertex.to(self.device), face.to(self.device), texture.to(self.device), bcoeff.to(self.device), kt.to(self.device), dense_label.to(self.device), nn_idx.to(self.device)
        except Exception as e:
            logging.error(f"Error processing file {mesh_file}: {e}")
            return None

    def infer(self, mesh_file: str, label_file: str) -> Optional[np.ndarray]:
        """
        Perform inference on a single mesh and return predicted labels.

        Args:
            mesh_file (str): Path to the mesh file for inference.
            label_file (str): Path to the label file.

        Returns:
            Optional[np.ndarray]: Predicted labels or None if an error occurs.
        """
        data = self.load_and_preprocess(mesh_file, label_file)
        if data is None:
            return None
        vertex, face, texture, bcoeff, kt, dense_label, nn_idx = data
        nv = torch.tensor([vertex.shape[0]], dtype=torch.int).to(self.device)
        mf = torch.tensor([face.shape[0]], dtype=torch.int).to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred_logits = self.model(vertex, face, nv, mf, texture, bcoeff, kt)
            pred_labels = torch.argmax(pred_logits, dim=-1)
            return pred_labels.cpu().numpy()
    
    def get_colormap(self, num_classes: int) -> List[Tuple[float, float, float]]:
        """
        Generate a colormap for visualization.

        Args:
            num_classes (int): Number of classes.

        Returns:
            List[Tuple[float, float, float]]: List of RGB colors.
        """
        cmap = plt.get_cmap("tab20", num_classes)
        return [cmap(i)[:3] for i in range(num_classes)]

    def create_color_legend(self, classnames, colormap):
        """Create a color legend as a separate window."""
        legend_window = gui.Application.instance.create_window("Color Legend", 200, 400)
        legend_widget = gui.Vert(0, gui.Margins(10, 10, 10, 10))

        for i, classname in enumerate(classnames):
            color = colormap[i]
            color_rect = gui.ColorEdit()
            color_rect.color_value = gui.Color(color[0], color[1], color[2])
            color_rect.enabled = False  # Make it non-interactive
            text_label = gui.Label(classname)

            # Create a horizontal layout for each class
            h_layout = gui.Horiz(0, gui.Margins(10, 5, 10, 5))
            h_layout.add_child(color_rect)
            h_layout.add_child(text_label)

            legend_widget.add_child(h_layout)

        legend_window.add_child(legend_widget)
        return legend_window

    def visualize_predictions(self, dense_pts: np.ndarray, pred_labels: np.ndarray, classnames: List[str]) -> None:
        """
        Visualize the prediction results in a 3D scene with labels.

        Args:
            dense_pts (np.ndarray): 3D points of the scene.
            pred_labels (np.ndarray): Predicted labels for each point.
            classnames (List[str]): List of class names.
        """
        if isinstance(dense_pts, torch.Tensor):
            dense_pts = dense_pts.cpu().numpy()

        # Generate color map
        colormap = self.get_colormap(len(classnames))

        # Create a point cloud for the scene
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(dense_pts)
        colors = [colormap[label] for label in pred_labels]
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Create and run Open3D application
        app = gui.Application.instance
        app.initialize()

        # Main visualization window
        vis = o3d.visualization.O3DVisualizer("Open3D - Scene Segmentation", 1024, 768)
        vis.show_settings = True
        vis.add_geometry("Segmented Scene", pcd)
        vis.reset_camera_to_default()
        app.add_window(vis)

        # Create a separate window for color legend
        colormap = self.get_colormap(len(classnames))
        legend_window = self.create_color_legend(classnames, colormap)
        app.run()

def main():
    parser = argparse.ArgumentParser(description="Scene Segmentation Inference")
    parser.add_argument('--config', required=True, help='Path to the config file')
    parser.add_argument('--dataset', required=True, choices=['s3dis', 'scannet'], help='Dataset to evaluate')
    parser.add_argument('--model_path', required=True, help='Path to the trained model')
    parser.add_argument('--mesh_file', required=True, help='Path to the mesh file for inference')
    parser.add_argument('--label_file', required=True, help='Path to the label file')
    args = parser.parse_args()
    # Load configuration
    config_loader = ConfigLoader(args.config)
    config = config_loader.load_config()
    device = torch.device(f"cuda:{config['gpu']}" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, config, device)
    classnames = config['classnames']
    if args.dataset == 's3dis':
        transform_fn = S3DISTransformTexture(config, voxel_size=3, alpha=5, beta=3, align_fn=s3dis_align_fn())
        inference = MeshInferenceForSceneSegmentation(model, device, transform_fn)
        predictions = inference.infer(args.mesh_file, args.label_file)
        print("Predicted Labels:", predictions)
        # Get dense points for visualization
        with h5py.File(args.mesh_file, 'r') as hf:
            dense_pts = torch.tensor(np.asarray(hf.get('vertex')), dtype=torch.float).to(device)
        # Visualize predictions
        inference.visualize_predictions(dense_pts, predictions, classnames)
    elif args.dataset == 'scannet':
        transform_fn = ScannetTransformTexture(config, align_fn=scannet_align_fn(), raw_mesh_dir=config['raw_mesh_dir'])
        inference = MeshInferenceForSceneSegmentation(model, device, transform_fn)
        predictions = inference.infer(args.mesh_file, args.label_file)
        print("Predicted Labels:", predictions)
        # Get dense points for visualization
        with h5py.File(args.mesh_file, 'r') as hf:
            dense_pts = torch.tensor(np.asarray(hf.get('vertex')), dtype=torch.float).to(device)
        # Visualize predictions
        inference.visualize_predictions(dense_pts, predictions, classnames)
    else:
        logging.info("Dataset {} is not supported".format(args.dataset))

"""
# Example command to run inference for S3DIS dataset:
# python -m inference.inference_scene_seg --config ./config/s3dis.yaml --dataset s3dis --model_path ./runs_scenes/s3dis_render_20230913_184719/model_epoch_50 \
    # --mesh_file ./data/S3DIS_3cm_hdf5_Rendered/Area_1/conferenceRoom_1.h5 --label_file ./data/S3DIS_3cm_hdf5_Rendered/Area_1/conferenceRoom_1.txt
    
# Example command to run inference for Scannet dataset:
# python -m inference.inference_scene_seg --config ./config/scannet.yaml --dataset scannet --model_path ./runs_scenes/scannet_render_20230919_160357/model_epoch_100 \
#     --mesh_file ./data/ScanNet_2cm_hdf5_Rendered/val/scene0231_00.h5 --label_file ./data/ScanNet_2cm_hdf5_Rendered/val/scene0231_00.txt
"""
if __name__=="__main__":
    main()