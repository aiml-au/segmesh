from .base_dataset import BaseMeshDataset
import numpy as np
import torch
import open3d as o3d
from typing import Tuple

class ShapeDataset(BaseMeshDataset):
    """
    Dataset class for custom mesh data stored in standard formats like .obj.
    """

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get an item from the ShapeDataset.

        Args:
        - idx (int): Index of the item to fetch.

        Returns:
        - Tuple[torch.Tensor, ...]: A tuple containing mesh data.
        """
        mesh_path = self.mesh_files[idx]
        label_path = self.mesh_labels[idx]

        # Attempt to read the mesh data from the given path
        try:
            raw_mesh = o3d.io.read_triangle_mesh(mesh_path)
        except Exception as e:
            raise ValueError(f"Error reading mesh from {mesh_path}: {e}")

        # Convert the raw mesh data to torch tensors
        vertex = torch.tensor(raw_mesh.vertices, dtype=torch.float)
        face = torch.tensor(raw_mesh.triangles, dtype=torch.int32)
        
        # Load label data
        try:
            label = np.loadtxt(label_path, dtype=np.int32)
        except Exception as e:
            raise ValueError(f"Error reading label from {label_path}: {e}")

        # Ensure face indices start from 0
        assert face.min() == 0, "Face indices should start from 0."

        # get torch tensors of (vertex, face, label)
        vertex = torch.tensor(vertex).to(torch.float)
        face = torch.tensor(face).to(torch.long)
        label = torch.tensor(label).view(-1)

        if self.normalize:
            vertex = self.normalize(vertex)

        if self.transform:
            vertex, face, label = self.transform(vertex, face, label)

        face = face.to(torch.int)
        nv = torch.tensor(vertex.shape[0]).to(torch.int)
        mf = torch.tensor(face.shape[0]).to(torch.int)

        return vertex, face, nv, mf, label
