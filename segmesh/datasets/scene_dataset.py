from .base_dataset import BaseMeshDataset
import numpy as np
import torch
import h5py
from typing import Tuple

class SceneDataset(BaseMeshDataset):
    """
    Dataset class for custom mesh data stored in HDF5 format.
    """

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get an item from the SceneDataset.

        Args:
        - idx (int): Index of the item to fetch.

        Returns:
        - Tuple[torch.Tensor, ...]: A tuple containing mesh data.
        """
        mesh_path = self.mesh_files[idx]
        label_path = self.mesh_labels[idx]

        # Attempt to read the label data from the given path
        try:
            label = np.loadtxt(label_path, dtype=np.int32)
        except Exception as e:
            raise ValueError(f"Error reading label from {label_path}: {e}")

        # Attempt to read the mesh data from the given HDF5 path
        try:
            with h5py.File(mesh_path, 'r') as hf:
                vertex = np.asarray(hf.get('vertex'))
                face = np.asarray(hf.get('face'))
                texture = np.asarray(hf.get('face_texture'))
                bcoeff = np.asarray(hf.get('bary_coeff'))
                kt = np.asarray(hf.get('num_texture'))
        except Exception as e:
            raise ValueError(f"Error reading HDF5 data from {mesh_path}: {e}")

        # Convert the loaded data to torch tensors
        vertex = torch.tensor(vertex).to(torch.float)
        face = torch.tensor(face).to(torch.long)
        texture = torch.tensor(texture).to(torch.float)
        bcoeff = torch.tensor(bcoeff).to(torch.float)
        kt = torch.tensor(kt).to(torch.int)
        label = torch.tensor(label).view(-1)

        # Ensure the lengths of the loaded data match the expected lengths
        assert len(vertex) == len(label), "Mismatch between vertex and label lengths."

        # Apply normalization and transformation if provided
        if self.normalize:
            vertex = self.normalize(vertex)
        if self.transform:
            vertex, face, label, texture, bcoeff, kt = self.transform(vertex, face, label, texture, bcoeff, kt)

        face = face.to(torch.int)
        nv = torch.tensor(vertex.shape[0]).to(torch.int)
        mf = torch.tensor(face.shape[0]).to(torch.int)
        return vertex, face, nv, mf, texture, bcoeff, kt, label