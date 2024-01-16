import torch
from typing import List, Tuple, Union

# ---------------------------------------------------------------------------------------------------------
class CustomCollateShapeDataset:
    """
    Custom collate function to handle the collation of batch data for shape mesh datasets.
    """

    def __call__(self, batch_data: List[Tuple[torch.Tensor]]) -> Tuple[torch.Tensor]:
        """
        Collate data.

        Args:
        - batch_data (list): List of data to be collated.

        Returns:
        - tuple: Collated data.
        """
        # Extract vertices, faces, and labels from the batch data
        batch_vertex = [item[0] for item in batch_data]
        batch_nv = [item[2] for item in batch_data]
        batch_mf = [item[3] for item in batch_data]
        batch_label = [item[-1] for item in batch_data]

        # Concatenate the data along the first dimension
        batch_vertex = torch.concat(batch_vertex, dim=0)
        batch_label = torch.concat(batch_label, dim=0)
        batch_mf = torch.tensor(batch_mf)

        # Calculate vertex offsets for face indices
        vid_offsets = torch.cumsum(torch.tensor([0] + batch_nv), dim=0)
        # Adjust face indices based on the vertex offsets
        batch_face = [item[1] + vid_offsets[i] for i, item in enumerate(batch_data)]
        batch_face = torch.concat(batch_face, dim=0)
        batch_nv = torch.tensor(batch_nv)

        return batch_vertex, batch_face, batch_nv, batch_mf, batch_label

# ---------------------------------------------------------------------------------------------------------
class CustomCollateSceneDataset:
    """
    Custom collate function to handle the collation of batch data for scene mesh datasets.
    """

    def __init__(self, batch_size: int = 1, max_nv: int = 1000000):
        """
        Initialize the CustomCollate class.

        Args:
        - batch_size (int): The size of the batch.
        - max_nv (int): Maximum number of vertices.
        """
        self.batch = []
        self.batch_size = batch_size
        self.max_nv = max_nv

    def __call__(self, data: List[Tuple[torch.Tensor]]) -> Union[None, Tuple[torch.Tensor]]:
        """
        Collate data.

        Args:
        - data (list): List of data to be collated.

        Returns:
        - tuple: Collated data.
        """
        self.batch += data
        trunc_id = self._trunc_batch_()
        # Check if the truncated batch size exceeds the current batch size
        if trunc_id==(len(self.batch)-1):
            batch_data = self._collate_fn_(self.batch[:trunc_id])
            self.batch = self.batch[trunc_id:]
            return batch_data

        # If the batch reaches the desired batch size, collate the data
        if len(self.batch)==self.batch_size:
            batch_data = self._collate_fn_(self.batch)
            self.batch = []
            return batch_data

    def _trunc_batch_(self) -> int:
        """
        Truncate the batch based on the maximum number of vertices.

        Returns:
        - int: The size of the truncated batch.
        """
        batch_nv = [item[2] for item in self.batch]
        batch_nv = torch.tensor(batch_nv)
        cumsum_nv = torch.cumsum(batch_nv, dim=0)
        valid_indices = torch.where(cumsum_nv <= self.max_nv)[0]
        if valid_indices.shape[0]>0:
            trunc_batch_size = valid_indices[-1] + 1
        else:
            trunc_batch_size = 1
        return trunc_batch_size


    def _collate_fn_(self, batch_data: List[Tuple[torch.Tensor]]) -> Tuple[torch.Tensor]:
        """
        Collate the batch data.

        Args:
        - batch_data (list): List of data to be collated.

        Returns:
        - tuple: Collated data.
        """
        # Extract texture, barycentric coefficients, and labels from the batch data
        batch_texture = [item[4] for item in batch_data]
        batch_bcoeff = [item[5] for item in batch_data]
        batch_kt = [item[6] for item in batch_data]
        batch_label = [item[7] for item in batch_data]

        # Concatenate the data along the first dimension
        batch_texture = torch.concat(batch_texture, dim=0)
        batch_bcoeff = torch.concat(batch_bcoeff, dim=0)
        batch_kt = torch.concat(batch_kt, dim=0)
        batch_label = torch.concat(batch_label, dim=0)

        # Extract vertices and faces from the batch data
        batch_vertex = [item[0] for item in batch_data]
        batch_nv = [item[2] for item in batch_data]
        batch_mf = [item[3] for item in batch_data]

        batch_vertex = torch.concat(batch_vertex, dim=0)
        batch_mf = torch.tensor(batch_mf)

        # Calculate vertex offsets for face indices
        vid_offsets = torch.cumsum(torch.tensor([0] + batch_nv), dim=0)
        # Adjust face indices based on the vertex offsets
        batch_face = [item[1] + vid_offsets[i] for i, item in enumerate(batch_data)]
        batch_face = torch.concat(batch_face, dim=0)
        batch_nv = torch.tensor(batch_nv)

        return batch_vertex, batch_face, batch_nv, batch_mf, \
            batch_texture, batch_bcoeff, batch_kt, batch_label
# ---------------------------------------------------------------------------------------------------------