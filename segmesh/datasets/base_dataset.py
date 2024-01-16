from torch.utils.data import Dataset
from typing import List, Callable, Optional

class BaseMeshDataset(Dataset):
    """
    Base class for mesh datasets.

    Attributes:
    - mesh_labels (List[str]): Paths to the annotation files.
    - mesh_files (List[str]): Paths to the mesh files.
    - transform (Callable, optional): Transformation function to apply to the mesh data.
    - normalize (Callable, optional): Normalization function to apply to the mesh vertices.
    """

    def __init__(self, annotations_files: List[str], mesh_files: List[str],
                 transform: Optional[Callable] = None,
                 normalize: Optional[Callable] = None):
        """
        Initialize the BaseMeshDataset.

        Args:
        - annotations_files (List[str]): List of paths to annotation files.
        - mesh_files (List[str]): List of paths to mesh files.
        - transform (Callable, optional): Optional transformation function to apply on the mesh data.
        - normalize (Callable, optional): Optional normalization function to apply on the mesh vertices.
        """
        super(BaseMeshDataset, self).__init__()
        self.mesh_labels = annotations_files
        self.mesh_files = mesh_files
        self.transform = transform
        self.normalize = normalize

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.mesh_files)

    def __getitem__(self, idx: int):
        """
        Abstract method to get an item from the dataset. 
        This method must be implemented by subclasses.
        
        Args:
        - idx (int): Index of the item to fetch.
        
        Raises:
        - NotImplementedError: This is an abstract method and should be overridden in subclasses.
        """
        raise NotImplementedError