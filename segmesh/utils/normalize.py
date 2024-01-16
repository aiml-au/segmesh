import torch

class BaseNormalize:
    """
    Base class for normalization operations.
    """
    def __call__(self, vertex: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the vertex data.

        Args:
            vertex: The vertex data to be normalized.

        Returns:
            Normalized vertex data.
        """
        raise NotImplementedError("This method should be implemented in a child class.")
        
class Normalize(BaseNormalize):
    """
    Normalization operation to scale vertex data.
    """
    def __call__(self, vertex: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the vertex data by scaling.

        Args:
            vertex: The vertex data to be normalized.

        Returns:
            Normalized vertex data.
        """
        assert vertex.shape[-1] == 3, "Vertex data should have 3 columns for x, y, z coordinates."
        xyz_min = torch.min(vertex, dim=0)[0]
        xyz_max = torch.max(vertex, dim=0)[0]
        xyz_center = (xyz_min + xyz_max) / 2
        vertex -= xyz_center
        scale = torch.max(xyz_max - xyz_min) / 2
        vertex /= scale
        return vertex

class NormalizeCubes(BaseNormalize):
    """
    Normalization operation to center vertex data in a cube.
    """
    def __call__(self, vertex: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the vertex data by centering in a cube.

        Args:
            vertex: The vertex data to be normalized.

        Returns:
            Normalized vertex data.
        """
        assert vertex.shape[-1] == 3, "Vertex data should have 3 columns for x, y, z coordinates."
        xyz_min = torch.min(vertex, dim=0)[0]
        xyz_max = torch.max(vertex, dim=0)[0]
        xyz_center = (xyz_min + xyz_max) / 2
        vertex -= xyz_center
        return vertex