import torch

class BaseAlign:
    """
    Base class for alignment operations.
    """
    def __call__(self, vertex: torch.Tensor) -> torch.Tensor:
        """
        Aligns the vertex data.

        Args:
            vertex: The vertex data to be aligned.

        Returns:
            Aligned vertex data.
        """
        raise NotImplementedError("This method should be implemented in a child class.")

class S3DISAlign(BaseAlign):
    """
    Alignment operation specific to the S3DIS dataset.
    """
    def __call__(self, vertex: torch.Tensor) -> torch.Tensor:
        """
        Aligns the vertex data for the S3DIS dataset.

        Args:
            vertex: The vertex data to be aligned.

        Returns:
            Aligned vertex data.
        """
        assert vertex.shape[-1] == 3, "Vertex data should have 3 columns for x, y, z coordinates."
        xyz_min = torch.min(vertex, dim=0, keepdim=True)[0]
        xyz_max = torch.max(vertex, dim=0, keepdim=True)[0]
        xyz_center = (xyz_min + xyz_max) / 2
        xyz_center[0][-1] = xyz_min[0][-1]
        vertex -= xyz_center
        return vertex

class ScannetAlign(S3DISAlign):
    """
    Align class for Scannet dataset.
    Inherits from S3DISAlign since the align functions are the same.
    """
    pass