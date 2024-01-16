import torch
from torchvision import transforms
from segmesh.augmentor import Augment
from typing import Tuple, Any

class BaseAugment:
    """
    Base class for augmentation operations.
    """
    def __call__(self, *args, **kwargs) -> Any:
        """
        Apply the augmentation operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Augmented data.
        """
        raise NotImplementedError("This method should be implemented in a child class.")

class S3DISAugment(BaseAugment):
    """
    Augmentation operations specific to the S3DIS dataset.
    """
    def __init__(self, prob: float = 0.5) -> None:
        self.prob = prob
        CustomColorJitter = transforms.ColorJitter(brightness=(0.75, 1.25), contrast=(0.5, 1.5))
        self.color_transform = torch.nn.Sequential(CustomColorJitter, transforms.RandomGrayscale(0.1))

    def __call__(self, vertex: torch.Tensor, face: torch.Tensor, label: torch.Tensor,
                 texture: torch.Tensor, bcoeff: torch.Tensor, kt: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Apply the S3DIS specific augmentation operations.

        Args:
            vertex: Vertex data.
            face: Face data.
            label: Label data.
            texture: Texture data.
            bcoeff: Bcoeff data.
            kt: Kt data.

        Returns:
            Augmented data.
        """
        assert vertex.shape[-1] == 3, "Vertex data should have 3 columns for x, y, z coordinates."

        # Geometry augmentation
        vertex = Augment.rotate_point_cloud(vertex, upaxis=3, prob=self.prob)
        vertex = Augment.rotate_perturbation_point_cloud(vertex, prob=self.prob)
        vertex = Augment.flip_point_cloud(vertex, prob=0.5)
        vertex = Augment.random_scale_point_cloud(vertex, prob=self.prob)
        vertex = Augment.shift_point_cloud(vertex, prob=self.prob)

        # Texture augmentation
        texture = Augment.shift_color(texture, prob=self.prob)
        texture = Augment.jitter_color(texture, prob=self.prob)
        texture = texture.permute([1, 0])[..., None]
        texture = self.color_transform(texture)
        texture = torch.squeeze(texture)
        texture = texture.permute([1, 0])

        # Random drop out faces
        vertex, face, label, \
        face_mask = Augment.random_drop_vertex(vertex, face, label,
                                               drop_rate=0.15, prob=0.5,
                                               return_face_mask=True)
        valid_indices = torch.repeat_interleave(face_mask, kt, dim=0)
        texture = texture[valid_indices]
        bcoeff = bcoeff[valid_indices]
        kt = kt[face_mask]

        assert vertex.shape[0] == label.shape[0], "Vertex and label dimensions mismatch."
        assert face.shape[0] == kt.shape[0], "Face and kt dimensions mismatch."
        assert texture.shape[0] == bcoeff.shape[0], "Texture and bcoeff dimensions mismatch."

        return vertex, face, label, texture, bcoeff, kt

class ScannetAugment(BaseAugment):
    """
    Augmentation class for Scannet dataset.

    Attributes:
        prob (float): Probability for applying certain augmentations.
        color_transform (torch.nn.Sequential): Sequence of color transformations.
    """

    def __init__(self, prob: float = 0.5) -> None:
        """
        Initializes the Scannet augmentation class.

        Args:
            prob (float): Probability for applying certain augmentations.
        """
        self.prob = prob
        CustomColorJitter = transforms.ColorJitter(brightness=(0.75, 1.25), contrast=(0.5, 1.5))
        self.color_transform = torch.nn.Sequential(CustomColorJitter, transforms.RandomGrayscale(0.1))

    def __call__(self, vertex: torch.Tensor, face: torch.Tensor, label: torch.Tensor,
                 texture: torch.Tensor, bcoeff: torch.Tensor, kt: torch.Tensor) -> tuple:
        """
        Applies augmentations to the input data.

        Args:
            vertex (torch.Tensor): Vertex data.
            face (torch.Tensor): Face data.
            label (torch.Tensor): Label data.
            texture (torch.Tensor): Texture data.
            bcoeff (torch.Tensor): Bcoeff data.
            kt (torch.Tensor): Kt data.

        Returns:
            tuple: Augmented data.
        """
        assert vertex.shape[-1] == 3, "Vertex shape mismatch"

        # Geometry augmentation
        vertex = Augment.rotate_point_cloud(vertex, upaxis=3, prob=self.prob)
        vertex = Augment.rotate_perturbation_point_cloud(vertex, prob=self.prob)
        vertex = Augment.flip_point_cloud(vertex, prob=0.5)
        vertex = Augment.random_scale_point_cloud(vertex, prob=self.prob)
        vertex = Augment.shift_point_cloud(vertex, prob=self.prob)

        # Texture augmentation
        texture = Augment.shift_color(texture, prob=self.prob)
        texture = Augment.jitter_color(texture, prob=self.prob)
        texture = texture.permute([1, 0])[..., None]
        texture = self.color_transform(texture)
        texture = torch.squeeze(texture)
        texture = texture.permute([1, 0])

        # Random drop out faces
        vertex, face, label, face_mask = Augment.random_drop_vertex(
            vertex, face, label, drop_rate=0.15, prob=0.5, return_face_mask=True
        )
        valid_indices = torch.repeat_interleave(face_mask, kt, dim=0)
        texture = texture[valid_indices]
        bcoeff = bcoeff[valid_indices]
        kt = kt[face_mask]

        return vertex, face, label, texture, bcoeff, kt

class HumanAugment(BaseAugment):
    """
    Augmentation class for human dataset.

    Attributes:
        prob (float): Probability for applying certain augmentations.
    """

    def __init__(self, prob: float = 0.5) -> None:
        """
        Initializes the HumanAugment class.

        Args:
            prob (float): Probability for applying certain augmentations.
        """
        self.prob = prob

    def __call__(self, vertex: torch.Tensor, face: torch.Tensor, label: torch.Tensor) -> tuple:
        """
        Applies augmentations to the input data.

        Args:
            vertex (torch.Tensor): Vertex data.
            face (torch.Tensor): Face data.
            label (torch.Tensor): Label data.

        Returns:
            tuple: Augmented data.
        """
        assert vertex.shape[1] == 3, "Vertex shape mismatch"

        # Geometry augmentation
        vertex = Augment.flip_point_cloud(vertex, prob=self.prob)
        vertex = Augment.random_scale_point_cloud(vertex, scale_low=0.5, scale_high=1.5, prob=self.prob)
        for axis in [1, 2, 3]:
            vertex = Augment.rotate_point_cloud(vertex, upaxis=axis, prob=self.prob)
        vertex = Augment.jitter_point_cloud(vertex, sigma=0.01, prob=self.prob)

        # Random drop out faces
        vertex, face, label = Augment.random_drop_vertex(vertex, face, label, drop_rate=0.15, prob=0.5)

        return vertex, face, label

class ShrecAugment(HumanAugment):
    """
    Augmentation class for Shrec dataset.
    Inherits from HumanAugment since the augmentations are the same.
    """
    pass


class ShapeNetCore(BaseAugment):
    """
    Augmentation class for ShapeNetCore dataset.

    Attributes:
        prob (float): Probability for applying certain augmentations.
    """

    def __init__(self, prob: float = 0.5) -> None:
        """
        Initializes the ShapeNetCore augmentation class.

        Args:
            prob (float): Probability for applying certain augmentations.
        """
        self.prob = prob

    def __call__(self, vertex: torch.Tensor, face: torch.Tensor, label: torch.Tensor) -> tuple:
        """
        Applies augmentations to the input data.

        Args:
            vertex (torch.Tensor): Vertex data.
            face (torch.Tensor): Face data.
            label (torch.Tensor): Label data.

        Returns:
            tuple: Augmented data.
        """
        assert vertex.shape[-1] == 3, "Vertex shape mismatch"

        # Geometry augmentation
        vertex = Augment.flip_point_cloud(vertex, prob=self.prob)
        vertex = Augment.random_scale_point_cloud(vertex, scale_low=0.5, scale_high=1.5, prob=self.prob)
        vertex = Augment.rotate_point_cloud(vertex, upaxis=3, prob=self.prob)
        vertex = Augment.rotate_perturbation_point_cloud(vertex, prob=self.prob)
        vertex = Augment.jitter_point_cloud(vertex, sigma=0.01, prob=self.prob)

        # Random drop out faces
        vertex, face, label = Augment.random_drop_vertex(vertex, face, label, drop_rate=0.15, prob=0.5)

        return vertex, face, label

class CosegAugment(BaseAugment):
    """
    Augmentation class for Coseg dataset.

    Attributes:
        prob (float): Probability for applying certain augmentations.
    """

    def __init__(self, prob: float = 0.5) -> None:
        """
        Initializes the CosegAugment class.

        Args:
            prob (float): Probability for applying certain augmentations.
        """
        self.prob = prob

    def __call__(self, vertex: torch.Tensor, face: torch.Tensor, label: torch.Tensor) -> tuple:
        """
        Applies augmentations to the input data.

        Args:
            vertex (torch.Tensor): Vertex data.
            face (torch.Tensor): Face data.
            label (torch.Tensor): Label data.

        Returns:
            tuple: Augmented data.
        """
        assert vertex.shape[1] == 3, "Vertex shape mismatch"

        # Geometry augmentation
        vertex = Augment.flip_point_cloud(vertex, prob=self.prob)
        vertex = Augment.random_scale_point_cloud(vertex, scale_low=0.5, scale_high=1.5, prob=self.prob)
        for axis in [1, 2, 3]:
            vertex = Augment.rotate_point_cloud(vertex, upaxis=axis, prob=self.prob)
        vertex = Augment.jitter_point_cloud(vertex, sigma=0.01, prob=self.prob)

        # Random drop out faces
        vertex, face, label = Augment.random_drop_vertex(vertex, face, label, drop_rate=0.15, prob=0.5)

        return vertex, face, label


class CubesAugment(CosegAugment):
    """
    Augmentation class for Cubes dataset.
    Inherits from CosegAugment since the augmentations are the same.
    """
    pass


class FaustMatchAugment(CosegAugment):
    """
    Augmentation class for FaustMatch dataset.
    Inherits from CosegAugment since the augmentations are the same.
    """
    pass