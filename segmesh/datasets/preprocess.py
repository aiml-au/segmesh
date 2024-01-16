import numpy as np
import os
import glob
from scipy import stats
import meshio as mio
import argparse
from segmesh.utils import ConfigLoader
from typing import List, Tuple, Optional

# ---------------------------------------------------------------------------------------------------------
class BasePreprocessClass:
    """
    A base class for preprocessing datasets.

    This class provides the basic structure for preprocessing tasks and should be inherited
    by dataset-specific preprocessing classes.

    Attributes:
        data_dir (str): Directory where the dataset is located.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def read_mesh(self, curr_path):
        """
        Abstract method to read the mesh from the given path.

        Args:
            curr_path (str): Path to the mesh file.

        Returns:
            tuple: Vertices and faces of the mesh.
        """
        raise NotImplementedError

    def normalize_location(self, xyz):
        """
        Abstract method to normalize the location of the mesh vertices.

        Args:
            xyz (np.array): Vertices of the mesh.

        Returns:
            np.array: Normalized vertices.
        """
        raise NotImplementedError

    def transform_labels(self, face, seg_labels):
        """
        Abstract method to transform edge labels to face labels.

        Args:
            face (np.array): Faces of the mesh.
            seg_labels (np.array): Edge labels.

        Returns:
            np.array: Face labels.
        """
        raise NotImplementedError

    def save_labels(self, cls_name, model_name, face_labels):
        """
        Abstract method to save the face labels to a file.

        Args:
            model_name (str): Name of the model.
            face_labels (np.array): Face labels to save.
        """
        raise NotImplementedError

    def save_labels2(self):
        raise NotImplementedError

    def save_labels3(self, labels, filename):
        """
        Abstract method to save the labels to a file.

        Args:
            labels (np.array): Array of labels to save.
            filename (str): Name of the file to save the labels to.
        """
        raise NotImplementedError

# ---------------------------------------------------------------------------------------------------------
class CoSegPreprocess(BasePreprocessClass):
    """
    A class for preprocessing the CoSeg dataset.

    This class inherits from BasePreprocessClass and implements methods specific to the
    CoSeg dataset.

    Dataset Reference:
    [Link to the CoSeg dataset]

    Attributes:
        data_dir (str): Directory where the CoSeg dataset is located.
    """
    def read_mesh(self, curr_path):
        """Read the mesh from the given path and return its vertices and faces."""
        try:
            mesh = mio.read(curr_path)
            xyz = np.asarray(mesh.points, dtype=np.float32)
            face = np.asarray(mesh.cells_dict['triangle'], dtype=np.int32)
            return xyz, face
        except Exception as e:
            print(f"Error reading mesh from {curr_path}: {e}")
            return None, None

    def normalize_location(self, xyz):
        """Normalize the location of the mesh vertices to be centered around the origin."""
        try:
            xyz_min = np.amin(xyz, axis=0, keepdims=True)
            xyz_max = np.amax(xyz, axis=0, keepdims=True)
            xyz_center = (xyz_min + xyz_max) / 2
            xyz -= xyz_center
            return xyz
        except Exception as e:
            print(f"Error normalizing location: {e}")
            return None

    def transform_labels(self, face, seg_labels):
        """Transform edge labels to face labels using a majority voting mechanism."""
        try:
            v1, v2, v3 = face[:, 0], face[:, 1], face[:, 2]
            edge = np.stack([v1, v2, v2, v3, v1, v3], axis=1)
            edge = np.reshape(edge, newshape=[-1, 2])
            edge_pool = np.sort(edge, axis=-1)
            edge, _, inv_indices = np.unique(edge_pool, axis=0, return_index=True, return_inverse=True)
            edge_labels = seg_labels[inv_indices]
            edge_pool_labels = np.reshape(edge_labels, [-1, 3])
            face_labels = stats.mode(edge_pool_labels, axis=1)[0]
            face_labels = np.squeeze(face_labels)
            return face_labels
        except Exception as e:
            print(f"Error transforming labels: {e}")
            return None

    def save_labels(self, cls_name, model_name, face_labels):
        """Save the face labels to a file."""
        try:
            np.savetxt(self.data_dir + f"/{cls_name}/face_label/{model_name}.txt", face_labels, fmt='%d')
            print(f"Saved labels for {model_name} in {cls_name}")
        except Exception as e:
            print(f"Error saving labels for {model_name} in {cls_name}: {e}")

# ---------------------------------------------------------------------------------------------------------
class HumanSegPreprocess(BasePreprocessClass):
    """
    A class for preprocessing the HumanSeg dataset.

    This class inherits from BasePreprocessClass and implements methods specific to the
    HumanSeg dataset.

    Attributes:
        data_dir (str): Directory where the HumanSeg dataset is located.
    """

    def __init__(self, data_dir: str):
        """
        Initialize the HumanSegPreprocess class.

        Args:
            data_dir (str): The directory where the HumanSeg dataset is located.
        """
        self.data_dir = data_dir
        # Create the 'face_label' directory if it doesn't exist
        # if not os.path.exists(os.path.join(self.data_dir, 'face_label')):
        #     os.makedirs(os.path.join(self.data_dir, 'face_label'))

    def read_mesh(self, curr_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Read the mesh from the given path and return its vertices and faces.

        Args:
            curr_path (str): The path to the mesh file.

        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: Vertices and faces of the mesh.
        """
        try:
            mesh = mio.read(curr_path)
            xyz = np.asarray(mesh.points, dtype=np.float32)
            face = np.asarray(mesh.cells_dict['triangle'], dtype=np.int32)
            return xyz, face
        except Exception as e:
            print(f"Error reading mesh from {curr_path}: {e}")
            return None, None

    def normalize_location(self, xyz: np.ndarray) -> Optional[np.ndarray]:
        """
        Normalize the location of the mesh vertices to be centered around the origin.

        Args:
            xyz (np.ndarray): The vertices of the mesh.

        Returns:
            Optional[np.ndarray]: The normalized vertices.
        """
        try:
            xyz_min = np.amin(xyz, axis=0, keepdims=True)
            xyz_max = np.amax(xyz, axis=0, keepdims=True)
            xyz_center = (xyz_min + xyz_max) / 2
            xyz -= xyz_center  # Center the mesh around the origin
            return xyz
        except Exception as e:
            print(f"Error normalizing location: {e}")
            return None

    def transform_labels(self, face: np.ndarray, seg_labels: np.ndarray) -> Optional[np.ndarray]:
        """
        Transform edge labels to face labels using a majority voting mechanism.

        Args:
            face (np.ndarray): The faces of the mesh.
            seg_labels (np.ndarray): The segmentation labels.

        Returns:
            Optional[np.ndarray]: The transformed face labels.
        """
        try:
            # Extract vertices for each face
            v1, v2, v3 = face[:, 0], face[:, 1], face[:, 2]
            edge = np.stack([v1, v2, v2, v3, v1, v3], axis=1)
            edge = np.reshape(edge, newshape=[-1, 2])
            edge_pool = np.sort(edge, axis=-1)
            # Find unique edges and their indices
            edge, uni_indices, \
                inv_indices = np.unique(edge_pool, axis=0, return_index=True, return_inverse=True)
            sortIdx = np.argsort(np.argsort(uni_indices))  # two argsort works
            edge_labels = seg_labels[sortIdx]
            edge_pool_labels = np.reshape(edge_labels[inv_indices], [-1, 3])
            face_labels = stats.mode(edge_pool_labels, axis=1)[0]
            face_labels = np.squeeze(face_labels)

            assert (face_labels.shape[0] == face.shape[0])
            assert (np.amin(face_labels) >= 1 and np.amax(face_labels) <= 8)

            # Adjust labels to start from 0
            face_labels = face_labels - np.min(face_labels)
            return face_labels
        except Exception as e:
            print(f"Error transforming labels: {e}")
            return None

    def save_labels(self, model_name: str, face_labels: np.ndarray):
        """
        Save the face labels to a file.

        Args:
            model_name (str): The name of the model.
            face_labels (np.ndarray): The face labels to save.
        """
        try:
            np.savetxt(self.data_dir + f"/face_label/{model_name[:-4]}.txt", face_labels, fmt='%d')
            print(f"Saved labels for {model_name}")
        except Exception as e:
            print(f"Error saving labels for {model_name}: {e}")

    def preprocess(self, obj_paths: List[str]):
        """
        Perform the preprocessing steps on the given list of object paths.

        Args:
            obj_paths (List[str]): The list of paths to the object files.
        """
        for curr_path in obj_paths:
            print(curr_path)
            model_name = os.path.basename(curr_path)
            label_path = os.path.join(self.data_dir, 'seg', model_name.replace('.obj', '.eseg'))
            seg_labels = np.loadtxt(label_path, dtype=np.int32)

            xyz, face = self.read_mesh(curr_path)
            if xyz is None or face is None:
                continue

            # Assert that face indices start from 0
            assert np.amin(face) == 0, "Face indices should start from 0."

            xyz = self.normalize_location(xyz)
            if xyz is None:
                continue

            face_labels = self.transform_labels(face, seg_labels)
            if face_labels is None:
                continue

            self.save_labels(model_name, face_labels)

# ---------------------------------------------------------------------------------------------------------
class CubesPreprocess(BasePreprocessClass):
    """
    A class for preprocessing the Cubes dataset.

    This class inherits from BasePreprocessClass and implements methods specific to the
    Cubes dataset.

    Dataset Reference:
    [Link to the Cubes dataset]

    Attributes:
        data_dir (str): Directory where the Cubes dataset is located.
    """
    def get_classnames(self):
        """Fetch class names for the Cubes dataset."""
        return ['apple', 'bat', 'bell', 'brick', 'camel', 'car', 'carriage', 'chopper',
                'elephant', 'fork', 'guitar', 'hammer', 'heart', 'horseshoe', 'key',
                'lmfish', 'octopus', 'shoe', 'spoon', 'tree', 'turtle', 'watch']

    def save_labels2(self):
        """Save label files for Cubes dataset in an external directory."""
        try:
            classnames = self.get_classnames()
            for label, clsName in enumerate(classnames):
                label = np.asarray([label])
                np.savetxt(f"{self.data_dir}/{clsName}/label.txt", label, fmt='%d')
                print(f"Saved label for {clsName}")
        except Exception as e:
            print(f"Error saving label for cubes dataset: {e}")

# ---------------------------------------------------------------------------------------------------------
class ShrecPreprocess(BasePreprocessClass):
    """
    A class for preprocessing the SHREC dataset.

    This class inherits from BasePreprocessClass and implements methods specific to the
    SHREC dataset.

    Dataset Reference:
    [Link to the SHREC dataset]

    Attributes:
        data_dir (str): Directory where the SHREC dataset is located.
    """
    def get_classnames(self):
        """Fetch class names for the SHREC dataset."""
        filelist = glob.glob(f"{self.data_dir}/*")
        classnames = [os.path.basename(file) for file in filelist]
        classnames.sort()
        return classnames

    def save_labels2(self):
        """Save label files for SHREC dataset in an external directory."""
        try:
            classnames = self.get_classnames()
            for label, clsName in enumerate(classnames):
                label = np.asarray([label])
                np.savetxt(f"{self.data_dir}/{clsName}/label.txt", label, fmt='%d')
                print(f"Saved label for {clsName}")
            print("\n\n\n")
        except Exception as e:
            print(f"Error saving label for shrec dataset: {e}")

# ---------------------------------------------------------------------------------------------------------
class FaustPreprocess(BasePreprocessClass):
    """
    A class for preprocessing the MPI-Faust dataset.

    This class inherits from BasePreprocessClass and implements methods specific to the
    MPI-Faust dataset.

    Dataset Reference:
    [Link to the MPI-Faust dataset]

    Attributes:
        data_dir (str): Directory where the MPI-Faust dataset is located.
    """

    def save_labels(self, labels, filename):
        """Save the labels to a file for the MPI-Faust dataset."""
        try:
            np.savetxt(os.path.join(os.path.dirname(self.data_dir), filename), labels, fmt='%d')
            print(f"Saved labels to {filename}")
        except Exception as e:
            print(f"Error saving labels to {filename}: {e}")
# ---------------------------------------------------------------------------------------------------------

def process_cubes(data_dir = './data/cubes'):
    """Preprocess the Cubes dataset."""
    processor = CubesPreprocess(data_dir)
    processor.save_labels2()

def process_shrec(data_dir = './data/shrec_16'):
    """Preprocess the SHREC dataset."""
    processor = ShrecPreprocess(data_dir)
    processor.save_labels2()

def process_coseg(classes, data_dir = './data'):
    """Preprocess the COSEG dataset."""
    processor = CoSegPreprocess(data_dir)
    for cls_name in classes:
        if not os.path.exists(data_dir + f"/{cls_name}/face_label"):
            os.makedirs(data_dir + f"/{cls_name}/face_label")
        train_files = glob.glob(os.path.join(data_dir, cls_name, 'train/*.obj'))
        test_files = glob.glob(os.path.join(data_dir, cls_name, 'test/*.obj'))
        obj_paths = train_files + test_files
        for curr_path in obj_paths:
            xyz, face = processor.read_mesh(curr_path)
            xyz = processor.normalize_location(xyz)
            model_name = os.path.basename(curr_path).replace('.obj', '')
            label_path = os.path.join(data_dir, cls_name, 'seg', model_name + '.eseg')
            seg_labels = np.loadtxt(label_path, dtype=np.int32)
            face_labels = processor.transform_labels(face, seg_labels)
            processor.save_labels(cls_name, model_name, face_labels)

def process_humanseg(data_dir = './data/human_seg'):
    """Preprocess the HumanSeg dataset."""
    if not os.path.exists(os.path.join(data_dir, 'face_label')):
        os.makedirs(os.path.join(data_dir, 'face_label'))
    train_files = glob.glob(os.path.join(data_dir, 'train/*.obj'))
    test_files = glob.glob(os.path.join(data_dir, 'test/*.obj'))

    obj_paths = train_files + test_files

    train_files = [os.path.basename(file)[:-4] for file in train_files]
    test_files = [os.path.basename(file)[:-4] for file in test_files]
    np.savetxt(data_dir + f"/train_files.txt", train_files, fmt='%s')
    np.savetxt(data_dir + f"/test_files.txt", test_files, fmt='%s')

    print(len(train_files), len(test_files))

    preprocess = HumanSegPreprocess(data_dir)
    preprocess.preprocess(obj_paths)

def process_faust(data_dir = "./data/MPI-Faust/registrations"):
    """Preprocess the Faust dataset."""
    processor = FaustPreprocess(data_dir)
    num_classes = 6890
    labels = np.arange(num_classes)
    try:
        processor.save_labels(labels, 'match_labels.txt')
    except Exception as e:
        print(f"Error processing MPI-Faust dataset: {e}")

def main(dataset_name, config_file_path):
    try:
        # Load configuration from YAML file
        print("Loading config file : {}".format(config_file_path))
        config_loader = ConfigLoader(config_file_path)
        config = config_loader.load_config()
        if 'data_dir' not in config.keys():
            print("Cannot access data dir in config file. Please check.")
            os._exit(-1)
        data_dir = config['data_dir']
        print("Data directory : {}".format(data_dir))
        if dataset_name == 'cubes':
            process_cubes(data_dir)
        elif dataset_name == 'shrec':
            process_shrec(data_dir)
        elif dataset_name == 'coseg_aliens':
            process_coseg(classes = ['coseg_aliens'])
        elif dataset_name == 'coseg_chairs':
            process_coseg(classes = ['coseg_chairs'])
        elif dataset_name == 'coseg_vases':
            process_coseg(classes=['coseg_vases'])
        elif dataset_name == 'humanseg':
            process_humanseg(data_dir)
        elif dataset_name == 'faust':
            process_faust(data_dir)
        else:
            print(f"Unknown dataset: {dataset_name}")
    except Exception as e:
        print(f"Error processing {dataset_name} dataset: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess different datasets")
    parser.add_argument("--dataset", required=True, help="Name of the dataset to process", choices=['cubes', 'shrec', 'coseg_aliens','coseg_chairs', 'coseg_vases','humanseg', 'faust'])
    parser.add_argument("--config_file", required=True, help="Path to the configuration file")
    args = parser.parse_args()
    main(args.dataset, args.config_file)