import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import torch, h5py
from sklearn.neighbors import NearestNeighbors
import open3d as o3d

class S3DISTransformTexture:
    def __init__(self, config, voxel_size=None, alpha=0, beta=1, align_fn=None):
        self.voxel_size = voxel_size
        self.alpha = alpha
        self.beta = beta
        self._align_fn_ = align_fn
        self.config = config

    def __call__(self, mesh_path, label_path):
        label = np.loadtxt(label_path, dtype=np.int32)
        hf = h5py.File(mesh_path, 'r')
        vertex = np.asarray(hf.get('vertex'))
        face = np.asarray(hf.get('face'))
        texture = np.asarray(hf.get('face_texture'))
        bcoeff = np.asarray(hf.get('bary_coeff'))
        kt = np.asarray(hf.get('num_texture'))
        hf.close()

        vertex = torch.tensor(vertex).to(torch.float)
        face = torch.tensor(face).to(torch.long)
        texture = torch.tensor(texture).to(torch.float)
        bcoeff = torch.tensor(bcoeff).to(torch.float)
        kt = torch.tensor(kt).to(torch.int)
        label = torch.tensor(label).view(-1)

        # load the dense/raw point cloud
        raw_cloud_dir = f"{self.config['raw_data_dir']}/Area_%d"%self.config['test_fold']
        scene_names = os.path.basename(mesh_path).replace('.h5','')
        dense_cloud = np.loadtxt(f"{raw_cloud_dir}/{scene_names}/cloud.txt", delimiter=',')
        dense_label = np.loadtxt(f"{raw_cloud_dir}/{scene_names}/labels.txt")
        dense_pts = torch.tensor(dense_cloud[:,:3]).to(torch.float)
        dense_label = torch.tensor(dense_label)

        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(vertex)
        nn_dst, nn_idx = nbrs.kneighbors(dense_pts)
        nn_idx = torch.tensor(nn_idx.reshape(-1)).to(torch.long)

        if self._align_fn_:
            vertex = self._align_fn_(vertex)

        face = face.to(torch.int)
        nv = torch.tensor([vertex.shape[0]]).to(torch.int)
        mf = torch.tensor([face.shape[0]]).to(torch.int)

        return vertex, face, nv, mf, texture, bcoeff, kt, dense_label, nn_idx

class ScannetTransformTexture:
    def __init__(self, config, align_fn=None, knn=5, raw_mesh_dir=None):
        self._align_fn_ = align_fn
        self.knn = knn
        self.raw_mesh_dir = raw_mesh_dir
        self.config = config

    def __call__(self, mesh_path, label_path):
        label = np.loadtxt(label_path, dtype=np.int32)
        hf = h5py.File(mesh_path, 'r')
        vertex = np.asarray(hf.get('vertex'))
        face = np.asarray(hf.get('face'))
        texture = np.asarray(hf.get('face_texture'))
        bcoeff = np.asarray(hf.get('bary_coeff'))
        kt = np.asarray(hf.get('num_texture'))
        hf.close()

        vertex = torch.tensor(vertex).to(torch.float)
        face = torch.tensor(face).to(torch.long)
        texture = torch.tensor(texture).to(torch.float)
        bcoeff = torch.tensor(bcoeff).to(torch.float)
        kt = torch.tensor(kt).to(torch.int)
        label = torch.tensor(label).view(-1)

        # load the dense point cloud
        scene_names = os.path.basename(mesh_path).replace('.h5', '')
        raw_mesh = o3d.io.read_triangle_mesh(f"{self.config['raw_mesh_dir']}/{scene_names}/{scene_names}.ply")
        dense_label = np.loadtxt(f"{self.config['raw_mesh_dir']}/{scene_names}/{scene_names}_scan20_labels.txt")
        dense_pts = torch.tensor(raw_mesh.vertices).to(torch.float)
        dense_label = torch.tensor(dense_label)

        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(vertex)
        nn_dst, nn_idx = nbrs.kneighbors(dense_pts)
        nn_idx = torch.tensor(nn_idx.reshape(-1)).to(torch.long)

        if self._align_fn_:
            vertex = self._align_fn_(vertex)

        face = face.to(torch.int)
        nv = torch.tensor([vertex.shape[0]]).to(torch.int)
        mf = torch.tensor([face.shape[0]]).to(torch.int)
        return vertex, face, nv, mf, texture, bcoeff, kt, dense_label, nn_idx