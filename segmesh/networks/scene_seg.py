import numpy as np
import torch
import torch.nn as nn
import segmesh.mesh.utils as meshUtil
from segmesh.mesh.layers import V2FConv3d, PerItemConv3d, SphHarmCoeff
from segmesh.mesh.layers import Pool3d as mesh_pool
from segmesh.mesh.layers import Unpool3d as mesh_unpool
from .pi2_blocks import DualBlock, EncoderMeshBlock, FirstBlock
from .pi2_blocks import FirstBlockTexture as FirstBlock


class PicassoNetII(nn.Module):
    def __init__(self, num_class, stride=None, spharm_L=3, use_height=False, pred_facet=False):
        super(PicassoNetII,self).__init__()

        if stride is None:
            self.stride = [4, 3, 3, 2, 2]
        else:
            self.stride = stride
        self.num_class = num_class

        self.L = spharm_L
        self.useArea, self.wgtBnd = (True, 1.) # recommended hyper-parameters for mesh decimation
        self.bn_momentum = 0.01
        self.num_blocks = 5
        self.Niters = [2, 2, 4, 4, 4]
        self.radius = [0.2, 0.4, 0.8]
        self.EncodeChannels = [32, 64, 96, 128, 192, 256]
        self.useHeight = use_height
        if self.useHeight:
            geometry_in_channels = 12
        else:
            geometry_in_channels = 9
        self.predFacet = pred_facet

        self.min_nv = 100  # set the minimum vertex number for the coarsest layer
        self.stride = np.float32(self.stride)
        self.nv_limit = np.int32(np.cumprod(self.stride[::-1])[::-1]*self.min_nv)

        self.build_mesh_hierarchy = meshUtil.MeshHierarchy(self.stride, self.nv_limit,
                                                           self.useArea, self.wgtBnd)
        self.cal_sph_harm = SphHarmCoeff(self.L)

        # FirstBlock
        self.Conv0 = FirstBlock([geometry_in_channels,3], self.EncodeChannels[0],
                                self.L, bn_momentum=self.bn_momentum)

        # setting of convolution blocks in the encoder
        growth_rate = 32
        Enc1 = EncoderMeshBlock(self.EncodeChannels[0], self.EncodeChannels[1], growth_rate,
                                self.L, max_iter=self.Niters[0],
                                bn_momentum=self.bn_momentum)
        Enc2 = EncoderMeshBlock(self.EncodeChannels[1], self.EncodeChannels[2], growth_rate,
                                self.L, max_iter=self.Niters[1],
                                bn_momentum=self.bn_momentum)
        Enc3 = DualBlock(self.EncodeChannels[2], self.EncodeChannels[3], growth_rate,
                         self.L, radius=self.radius[0], max_iter=self.Niters[2],
                         bn_momentum=self.bn_momentum)
        Enc4 = DualBlock(self.EncodeChannels[3], self.EncodeChannels[4], growth_rate,
                         self.L, radius=self.radius[1], max_iter=self.Niters[3],
                         bn_momentum=self.bn_momentum)
        Enc5 = DualBlock(self.EncodeChannels[4], self.EncodeChannels[5], growth_rate,
                         self.L, radius=self.radius[2], max_iter=self.Niters[4],
                         bn_momentum=self.bn_momentum)
        self.Block = nn.ModuleList([Enc1, Enc2, Enc3, Enc4, Enc5])

        # reducing channels of low-level features in the encoder
        self.Decode = []
        self.DecodeChannels = [self.EncodeChannels[-1], 128, 128, 96, 96, 96]
        for k in range(self.num_blocks):
            m = self.num_blocks - k - 1
            self.Decode.append(PerItemConv3d(self.EncodeChannels[m]+self.DecodeChannels[k],
                                             self.DecodeChannels[k+1], bn_momentum=self.bn_momentum))
        self.Decode = nn.ModuleList(self.Decode)

        # mesh pooling, unpooling configuration
        self.Mesh_Pool = mesh_pool(method='max')
        self.Mesh_Unpool = mesh_unpool()

        # classifier
        if self.predFacet:
            self.V2F0 = V2FConv3d(96, 96)
        self.Predict = PerItemConv3d(96, self.num_class, with_bn=False, activation_fn=None)

    def extract_facet_geometric_features(self, vertex, face, normals):
        face = face.to(torch.long)
        V1 = vertex[face[:,0],:]
        V2 = vertex[face[:,1],:]
        V3 = vertex[face[:,2],:]

        Height_features = torch.stack([V1[:,2], V2[:,2], V3[:,2]], dim=1)

        Delta12 = V2[:,:3] - V1[:,:3]
        Delta23 = V3[:,:3] - V2[:,:3]
        Delta31 = V1[:,:3] - V3[:,:3]

        # length of edges
        L12 = torch.norm(Delta12, dim=-1, keepdim=True)
        L23 = torch.norm(Delta23, dim=-1, keepdim=True)
        L31 = torch.norm(Delta31, dim=-1, keepdim=True)
        Length_features = torch.cat([L12, L23, L31], dim=-1)

        # angles between edges
        Theta1 = torch.sum(Delta12*(-Delta31), dim=-1, keepdim=True)/(L12*L31)
        Theta2 = torch.sum((-Delta12)*Delta23, dim=-1, keepdim=True)/(L12*L23)
        Theta3 = torch.sum((-Delta23)*Delta31, dim=-1, keepdim=True)/(L23*L31)
        Theta_features = torch.cat([Theta1, Theta2, Theta3], dim=-1)

        if self.useHeight:
            geometry_features = torch.cat([Length_features, Theta_features,
                                           normals, Height_features], dim=-1)
        else:
            geometry_features = torch.cat([Length_features, Theta_features,
                                           normals], dim=-1)
        return geometry_features

    def get_oriented_normals(self, geometry_in, vertex, face):
        face = face.to(torch.long)
        face_normals = geometry_in[:, :3]
        center = (vertex[face[:,0],:] + vertex[face[:,1],:] + vertex[face[:,2],:])/3
        sign = torch.sign(torch.sum(-center*face_normals, dim=-1, keepdim=True)) # [0,0,0] as viewpoint
        sign[sign>-1] = 1.
        face_normals = face_normals*sign
        return face_normals

    def __call__(self, vertex_in, face_in, nv_in, mf_in, facet_textures, bary_coeff, num_texture):
        vertex_input = vertex_in
        vertex_in = vertex_in[:,:3]

        # build mesh hierarchy
        mesh_hierarchy = self.build_mesh_hierarchy(vertex_in, face_in, nv_in, mf_in)

        # import open3d as o3d
        # for i in range(6):
        #     vertex_in, face_in, geometry_in, nv_in = mesh_hierarchy[i][:4]
        #     mesh = o3d.geometry.TriangleMesh()
        #     mesh.vertices = o3d.utility.Vector3dVector(vertex_in[:,:3].cpu().numpy())
        #     mesh.triangles = o3d.utility.Vector3iVector(face_in.cpu().numpy())
        #     # mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_in[:,3:].cpu().numpy())
        #     o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

        decoder_helper = []
        # ============================================Initial Conv==============================================
        vertex_in, face_in, geometry_in, nv_in = mesh_hierarchy[0][:4]
        full_vt_map = torch.arange(vertex_in.shape[0],dtype=torch.int).to(vertex_in.get_device())
        full_nf_count = meshUtil.count_vertex_adjface(face_in, full_vt_map, vertex_in)
        face_normals = self.get_oriented_normals(geometry_in, vertex_in, face_in)
        facet_geometrics = self.extract_facet_geometric_features(vertex_input, face_in, face_normals)
        filt_coeff = self.cal_sph_harm(cart_coords=face_normals)
        bary_spharm_coeff = self.cal_sph_harm(bary_coeff)
        feats = self.Conv0(facet_geometrics, facet_textures, bary_spharm_coeff, num_texture,
                           face_in, full_nf_count, full_vt_map, filt_coeff)

        decoder_helper.append(feats)
        vertex_out, _, _, _, _, vt_replace, vt_map = mesh_hierarchy[1]
        pooled_feats = self.Mesh_Pool(feats, vt_replace, vt_map, vertex_out)  # mesh pool0
        # ===============================================The End================================================

        # ============================================Encoder Flow==============================================
        for k in range(self.num_blocks):
            # block computation:
            vertex_in, face_in, geometry_in, nv_in = mesh_hierarchy[k+1][:4]
            full_vt_map = torch.arange(vertex_in.shape[0],dtype=torch.int).to(vertex_in.get_device())
            full_nf_count = meshUtil.count_vertex_adjface(face_in, full_vt_map, vertex_in)
            face_normals = self.get_oriented_normals(geometry_in, vertex_in, face_in)
            filt_coeff = self.cal_sph_harm(cart_coords=face_normals)
            feats = self.Block[k](pooled_feats, vertex_in, face_in, full_nf_count, full_vt_map, filt_coeff, nv_in)

            if k<(self.num_blocks-1):
                decoder_helper.append(feats)
                vertex_out, _, _, _, _, vt_replace, vt_map = mesh_hierarchy[k+2]
                pooled_feats = self.Mesh_Pool(feats, vt_replace, vt_map, vertex_out)  # mesh pool
        # ===============================================The End================================================

        # ===============================combine with low-level encoder features================================
        for k in range(self.num_blocks):
            # upsampling and feature combination with Encoder features
            it = -k + self.num_blocks
            vt_replace, vt_map = mesh_hierarchy[it][-2:]
            low_level_feats = decoder_helper[it-1]
            upsampled_feats = self.Mesh_Unpool(feats, vt_replace, vt_map)
            feats = torch.concat([low_level_feats, upsampled_feats], dim=-1)
            feats = self.Decode[k](feats)
        # ===============================================The End================================================

        if self.predFacet:
            face_in = mesh_hierarchy[0][1]
            feats = self.V2F0(feats, face_in)
        pred_logits = self.Predict(feats)

        return pred_logits