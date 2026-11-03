import torch
from typing import Optional, Union
import warnings
import os 
import pickle
from manotorch.manolayer import ManoLayer
import numpy as np
import chumpy as ch
from scipy import sparse
import trimesh
from PIL import Image

class HandyLayer(ManoLayer):
    def __init__(self, rot_mode: str = "axisang", side: str = "right", center_idx: Optional[int] = None, mano_assets_root: str = "assets/mano", use_pca: bool = False, flat_hand_mean: bool = True,
                ncomps: int = 15, shape_ncomps: int = 30, handy_assets_root: str = "assets/handy",  **kargs,):
        super().__init__(rot_mode=rot_mode, side=side, center_idx=center_idx, mano_assets_root=mano_assets_root, use_pca=use_pca, flat_hand_mean=flat_hand_mean, ncomps=ncomps, **kargs)

        self.shape_ncomps = shape_ncomps
        
        # load model according to side flag
        handy_assets_path = os.path.join(handy_assets_root, "Handy_Models", f"{side[0].upper()}{side[1:]}_Hand_Shape.pkl")  # eg.  MANO_RIGHT.pkl
        assert os.path.isfile(
            handy_assets_path), f"Can not find Handy assets {handy_assets_path}, please follow steps in README.md"

        with open(handy_assets_path, "rb") as f:
            handy_model = pickle.load(f)
            
        self.register_buffer("th_v_template", torch.Tensor(handy_model["v_template"]))
        self.register_buffer("th_shapescale", torch.Tensor(handy_model["eigenvalues"][:shape_ncomps]))
        self.register_buffer("th_faces", torch.Tensor(handy_model["f"]))
        
        # for uv texture
        self.transform_verts_to_uv = handy_model["transform_to_uv"]
        self.uv_faces = handy_model["f_uv"]
        self.uv_coords = handy_model["uv_coords"]
        
        correspondence_file_path = os.path.join(handy_assets_root, "Mano_7k_Template_correspondence.npy")
        correspondence_dict = np.load(correspondence_file_path, allow_pickle=True).item()
        
        vert_weights = np.stack(correspondence_dict["weights"])
        vert_weights = torch.from_numpy(vert_weights).to(self.th_faces.device)
        corresponding_face_idx = correspondence_dict["faces_idx"]
        # faces_idx  = correspondence_dict['faces_idx']
        # vert_weights = correspondence_dict['weights']
        mano_faces = correspondence_dict['mano_faces']
        
        self.handy2mano_face_idx = corresponding_face_idx
        self.mano2handy_weight = vert_weights
        self.mano_faces = mano_faces
        
        mano2handy_mtx = torch.zeros(7231, 778)
        verts_weight = self.mano_faces[self.handy2mano_face_idx]
        for i in range(3):
            mano2handy_mtx[torch.arange(7231), verts_weight[:, i]] = self.mano2handy_weight[i]
        self.mano2handy_mtx = mano2handy_mtx
        self.handy2mano_mtx = torch.linalg.pinv(mano2handy_mtx)
        
# )
        weights = mano2handy_mtx @ self.th_weights
        J_reg = mano2handy_mtx @ self.th_J_regressor.T
        # weights = get_deformation(self.th_weights, corresponding_face_idx, vert_weights, mano_faces)
        # J_reg = get_deformation(self.th_J_regressor.T, corresponding_face_idx, vert_weights, mano_faces)
        J_reg = J_reg / J_reg.sum(0)
        posesdirs = (mano2handy_mtx @ self.th_posedirs.reshape(778, -1)).reshape(7231, 3, 135)
        
        # posesdirs = get_deformation(self.th_posedirs.reshape(778, -1), corresponding_face_idx, vert_weights, mano_faces).reshape(7231, 3, 135)
        
        self.register_buffer("th_shapedirs", torch.Tensor(handy_model["components"][:shape_ncomps].T.reshape(7231, 3, -1)))
        # shapedirs = get_deformation(self.th_shapedirs.reshape(778, -1), corresponding_face_idx, vert_weights, mano_faces).reshape(7231, 3, -1)
        # self.register_buffer("th_shapedirs", shapedirs)
        self.register_buffer("th_weights", weights)
        self.register_buffer("th_J_regressor", J_reg.T)#torch.from_numpy(sparse.csr_matrix(J_reg).T))
        self.register_buffer("th_posedirs", posesdirs)
    
    def mano_to_handy(self, mano_verts: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            mano_verts (torch.Tensor): (B, 778, D)

        Returns:
            torch.Tensor: handy verts (B, 7231, D)
        """
        if mano_verts.ndim == 2:
            mano_verts = mano_verts.unsqueeze(0)
        assert mano_verts.ndim == 3, "invalid input dimensions"
        assert mano_verts.shape[1] == 778, "invalid num of mano vertices"
        
        handy_verts = torch.einsum("ij,njk->nik", self.mano2handy_mtx, mano_verts)
        return handy_verts
    
    def handy_to_mano(self, handy_verts: torch.Tensor) -> torch.Tensor:
        if handy_verts.ndim == 2:
            handy_verts = handy_verts.unsqueeze(0)
        assert handy_verts.ndim == 3,  "invalid input dimensions"
        assert handy_verts.shape[1] == 7231, "invalid num of mano vertices"
        mano_verts = torch.einsum("ij,njk->nik", self.handy2mano_mtx, handy_verts)
        return mano_verts

    def to_textured_obj(self, handy_verts: Union[torch.Tensor, np.ndarray], texture_img: Image, save_path = None) -> trimesh.Trimesh: 
        assert handy_verts.ndim == 2, "invalid shape, only support one mesh"
        
        if isinstance(handy_verts, torch.Tensor):
            handy_verts = handy_verts.to("cpu").numpy()
        hand_verts = handy_verts[self.transform_verts_to_uv]
        mesh = trimesh.Trimesh(hand_verts, self.uv_faces, 
                               visual = trimesh.visual.texture.TextureVisuals(self.uv_coords , image = texture_img), 
                               process=False)
        
        if save_path is not None:
            mesh.export(save_path)
        return mesh
        
        

if __name__ == "__main__":
    handylayer = HandyLayer(handy_assets_root="assets/HandyModels", use_pca=True, ncomps=6, flat_hand_mean=False, shape_ncomps=10)
    manolayer = ManoLayer(ncomps=6, use_pca=True, flat_hand_mean=False)
    
    sigma = 2
    beta = torch.zeros((1, 10))
    pose = torch.zeros((1, 9))
    pose[:4] = 1
    
    handy_output = handylayer.forward(pose, beta)
    mano_output = manolayer.forward(pose, beta)
    
    from wis3d import Wis3D
    
    wis3d = Wis3D(out_folder="wis3d", sequence_name="handy_output", xyz_pattern=("-x", "-y", "z")) 
    
    ### Same pose vector, but different shape representation, give different shape vector
    wis3d.add_mesh(mano_output.verts[0], manolayer.th_faces, name="mano")
    wis3d.add_mesh(handy_output.verts[0], handylayer.th_faces, name="handy")
    
    ### Note that the MANO template and Handy template is different from each other
    wis3d.add_mesh(manolayer.th_v_template[0], manolayer.th_faces, name="mano_template")
    wis3d.add_mesh(handylayer.th_v_template, handylayer.th_faces, name="handy_template")
    
   
    ### Convert a MANO mesh to Handy mesh or vice versa.
    wis3d.add_point_cloud(mano_output.verts[0], name="mano")
    computed_handy_verts = handylayer.mano_to_handy(mano_output.verts)
    wis3d.add_mesh(computed_handy_verts[0], handylayer.th_faces, name="computed_handy")
    
    inverse_mano_verts = handylayer.handy_to_mano(computed_handy_verts)
    wis3d.add_mesh(inverse_mano_verts[0], manolayer.th_faces, name = "inverse mano")
    
    ### Save the mesh to obj file
    img_path = "path/to/hand_texture.png"
    texture_img = Image.open(img_path) 
    save_path = "mano.obj"
    handylayer.to_textured_obj(computed_handy_verts[0], texture_img, save_path)
