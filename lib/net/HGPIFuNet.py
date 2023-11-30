# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de


from lib.renderer.mesh import compute_normal_batch
from lib.dataset.mesh_util import feat_select, read_smpl_constants, surface_field_deformation
from lib.net.NormalNet import NormalNet
from lib.net.MLP import TransformerEncoderLayer
from lib.net.spatial import SpatialEncoder
from lib.dataset.PointFeat import PointFeat
from lib.dataset.mesh_util import SMPLX
from lib.net.VE import VolumeEncoder
from lib.net.ResBlkPIFuNet import ResnetFilter
from lib.net.UNet import UNet
from lib.net.HGFilters import *
from lib.net.Transformer import ViTVQ
from termcolor import colored
from lib.net.BasePIFuNet import BasePIFuNet
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from lib.net.nerf_util import raw2outputs
from torchsummary import summary
import cv2


import torch
from torch.autograd import Function
from torchvision import models




def draw_features(width,height,x,savename,transpose=False,special=False):
    
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)

    idx=[1,6,7,9,11,16,20,26,31,38,39,41,53,55,59,64]
    if transpose:
        # transpose x 
        x=np.transpose(x, (0, 1, 3, 2))
        x=np.flip(x,[2,3])

    for i in range(width*height):
        id=idx[i]-1
        plt.subplot(height,width, i + 1)
        plt.axis('off')
        if special:
            img = x[0, id, :, :]
        else:
            img=x[0,i,:,:]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001))*255  #
        img=img.astype(np.uint8)  #
        img=cv2.applyColorMap(img, cv2.COLORMAP_JET) #
        img = img[:, :, ::-1]#
        plt.imshow(img)
     
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()


class HGPIFuNet(BasePIFuNet):


    def __init__(self,
                 cfg,
                 projection_mode="orthogonal",
                 error_term=nn.MSELoss()):

        super(HGPIFuNet, self).__init__(projection_mode=projection_mode,
                                        error_term=error_term)

        self.l1_loss = nn.SmoothL1Loss()
        self.opt = cfg.net
        self.root = cfg.root
        self.overfit = cfg.overfit

        channels_IF = self.opt.mlp_dim

        self.use_filter = self.opt.use_filter
        self.prior_type = self.opt.prior_type
        self.smpl_feats = self.opt.smpl_feats

        self.smpl_dim = self.opt.smpl_dim
        self.voxel_dim = self.opt.voxel_dim
        self.hourglass_dim = self.opt.hourglass_dim

        self.in_geo = [item[0] for item in self.opt.in_geo]
        self.in_nml = [item[0] for item in self.opt.in_nml]

        self.in_geo_dim = sum([item[1] for item in self.opt.in_geo])
        self.in_nml_dim = sum([item[1] for item in self.opt.in_nml])

        self.in_total = self.in_geo + self.in_nml
        self.smpl_feat_dict = None
        self.smplx_data = SMPLX()

        image_lst = [0, 1, 2]
        normal_F_lst = [0, 1, 2] if "image" not in self.in_geo else [3, 4, 5]
        normal_B_lst = [3, 4, 5] if "image" not in self.in_geo else [6, 7, 8]

        # only ICON or ICON-Keypoint use visibility

        if self.prior_type in ["icon", "keypoint"]:
            if "image" in self.in_geo:
                self.channels_filter = [
                    image_lst + normal_F_lst,
                    image_lst + normal_B_lst,
                ]
            else:
                self.channels_filter = [normal_F_lst, normal_B_lst]

        else:
            if "image" in self.in_geo:
                self.channels_filter = [
                    image_lst + normal_F_lst + normal_B_lst
                ]
            else:
                self.channels_filter = [normal_F_lst + normal_B_lst]

        use_vis = (self.prior_type in ["icon", "keypoint"
                                       ]) and ("vis" in self.smpl_feats)

        if self.use_filter:
            channels_IF[0] = (self.hourglass_dim) * (2 - use_vis)
        else:
            channels_IF[0] = len(self.channels_filter[0]) * (2 - use_vis)

        if self.prior_type in ["icon", "keypoint"]:
            channels_IF[0] += self.smpl_dim
        else:
            print(f"don't support {self.prior_type}!")

        self.base_keys = ["smpl_verts", "smpl_faces"]

        self.icon_keys = self.base_keys + [
            f"smpl_{feat_name}" for feat_name in self.smpl_feats
        ]
        self.keypoint_keys = self.base_keys + [
            f"smpl_{feat_name}" for feat_name in self.smpl_feats
        ]

     


     
        norm_type = get_norm_layer(norm_type=self.opt.norm_color)
   
        self.image_filter=ViTVQ(image_size=512,channels=9)
        self.transformer=TransformerEncoderLayer(skips=4,multires=6,opt=self.opt)
        self.color_loss=nn.L1Loss()
        self.sp_encoder = SpatialEncoder()
        self.step=0
        self.features_costume=None

        # network
        if self.use_filter:
            if self.opt.gtype == "HGPIFuNet":
                self.F_filter = HGFilter(self.opt, self.opt.num_stack,
                                         len(self.channels_filter[0]))
                self.refine_filter = FuseHGFilter(self.opt, self.opt.num_stack,
                                                len(self.channels_filter[0]))
                
            else:
                print(
                    colored(f"Backbone {self.opt.gtype} is unimplemented",
                            "green"))

        summary_log = (f"{self.prior_type.upper()}:\n" +
                       f"w/ Global Image Encoder: {self.use_filter}\n" +
                       f"Image Features used by MLP: {self.in_geo}\n")

        if self.prior_type == "icon":
            summary_log += f"Geometry Features used by MLP: {self.smpl_feats}\n"
            summary_log += f"Dim of Image Features (local): {3 if (use_vis and not self.use_filter) else 6}\n"
            summary_log += f"Dim of Geometry Features (ICON): {self.smpl_dim}\n"
        elif self.prior_type == "keypoint":
            summary_log += f"Geometry Features used by MLP: {self.smpl_feats}\n"
            summary_log += f"Dim of Image Features (local): {3 if (use_vis and not self.use_filter) else 6}\n"
            summary_log += f"Dim of Geometry Features (Keypoint): {self.smpl_dim}\n"
        elif self.prior_type == "pamir":
            summary_log += f"Dim of Image Features (global): {self.hourglass_dim}\n"
            summary_log += f"Dim of Geometry Features (PaMIR): {self.voxel_dim}\n"
        else:
            summary_log += f"Dim of Image Features (global): {self.hourglass_dim}\n"
            summary_log += f"Dim of Geometry Features (PIFu): 1 (z-value)\n"

        summary_log += f"Dim of MLP's first layer: {channels_IF[0]}\n"

        print(colored(summary_log, "yellow"))

        self.normal_filter = NormalNet(cfg)

        init_net(self, init_type="normal")

    def get_normal(self, in_tensor_dict):

        # insert normal features
        if (not self.training) and (not self.overfit):
            # print(colored("infer normal","blue"))
            with torch.no_grad():
                feat_lst = []
                if "image" in self.in_geo:
                    feat_lst.append(
                        in_tensor_dict["image"])  # [1, 3, 512, 512]
                if "normal_F" in self.in_geo and "normal_B" in self.in_geo:
                    if ("normal_F" not in in_tensor_dict.keys()
                            or "normal_B" not in in_tensor_dict.keys()):
                        (nmlF, nmlB) = self.normal_filter(in_tensor_dict)
                    else:
                        nmlF = in_tensor_dict["normal_F"]
                        nmlB = in_tensor_dict["normal_B"]
                    feat_lst.append(nmlF)  # [1, 3, 512, 512]
                    feat_lst.append(nmlB)  # [1, 3, 512, 512]
            in_filter = torch.cat(feat_lst, dim=1)

        else:
            in_filter = torch.cat([in_tensor_dict[key] for key in self.in_geo],
                                  dim=1)

        return in_filter

    def get_mask(self, in_filter, size=128):

        mask = (F.interpolate(
            in_filter[:, self.channels_filter[0]],
            size=(size, size),
            mode="bilinear",
            align_corners=True,
        ).abs().sum(dim=1, keepdim=True) != 0.0)

        return mask

    def costume_filter(self, in_tensor_dict, return_inter=False):
        
        in_filter = self.get_normal(in_tensor_dict)
        image= in_tensor_dict["image"]
        fuse_image=torch.cat([image,in_filter], dim=1)
        
        features_G = []

        if self.prior_type in ["icon", "keypoint"]:
            if self.use_filter:
                triplane_features = self.image_filter(fuse_image)
                
                features_F = self.F_filter(in_filter[:,
                                                     self.channels_filter[0]]
                                           )  # [(B,hg_dim,128,128) * 4]
                features_B = self.F_filter(in_filter[:,
                                                     self.channels_filter[1]]
                                           )  # [(B,hg_dim,128,128) * 4]
            else:
                assert 0

            triplane_dim=32
            xy_plane_feat=triplane_features[:,:triplane_dim,:,:]
            yz_plane_feat=triplane_features[:,triplane_dim:2*triplane_dim,:,:]
            xz_plane_feat=triplane_features[:,2*triplane_dim:3*triplane_dim,:,:]

            refine_xy_plane_feat=self.refine_filter(image,xy_plane_feat)
            features_G.append(refine_xy_plane_feat)
            features_G.append(yz_plane_feat)
            features_G.append(xz_plane_feat)
            features_G.append(torch.cat([features_F[-1],features_B[-1]], dim=1))


        self.cos_smpl_feat_dict = {
            k: in_tensor_dict[k] if k in in_tensor_dict.keys() else None
            for k in getattr(self, f"{self.prior_type}_keys")
        }
            
        self.features_costume = features_G
        self.costume_smpl_verts = self.cos_smpl_feat_dict["smpl_verts"].permute(0,2,1)
        upbody_part_index=np.load("./data/up_part_body.npy",allow_pickle=True)
        lowerbody_part_index=np.load("./data/lower_part_body.npy",allow_pickle=True)
        body_part_index=np.concatenate([upbody_part_index,lowerbody_part_index],axis=0)
        self.upbody_part_index=torch.LongTensor(body_part_index).to(image.device)
        
        # If it is not in training, only produce the last im_feat
        if not self.training:
            features_out = features_G
        else:
            features_out = features_G

        if return_inter:
            return features_out, in_filter
        else:
            return features_out


    def filter(self, in_tensor_dict, return_inter=False):
        """
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        """

        in_filter = self.get_normal(in_tensor_dict)
        image= in_tensor_dict["image"]
        fuse_image=torch.cat([image,in_filter], dim=1)
        
        features_G = []

        if self.prior_type in ["icon", "keypoint"]:
            if self.use_filter:
                triplane_features, seq = self.image_filter(fuse_image)
                
                features_F = self.F_filter(in_filter[:,
                                                     self.channels_filter[0]]
                                           )  # [(B,hg_dim,128,128) * 4]
                features_B = self.F_filter(in_filter[:,
                                                     self.channels_filter[1]]
                                           )  # [(B,hg_dim,128,128) * 4]
            else:
                assert 0

            xy_plane_feat,yz_plane_feat,xz_plane_feat=triplane_features
            xy_seq,yz_seq,xz_seq=seq

            refine_xy_plane_feat=self.refine_filter(image,xy_plane_feat)
            features_G.append(refine_xy_plane_feat)
            features_G.append(yz_plane_feat)
            features_G.append(xz_plane_feat)
            features_G.append(torch.cat([features_F[-1],features_B[-1]], dim=1))

           
        else:
            assert 0

        self.smpl_feat_dict = {
            k: in_tensor_dict[k] if k in in_tensor_dict.keys() else None
            for k in getattr(self, f"{self.prior_type}_keys")
        }
        if 'animated_smpl_verts' not in in_tensor_dict.keys():
            self.point_feat_extractor = PointFeat(self.smpl_feat_dict["smpl_verts"],
                                               self.smpl_feat_dict["smpl_faces"])
        else:
            self.smpl_feat_dict.update({'animated_smpl_verts':in_tensor_dict['animated_smpl_verts']})
            self.point_feat_extractor = PointFeat(self.smpl_feat_dict["animated_smpl_verts"],
                                               self.smpl_feat_dict["smpl_faces"])
            
        self.features_G = features_G
        
        # If it is not in training, only produce the last im_feat
        if not self.training:
            features_out = features_G
        else:
            features_out = features_G

        if return_inter:
            return features_out, in_filter
        else:
            return features_out
        
        
    def update_SMPL(self, in_tensor_dict):
        if 'animated_smpl_verts' in in_tensor_dict.keys():
            self.smpl_feat_dict.update({'animated_smpl_verts':in_tensor_dict['animated_smpl_verts']})
            animated_smpl_normal=compute_normal_batch(in_tensor_dict['animated_smpl_verts'],in_tensor_dict['smpl_faces'])
            self.smpl_feat_dict.update({'animated_smpl_norm':animated_smpl_normal})

            self.point_feat_extractor = PointFeat(self.smpl_feat_dict["animated_smpl_verts"],
                                               self.smpl_feat_dict["smpl_faces"])
        
    

    def query(self, features, points, calibs, transforms=None,type='shape'):

        xyz = self.projection(points, calibs, transforms) # project to image plane
        if self.training and self.query_point is None:
            self.query_point=xyz

        (xy, z) = xyz.split([2, 1], dim=1)
        (x,yz) = xyz.split([1, 2], dim=1)
        xz = torch.cat([xyz[:,0:1],xyz[:,2:3]],dim=1)

        in_cube = (xyz > -1.0) & (xyz < 1.0)
        in_cube = in_cube.all(dim=1, keepdim=True).detach().float()

        preds_list = []
        vol_feats = features

        if self.prior_type in ["icon", "keypoint"]:

            
            
            densely_smpl=self.smpl_feat_dict['smpl_verts'].permute(0,2,1)
            #smpl_origin=self.projection(densely_smpl, torch.inverse(calibs), transforms)
            smpl_vis=self.smpl_feat_dict['smpl_vis'].permute(0,2,1)
            #verts_ids=self.smpl_feat_dict['smpl_sample_id']

            

            (smpl_xy,smpl_z)=densely_smpl.split([2,1],dim=1)
            (smpl_x,smpl_yz)=densely_smpl.split([1,2],dim=1)
            smpl_xz=torch.cat([densely_smpl[:,0:1],densely_smpl[:,2:3]],dim=1)
                                
            point_feat_out = self.point_feat_extractor.query(  # this extractor changes if has animated smpl
                xyz.permute(0, 2, 1).contiguous(), self.smpl_feat_dict)
            vis=point_feat_out['vis'].permute(0,2,1)
            #sdf_body=-point_feat_out['sdf']    # this sdf needs to be multiplied by -1
            feat_lst = [
                point_feat_out[key] for key in self.smpl_feats
                if key in point_feat_out.keys()
            ]
            smpl_feat = torch.cat(feat_lst, dim=2).permute(0, 2, 1)

        if len(features)==4: 
            
            xy_plane_feat1,xy_plane_feat2=features[0].chunk(2,dim=1)
            yz_plane_feat1,yz_plane_feat2=features[1].chunk(2,dim=1)
            xz_plane_feat1,xz_plane_feat2=features[2].chunk(2,dim=1)
            in_feat=features[3]
            
            ### point query ###
            if "animated_smpl_verts" in self.smpl_feat_dict.keys():
                # change the xyz according to the deformed smpl and get features
               
                
                
                de_nn_points, de_nn_idx=self.point_feat_extractor.get_nearest_point(xyz.permute(0,2,1).contiguous())
                de_points_normal=self.smpl_feat_dict['animated_smpl_norm'][:,de_nn_idx,:] # BxNxC
                ori_nn_points=self.smpl_feat_dict['smpl_verts'][:,de_nn_idx,:] # BxNxC
                ori_points_normal=self.smpl_feat_dict['smpl_norm'][:,de_nn_idx,:] # BxNxC
                # convert to cpu to avoid cuda out of memory 
                de_xyz = surface_field_deformation(xyz.permute(0,2,1).contiguous(),
                                                   de_nn_points,de_points_normal,
                                                   ori_nn_points,ori_points_normal).permute(0,2,1)
                
                #de_xyz=de_xyz.to(xyz.device)

                (de_xy, de_z) = de_xyz.split([2, 1], dim=1)
                (de_x,de_yz) = de_xyz.split([1, 2], dim=1)
                de_xz = torch.cat([de_xyz[:,0:1],de_xyz[:,2:3]],dim=1)
                xy_feat=self.index(xy_plane_feat1,de_xy)
                yz_feat=self.index(yz_plane_feat1,de_yz)
                xz_feat=self.index(xz_plane_feat1,de_xz)
                normal_feat=feat_select(self.index(in_feat, de_xy),vis)  

            else:
                xy_feat=self.index(xy_plane_feat1,xy)
                yz_feat=self.index(yz_plane_feat1,yz)
                xz_feat=self.index(xz_plane_feat1,xz)
                normal_feat=feat_select(self.index(in_feat, xy),vis)
            two_plane_feat=(yz_feat+xz_feat)/2
            triplane_feat=torch.cat([xy_feat,two_plane_feat],dim=1)        # 32+32=64

            ### smpl query ###
            smpl_xy_feat=self.index(xy_plane_feat2,smpl_xy)
            smpl_yz_feat=self.index(yz_plane_feat2,smpl_yz)
            smpl_xz_feat=self.index(xz_plane_feat2,smpl_xz)
            smpl_two_plane_feat=(smpl_yz_feat+smpl_xz_feat)/2
            smpl_triplane_feat=torch.cat([smpl_xy_feat,smpl_two_plane_feat],dim=1)        # 32+32=64
            bary_centric_feat=self.point_feat_extractor.query_barycentirc_feats(xyz.permute(0,2,1).contiguous()
                                                                      ,smpl_triplane_feat.permute(0,2,1))




            
            final_feat=torch.cat([triplane_feat,bary_centric_feat.permute(0,2,1),normal_feat],dim=1)  # 64+64+6=134



            if self.features_costume is not None:
                cos_smpl_vis=self.cos_smpl_feat_dict['smpl_vis'].permute(0,2,1)

                cos_xy_plane=self.features_costume[0]
                cos_yz_plane=self.features_costume[1]
                cos_xz_plane=self.features_costume[2]

                cos_in_feat=self.features_costume[3]
                
                (costume_xy,costume_z)=self.costume_smpl_verts.split([2,1],dim=1)
                (costume_x,costume_yz)=self.costume_smpl_verts.split([1,2],dim=1)
                costume_xz=torch.cat([self.costume_smpl_verts[:,0:1],self.costume_smpl_verts[:,2:3]],dim=1)
                cos_xy_feat=self.index(cos_xy_plane,costume_xy)
                cos_yz_feat=self.index(cos_yz_plane,costume_yz)
                cos_xz_feat=self.index(cos_xz_plane,costume_xz)
                cos_normal_feat=feat_select(self.index(cos_in_feat, costume_xy),cos_smpl_vis)
                cos_triplane_feat=torch.cat([cos_xy_feat,cos_yz_feat,cos_xz_feat],dim=1)        # 32+32+128=192
                costume_final_feat=torch.cat([cos_triplane_feat,cos_normal_feat],dim=1)  # 192+6
                
                final_feat=final_feat.permute(0,2,1)
                # exchange features
                final_feat[:,self.upbody_part_index]=costume_final_feat.permute(0,2,1)[:,self.upbody_part_index]
                final_feat=final_feat.permute(0,2,1)


            if type=='shape':
                if 'animated_smpl_verts' in self.smpl_feat_dict.keys():
                    animated_smpl=self.smpl_feat_dict['animated_smpl_verts']
                    
                    occ=self.transformer(xyz.permute(0,2,1).contiguous(),animated_smpl,
                                                        final_feat,smpl_feat,training=self.training,type=type)
                else:
                    
                    occ=self.transformer(xyz.permute(0,2,1).contiguous(),densely_smpl.permute(0,2,1),
                                                        final_feat,smpl_feat,training=self.training,type=type)
              
                occ=occ*in_cube
                preds_list.append(occ)   

            elif type=='color':
                if 'animated_smpl_verts' in self.smpl_feat_dict.keys():
                    animated_smpl=self.smpl_feat_dict['animated_smpl_verts']
                    color_preds=self.transformer(xyz.permute(0,2,1).contiguous(),animated_smpl,
                                                        final_feat,smpl_feat,training=self.training,type=type)
                    
                
                else:
                    color_preds=self.transformer(xyz.permute(0,2,1).contiguous(),densely_smpl.permute(0,2,1),
                                                        final_feat,smpl_feat,training=self.training,type=type)
                preds_list.append(color_preds)  


        return preds_list




    def get_error(self, preds_if_list, labels):
        """calculate error

        Args:
            preds_list (list): list of torch.tensor(B, 3, N)
            labels (torch.tensor): (B, N_knn, N)

        Returns:
            torch.tensor: error
        """
        error_if = 0

        for pred_id in range(len(preds_if_list)):
            pred_if = preds_if_list[pred_id]
            error_if += F.binary_cross_entropy(pred_if, labels)

        error_if /= len(preds_if_list)

        return error_if

    def volume_rendering(self,pts, z_vals, rays_d, in_feat, calib_tensor):
        n_batch, n_pixel, n_sample = z_vals.shape
        pts=pts.reshape(n_batch,n_pixel*n_sample,3).permute(0,2,1).contiguous()
        raw=self.query(in_feat,pts,calib_tensor,type='shape_color')  # B N*S 4
        raw=raw.reshape(n_batch,n_pixel,n_sample,4).reshape(-1,n_sample,4)
        rays_d=rays_d.reshape(-1,3)
        z_vals=z_vals.reshape(-1,n_sample)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(  # integrate along the ray
            raw, z_vals, rays_d, white_bkgd = False)

        rgb_map = rgb_map.view(n_batch, n_pixel, -1)

        return rgb_map



    def forward(self, in_tensor_dict):
        """
        sample_tensor [B, 3, N]
        calib_tensor [B, 4, 4]
        label_tensor [B, 1, N]
        smpl_feat_tensor [B, 59, N]
        """
        

        sample_tensor = in_tensor_dict["sample"]
        calib_tensor = in_tensor_dict["calib"]
        label_tensor = in_tensor_dict["label"]
        # eikonal_points=in_tensor_dict["eikonal_points"].requires_grad_()
        color_sample=in_tensor_dict["sample_color"]
        color_label=in_tensor_dict["color"]
        # nerf_dict=in_tensor_dict['nerf_dict']
        # rays_d=nerf_dict['rays_d']
        self.sdf_preds=None
        self.query_point=None

        in_feat = self.filter(in_tensor_dict)
       
    
        preds_if_list = self.query(in_feat,
                                   sample_tensor,
                                   calib_tensor,type='shape')

        BCEloss = self.get_error(preds_if_list, label_tensor)

        color_preds=self.query(in_feat,
                               color_sample,
                               calib_tensor,type='color')
        color_loss=self.color_loss(color_preds[0],color_label)


        if self.training:
          
            self.color3d_loss= color_loss # torch.tensor(0.).float().to(BCEloss.device)
            # self.rgb_loss=rgb_loss
            error=BCEloss+color_loss
            self.grad_loss=torch.tensor(0.).float().to(BCEloss.device)
        else:
            error=BCEloss

        return preds_if_list[-1].detach(), error
