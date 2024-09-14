
"""
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com> 
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

import sys
import os
import cv2 
import torch
import time
import platform 
import config
config.cfg.set_lib('silk') 
from threading import RLock
from utils_sys import Printer
import os
from copy import deepcopy
import torch
from silk.backbones.silk.silk import SiLKVGG as SiLK
from silk.backbones.superpoint.vgg import ParametricVGG
from silk.config.model import load_model_from_checkpoint

kVerbose = True   
import numpy as np


class SiLKOptions:
    def __init__(self, do_cuda=True): 
        # default options from demo_superpoints
        # self.weights_path=config.cfg.root_folder + '/thirdparty/silk/silk_coda_dnn.pth'
        self.weights_path='/home/caluckal/Developer/fall24/feature/mypyslam/thirdparty/silk/silk_dnn_kitti11_epoch99.ckpt'
        print(f'SiLK weights: {self.weights_path}')
        self.nn_thresh=0.7
        use_cuda = torch.cuda.is_available() and do_cuda
        device = torch.device('cuda' if use_cuda else 'cpu')
        print('SiLK using ', device)        
        self.cuda=use_cuda     

# interface for pySLAM 
class SiLKFeature2D: 
    def __init__(self, num_features=1000, do_cuda=True): 
        if platform.system() == 'Darwin':        
            do_cuda=False
        self.lock = RLock()
        self.opts = SiLKOptions(do_cuda)
        print(self.opts)        
        
        print('SiLKFeature2D')
        print('==> Loading pre-trained network.')
        # This class runs the SuperPoint network and processes its outputs.
        self.fe = SiLK(
            in_channels=1,
            backbone=deepcopy(ParametricVGG(use_max_pooling=False,padding=0, normalization_fn=[torch.nn.BatchNorm2d(i) for i in (64, 64, 128, 128)],)),
            detection_threshold=1.0,
            detection_top_k=num_features,
            nms_dist=0,
            border_dist=0,
            default_outputs=("sparse_positions", "sparse_descriptors"),
            descriptor_scale_factor=1.41, # scaling of descriptor output, do not change
            padding=0,
        )
        self.fe = load_model_from_checkpoint(
            self.fe,
            checkpoint_path=self.opts.weights_path,
            state_dict_fn=lambda x: {k[len("_mods.model.") :]: v for k, v in x.items()},
            device='cuda' if do_cuda else 'cpu',
            freeze=True,
            eval=True,
        )
        
        print('==> Successfully loaded pre-trained network.')
                        
        self.pts = []
        self.kps = []        
        self.des = []
        self.frame = None 
        self.frameFloat = None 
        self.keypoint_size = 20  # just a representative size for visualization and in order to convert extracted points to cv2.KeyPoint 
          
    def silkload(self, x, as_gray=True):
        x = torch.tensor(x, device='cuda', dtype=torch.float32)
        if not as_gray:
            x = x.permute(0, 3, 1, 2)
            images = images / 255.0
        else:
            # stack x 3 times to make it 3 channel
            
            x = x.unsqueeze(1)  # add channel dimension
            print(f"Frame Shape before permute: {x.shape}")
            # x = x.permute(1,3,0,2)
            x = x.permute(3, 1, 0, 2)
            x = x / 255.0
        print(f"Frame Shape before passing to silk: {x.shape}")
        return x

    # compute both keypoints and descriptors       
    def detectAndCompute(self, frame, mask=None):  # mask is a fake input 
        with self.lock: 
            self.frame = frame 
            temp_frame = self.silkload(self.frame)
            self.frameFloat  = temp_frame
            self.kps, self.des = self.fe(self.frameFloat)
            og_kp = []

            

            for p in self.kps[0]:
                x_coord, y_coord, _ = float(p[0].cpu().numpy()), float(p[1].cpu().numpy()), float(p[2].cpu().numpy())
                kp = cv2.KeyPoint(y_coord, x_coord, 1)
                og_kp.append(kp)

            self.des = np.stack([tensor.cpu().numpy() for tensor in self.des[0]])
            self.kps = og_kp
            
            if kVerbose:
                print('detector: SILK, #features: ', len(self.kps), ', frame res: ', frame.shape[0:2])      
            return self.kps, self.des               
            
    # return keypoints if available otherwise call detectAndCompute()    
    def detect(self, frame, mask=None):  # mask is a fake input  
        with self.lock:         
            self.detectAndCompute(frame)        
            return self.kps
    
    # return descriptors if available otherwise call detectAndCompute()  
    def compute(self, frame, kps=None, mask=None): # kps is a fake input, mask is a fake input
        with self.lock: 
            if self.frame is not frame:
                Printer.orange('WARNING: SILK is recomputing both kps and des on last input frame', frame.shape)
                self.detectAndCompute(frame)
            return self.kps, self.des
           
