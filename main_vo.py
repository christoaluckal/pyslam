#!/usr/bin/env -S python3 -O
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

import numpy as np
import cv2
import math
import time 
import platform 

from config import Config

from visual_odometry import VisualOdometry
from camera  import PinholeCamera
from ground_truth import groundtruth_factory
from dataset import dataset_factory

#from mplot3d import Mplot3d
#from mplot2d import Mplot2d
from mplot_thread import Mplot2d, Mplot3d

from feature_tracker import feature_tracker_factory, FeatureTracker 
from feature_manager import feature_manager_factory
from feature_types import FeatureDetectorTypes, FeatureDescriptorTypes, FeatureInfo
from feature_matcher import FeatureMatcherTypes

from feature_tracker_configs import FeatureTrackerConfigs

# from rerun_interface import Rerun
import matplotlib.pyplot as plt
plt.set_loglevel('error')


# kUseRerun = True
# # check rerun does not have issues 
# if kUseRerun and not Rerun.is_ok():
#     kUseRerun = False
    
"""
use or not pangolin (if you want to use it then you need to install it by using the script install_thirdparty.sh)
"""
kUsePangolin = False  
if platform.system() == 'Darwin':
    kUsePangolin = True # Under mac force pangolin to be used since Mplot3d() has some reliability issues
                
if kUsePangolin:
    from viewer3D import Viewer3D

import os
from tqdm import tqdm
if __name__ == "__main__":

    f_start = 0
    f_end = 10
    num = f_end - f_start


    # mask_loc = '/mnt/share/nas/Projects/Featurability/masks/kitti_00_npy_masks/0.0_0.1'
    # masks_temp = sorted(os.listdir(mask_loc), key=lambda x: int(x.split('.')[0]))[f_start:f_end]
    # masks = [np.load(os.path.join(mask_loc, mask)) for mask in masks_temp]

    # mask_loc = '/mnt/share/nas/Projects/Featurability/masks/coda_00_masks/0.0_0.15'
    # masks_temp = sorted(os.listdir(mask_loc), key=lambda x: int(x.split('.')[0]))[f_start:f_end]
    # masks = []
    # for mask in tqdm(masks_temp):
    #     mask_img = cv2.imread(os.path.join(mask_loc, mask), cv2.IMREAD_GRAYSCALE)
    #     # convert to boolean mask
    #     mask_img = mask_img > 0
    #     masks.append(mask_img)

    # mask_loc = '/mnt/share/nas/Projects/Featurability/masks/day2_davis_run1_masks/0.0_0.15'
    
    # masks_temp = sorted([x for x in os.listdir(mask_loc) if x.endswith('.png')], key=lambda x: int(x.split('.')[0]))[f_start:f_end]
    # masks = []
    # for mask in tqdm(masks_temp):
    #     mask_img = cv2.imread(os.path.join(mask_loc, mask), cv2.IMREAD_GRAYSCALE)
        
    #     # convert to boolean mask
    #     mask_img = mask_img > 0
    #     masks.append(mask_img)
    

    


    # select your tracker configuration (see the file feature_tracker_configs.py) 
    # LK_SHI_TOMASI, LK_FAST
    # SHI_TOMASI_ORB, FAST_ORB, ORB, BRISK, AKAZE, FAST_FREAK, SIFT, ROOT_SIFT, SURF, SUPERPOINT, FAST_TFEAT, LIGHTGLUE, XFEAT, XFEAT_XFEAT, LOFTR
    features = {
        # "LK_SHI_TOMASI": FeatureTrackerConfigs.LK_SHI_TOMASI,
        # "FAST_ORB": FeatureTrackerConfigs.FAST_ORB,
        # "ORB": FeatureTrackerConfigs.ORB,
        # # # "KAZE": FeatureTrackerConfigs.KAZE,
        # "AKAZE": FeatureTrackerConfigs.AKAZE,
        # "BRISK": FeatureTrackerConfigs.BRISK,
        # "SIFT": FeatureTrackerConfigs.SIFT,

        # "SURF": FeatureTrackerConfigs.SURF,
        # "SUPERPOINT": FeatureTrackerConfigs.SUPERPOINT,
        # # "LIGHTGLUE": FeatureTrackerConfigs.LIGHTGLUE,

        # "LOFTR": FeatureTrackerConfigs.LOFTR
        "SILK": FeatureTrackerConfigs.SILK
    }
    for n_features in [1000]:
    # for n_features in [1000]:
        for feat in features.keys():
            for exp_type in ['KITTI']:
            # for exp_type in ['KITTI','KITTI_masked']:
            # for exp_type in ['CODA','CODA_masked']:
                time.sleep(2)
                # try:
                if exp_type == 'KITTI':
                    config = Config(cfg_file='config_kitti.ini')
                elif exp_type == 'KITTI_masked':
                    config = Config(cfg_file='config_kitti_masked.ini')
                elif exp_type == 'CODA' or exp_type == 'CODA_masked':
                    config = Config(cfg_file='config_coda.ini')
                elif exp_type == 'DAVIS' or exp_type == 'DAVIS_masked':
                    config = Config(cfg_file='config_davis.ini')
                else:
                    raise ValueError(f"Invalid exp_type: {exp_type}")

        
                dataset = dataset_factory(config.dataset_settings)

                groundtruth = groundtruth_factory(config.dataset_settings,frame_start=f_start,frame_end=f_end)

                cam = PinholeCamera(config.cam_settings['Camera.width'], config.cam_settings['Camera.height'],
                                    config.cam_settings['Camera.fx'], config.cam_settings['Camera.fy'],
                                    config.cam_settings['Camera.cx'], config.cam_settings['Camera.cy'],
                                    config.DistCoef, config.cam_settings['Camera.fps'])


                num_features=n_features  # how many features do you want to detect and track?

                tracker_config = features[feat]
                tracker_config['num_features'] = num_features
            
                feature_tracker = feature_tracker_factory(**tracker_config)

                # create visual odometry object 
                vo = VisualOdometry(cam, groundtruth, feature_tracker,outlier_removal=True)

                is_draw_traj_img = True
                traj_img_size = 800
                traj_img = np.zeros((traj_img_size, traj_img_size, 3), dtype=np.uint8)
                half_traj_img_size = int(0.5*traj_img_size)
                draw_scale = 1

                make_plots = False

                is_draw_3d = True
                
                is_draw_with_rerun = False
                if is_draw_with_rerun:
                    Rerun.init_vo()
                else: 
                    if make_plots:
                        if kUsePangolin:
                            viewer3D = Viewer3D()
                        else:
                            plt3d = Mplot3d(title='3D trajectory')

                is_draw_err = True 
                err_plt = Mplot2d(xlabel='img id', ylabel='m',title='error')
                
                is_draw_matched_points = True 
                matched_points_plt = Mplot2d(xlabel='img id', ylabel='# matches',title='# matches')

                
                img_id = 0
                rows = []
                vo_start = time.time()
                while dataset.isOk():
                    if img_id == num-1:
                        break
                    img = dataset.getImage(img_id)

                    if img is not None:
                        start = time.time()

                        if exp_type == 'KITTI' or exp_type == 'CODA' or exp_type == 'DAVIS':
                            kp,ins,outs,pxs=vo.track(img, img_id)
                        elif exp_type == 'KITTI_masked' or exp_type == 'CODA_masked' or exp_type == 'DAVIS_masked':
                            mask = masks[img_id]
                            kp,ins,outs,pxs=vo.track(img, img_id,mask)  # main VO function 
                        end = time.time()

                        if(img_id > 2):	       # start drawing from the third image (when everything is initialized and flows in a normal way)

                            x, y, z = vo.traj3d_est[-1]
                            x_true, y_true, z_true = vo.traj3d_gt[-1]
                            row = [img_id, x.item(), y.item(), z.item(), x_true, y_true, z_true, end - start]
                            
                            if is_draw_traj_img:      # draw 2D trajectory (on the plane xz)
                                draw_x, draw_y = int(draw_scale*x) + half_traj_img_size, half_traj_img_size - int(draw_scale*z)
                                true_x, true_y = int(draw_scale*x_true) + half_traj_img_size, half_traj_img_size - int(draw_scale*z_true)
                                cv2.circle(traj_img, (draw_x, draw_y), 1,(img_id*255/4540, 255-img_id*255/4540, 0), 1)   # estimated from green to blue
                                cv2.circle(traj_img, (true_x, true_y), 1,(0, 0, 255), 1)  # groundtruth in red
                                # write text on traj_img
                                cv2.rectangle(traj_img, (10, 20), (600, 60), (0, 0, 0), -1)
                                text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z)
                                cv2.putText(traj_img, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
                                # show 		

                                if make_plots:
                                    if is_draw_with_rerun:
                                        Rerun.log_img_seq('trajectory_img/2d', img_id, traj_img)
                                    else:
                                        cv2.imshow('Trajectory', traj_img)


                            if is_draw_with_rerun:   
                                if make_plots:                                     
                                    Rerun.log_2d_seq_scalar('trajectory_error/err_x', img_id, math.fabs(x_true-x))
                                    Rerun.log_2d_seq_scalar('trajectory_error/err_y', img_id, math.fabs(y_true-y))
                                    Rerun.log_2d_seq_scalar('trajectory_error/err_z', img_id, math.fabs(z_true-z))
                                    
                                    Rerun.log_2d_seq_scalar('trajectory_stats/num_matches', img_id, vo.num_matched_kps)
                                    Rerun.log_2d_seq_scalar('trajectory_stats/num_inliers', img_id, vo.num_inliers)
                                    Rerun.log_3d_camera_img_seq(img_id, vo.draw_img, cam, vo.poses[-1])
                                    Rerun.log_3d_trajectory(img_id, vo.traj3d_est, 'estimated', color=[0,0,255])
                                    Rerun.log_3d_trajectory(img_id, vo.traj3d_gt, 'ground_truth', color=[255,0,0])     
                            else:
                                if is_draw_3d:           # draw 3d trajectory 
                                    if kUsePangolin:
                                        viewer3D.draw_vo(vo)   
                                    else:
                                        if make_plots:
                                            plt3d.drawTraj(vo.traj3d_gt,'ground truth',color='r',marker='.')
                                            plt3d.drawTraj(vo.traj3d_est,'estimated',color='g',marker='.')
                                            plt3d.refresh()

                                if is_draw_err:         # draw error signals 
                                    if make_plots:
                                        errx = [img_id, math.fabs(x_true-x)]
                                        erry = [img_id, math.fabs(y_true-y)]
                                        errz = [img_id, math.fabs(z_true-z)] 
                                        err_plt.draw(errx,'err_x',color='g')
                                        err_plt.draw(erry,'err_y',color='b')
                                        err_plt.draw(errz,'err_z',color='r')
                                        err_plt.refresh()    

                                if is_draw_matched_points:
                                    matched_kps_signal = [img_id, vo.num_matched_kps]
                                    inliers_signal = [img_id, vo.num_inliers]  
                                    if make_plots:                  
                                        matched_points_plt.draw(matched_kps_signal,'# matches',color='b')
                                        matched_points_plt.draw(inliers_signal,'# inliers',color='g')                    
                                        matched_points_plt.refresh()     

                            row.append(vo.num_matched_kps)
                            row.append(vo.num_inliers)
                            row.append(kp if kp is not None else 0)
                            row.append(ins if ins is not None else 0)
                            row.append(outs if outs is not None else 0)
                            row.append(pxs if pxs is not None else 0)
                            rows.append(row)                                  
                                
                        # draw camera image 
                        if not is_draw_with_rerun:
                            if make_plots:
                                cv2.imshow('Camera', vo.draw_img)				

                    # press 'q' to exit!
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break

                    img_id += 1

                vo_end = time.time()
                rows.append([0, 0, 0, 0, 0, 0, 0, vo_end - vo_start, 0, 0, 0, 0, 0, 0])
                rows = np.array(rows)

                exp_name = f"{exp_type}_feature_{feat}_n_{n_features}"

                plt.plot(rows[:-1, 1], rows[:-1, 3], label='estimated')
                plt.plot(rows[:-1, 4], rows[:-1, 6], label='gt')
                plt.legend()
                plt.gca().set_aspect('equal', adjustable='box')
                plt.title(f"{exp_name} 2D trajectory | Mean Time: {np.mean(rows[:, 7]):.5f} s")
                plt.savefig(f"{exp_name}_traj", dpi=300)
                plt.close()

                idxs = np.arange(0, rows.shape[0]-1)
                plt.plot(idxs, rows[:-1, 8], label='matches')
                plt.plot(idxs, rows[:-1, 9], label='inliers')
                plt.legend()
                plt.title(f"{exp_name} Matches and Inliers")
                plt.savefig(f"{exp_name}_matches", dpi=300)
                plt.close()

                import pandas as pd
                df = pd.DataFrame(rows, columns=['img_id', 'x', 'y', 'z', 'x_true', 'y_true', 'z_true', 'time', 'matches', 'inliers', 'kps','ins','outs','px_shift'])
                df.to_csv(f"{exp_name}_data.csv", index=False)

                # except Exception as e:
                #     print(f"Error in {feat}: {e}")
                #     continue
                        
            # cv2.destroyAllWindows()
