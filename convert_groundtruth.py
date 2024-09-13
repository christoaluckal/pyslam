import sys
import numpy as np

sys.path.append("../")

from ground_truth import groundtruth_factory,CODAGroundTruth

# convert kitti ground truth in a simple format which can be used with video datasets
groundtruth_settings = {}
groundtruth_settings['type']='video'
groundtruth_settings['base_path'] ='/home/belk/pyslam_data/davis/'
groundtruth_settings['name'] = 'groundtruth.txt'
groundtruth_settings['groundtruth_file'] = 'groundtruth.txt'

def main(settings = groundtruth_settings, out_filename = 'codagroundtruth.txt'):
    grountruth = groundtruth_factory(groundtruth_settings)
    grountruth.convertToSimpleXYZ(vo_f=False)

if __name__ == '__main__':
    main()