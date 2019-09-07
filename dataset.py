import os
import glob
import numpy as np


class CameraParameters():
    def __init__(self):
        pass

    @property
    def camera_matrix(self):
        matrix = K = np.array([[7.215377000000e+02,0.000000000000e+00,6.095593000000e+02],
                               [0.000000000000e+00,7.215377000000e+02,1.728540000000e+02],
                               [0.000000000000e+00,0.000000000000e+00,1.000000000000e+00]])
        return matrix

    def __call__(self):
        return self.camera_matrix


class Dataset():
    def __init__(self):
        pass

    def image_path_left(self, index):
        return os.path.join(self.path, self.image_format_left).format(index)

    def count_image(self):
        extension = os.path.splitext(self.image_format_left)[-1]
        wildcard = os.path.join(self.path, '*' + extension)
        self.image_count = len(glob.glob(wildcard))

    def load_ground_truth_pose(self, gt_path):
        ground_truth = None
        if not os.path.exists(gt_path):
            print(gt_path,"ground truth path is not found.")
            return None

        ground_truth = []

        with open(gt_path) as gt_file:
            gt_lines = gt_file.readlines()

            for gt_line in gt_lines:
                pose = self.convert_text_to_ground_truth(gt_line)
                ground_truth.append(pose)
        return ground_truth

    def convert_text_to_ground_truth(self, gt_line):
        pass


class KittiDataset(Dataset):
    def __init__(self, path):
        self.image_format_left = '{:06d}.png'
        #self.path = os.path.join(path, 'image_0')
        self.path = os.path.join(path)
        self.calibfile = os.path.join(path, '../calib.txt')
        sequence_count = os.path.dirname(self.path).split('/')[-1]

        gt_path = os.path.join(self.path, '..','ground-truth.txt')

        self.count_image()
        self.ground_truth = self.load_ground_truth_pose(gt_path)
        self.camera_matrix = self.load_camera_parameters(self.calibfile)

    def convert_text_to_ground_truth(self, gt_line):
        matrix = np.array(gt_line.split()).reshape((3, 4)).astype(np.float32)
        return matrix

    def load_camera_parameters(self, calibfile):
        if not os.path.exists(calibfile):
            print(calibfile,"camera parameter file path is not found.")
            return None

        with open(calibfile, 'r') as f:
            line = f.readline()
            part = line.split()
            param = CameraParameters()
            return param


dataset_dict = {'kitti': KittiDataset}


def create_dataset(options):
    return dataset_dict[options.dataset](options.path)
