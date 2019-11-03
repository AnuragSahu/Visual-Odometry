import os
import glob
from cv2 import *
import numpy as np
from pylab import *
import math
from decimal import Decimal

def denormalize_matrix(p1,p2):
    E,mask = findEssentialMat(p1,p2,K,cv2.RANSAC,0.999,1.0)
    return np.linalg.inv(K.T).dot(E.dot(np.linalg.inv(K)))

K = np.array([[7.215377000000e+02,0.000000000000e+00,6.095593000000e+02],
              [0.000000000000e+00,7.215377000000e+02,1.728540000000e+02],
              [0.000000000000e+00,0.000000000000e+00,1.000000000000e+00]])

class LoadDataset():
    def __init__(self):
        #pass
        path = "../q2/images"
        self.image_format_left = '{:06d}.png'
        self.path = os.path.join(path)
        sequence_count = os.path.dirname(self.path).split('/')[-1]
        gt_path = os.path.join(self.path, '..','ground-truth.txt')
        self.count_image()
        self.original_value = self.load_ground_truth_pose(gt_path)

    def image_path_left(self, index):
        return os.path.join(self.path, self.image_format_left).format(index)

    def count_image(self):
        extension = os.path.splitext(self.image_format_left)[-1]
        wildcard = os.path.join(self.path, '*' + extension)
        self.image_count = len(glob.glob(wildcard))

    def load_ground_truth_pose(self, gt_path):
        original_value = []
        with open(gt_path) as gt_file:
            gt_lines = gt_file.readlines()

            for gt_line in gt_lines:
                pose = self.convert_text_to_ground_truth(gt_line)
                original_value.append(pose)
        return original_value

    def convert_text_to_ground_truth(self, gt_line):
        matrix = np.array(gt_line.split()).reshape((3, 4)).astype(np.float32)
        return matrix


def main():
    dataset = LoadDataset()

    # Function for Detection of the Features
    feature_detector = cv2.xfeatures2d.SIFT_create()

    current_pos = np.zeros((3, 1))
    current_rot = np.eye(3)

    file = open("results.txt","w+")
    print("{} images found.".format(dataset.image_count))

    prev_image = None
    camera_matrix = K
    valid_ground_truth = True

    for index in range(dataset.image_count):
        # load image
        image = cv2.imread(dataset.image_path_left(index))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # main process
        keypoint = feature_detector.detect(image, None)

        #for first case
        if prev_image is None:
            prev_image = image
            prev_keypoint = keypoint
            continue

        points = np.array(list(map(lambda x: [x.pt], prev_keypoint)),dtype=np.float32)
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_image, image, points,None, )
        p1_tmp = []
        points_tmp = []
        for a in p1:
            p1_tmp.append(a[0])
        p1 = np.array(p1_tmp)
        for a in points:
            points_tmp.append(a[0])
        points = np.array(points_tmp)
        F = denormalize_matrix(p1, points)
        E = K.T.dot(F.dot(K))
        points, R, t, mask = cv2.recoverPose(E, p1, points, camera_matrix)
        scale = 1.0

        # calc scale from ground truth if exists.

        original_value = dataset.original_value[index]
        original_position = [original_value[0, 3], original_value[2, 3]]
        previous_ground_truth = dataset.original_value[index - 1]
        previous_ground_truth_pos = [
            previous_ground_truth[0, 3],
            previous_ground_truth[2, 3]]
        scale = math.sqrt(math.pow((original_position[0] - previous_ground_truth_pos[0]), 2.0) + math.pow((original_position[1] - previous_ground_truth_pos[1]), 2.0))

        current_pos += current_rot.dot(t) * scale
        current_rot = R.dot(current_rot)

        transformation_matrix = np.zeros((3,4))
        transformation_matrix[:,:-1] = current_rot
        transformation_matrix[:,-1] = current_pos.T

        transformation_matrix = transformation_matrix.reshape(1,12)
        count = 0
        for vals in transformation_matrix[0]:
            count+=1
            file.write(str('%.6e'%Decimal(vals)))
            if(count<12):
                file.write(" ")
        file.write("\n")
        if(index == dataset.image_count-1):
            count = 0
            for vals in transformation_matrix[0]:
                count+=1
                file.write(str('%.6e'%Decimal(vals)))
                if(count<12):
                    file.write(" ")
            file.write("\n")


        # get ground truth if eist.
            original_value = dataset.original_value[index]

        # calc rotation error with ground truth.
        original_value = dataset.original_value[index]
        original_rotation = original_value[0: 3, 0: 3]
        r_vec, _ = cv2.Rodrigues(current_rot.dot(original_rotation.T))
        rotation_error = np.linalg.norm(r_vec)

        prev_image = image
        prev_keypoint = keypoint
        print((index/dataset.image_count)*100,"%")

    file.close()
    os.system("evo_traj kitti ground-truth.txt --ref results.txt -va --plot --plot_mode xz")

if __name__ == "__main__":
    main()
