# Visual-Odometry

## About Visual Odometry
Visual Odometry is the process of recovering the trajectory of an agent using only the input of a camera or a system of cameras attached to the robot in motion. 

## Dataset
Here I have used KITTI Dataset, This is a very popular datset in Visual domain and one of the best datasets to start with for Such problems.

## Algorithm
- Find the corresponding features between adjacent frames
- Calculate the estimate essential matrix between those two images.
- decompose the essential matrix into rotation and translation.
- Scale the translation with the absolute or relative scale.
- Concatenate with the relative transformation.

This is the screen shot of the output of the code.
![Screenshot of the Ouput](https://github.com/AnuragSahu/Visual-Odometry/blob/master/result.png)
