# PythonComputerVision-5-AR
借助pygame和openGL在平面内实现简单的AR例子
## 以平面和标记物进行姿态评估
通过RANSAC算法我们可以得到一个稳健地单应性矩阵，该单应性矩阵将一副图片中标记物（比如书本）上的点映射到另外一幅图像中的对应点。我们定义相应的三位坐标系，使标记物在X-Y平面上(Z=0),原点在标记物的某位置上。我们可以检验单应性矩阵结果的正确性，比如将一些简单的三维物体放置在标记物上，这里我们用一个立方体举例：  
代码如下：  
~~~python
from pylab import *
from PIL import Image

# If you have PCV installed, these imports should work
from PCV.geometry import homography, camera
from PCV.localdescriptors import sift

"""
This is the augmented reality and pose estimation cube example from Section 4.3.
"""


def cube_points(c, wid):
    """ Creates a list of points for plotting
        a cube with plot. (the first 5 points are
        the bottom square, some sides repeated). """
    p = []
    # bottom
    p.append([c[0]-wid, c[1]-wid, c[2]-wid])
    p.append([c[0]-wid, c[1]+wid, c[2]-wid])
    p.append([c[0]+wid, c[1]+wid, c[2]-wid])
    p.append([c[0]+wid, c[1]-wid, c[2]-wid])
    p.append([c[0]-wid, c[1]-wid, c[2]-wid]) #same as first to close plot
    
    # top
    p.append([c[0]-wid, c[1]-wid, c[2]+wid])
    p.append([c[0]-wid, c[1]+wid, c[2]+wid])
    p.append([c[0]+wid, c[1]+wid, c[2]+wid])
    p.append([c[0]+wid, c[1]-wid, c[2]+wid])
    p.append([c[0]-wid, c[1]-wid, c[2]+wid]) #same as first to close plot
    
    # vertical sides
    p.append([c[0]-wid, c[1]-wid, c[2]+wid])
    p.append([c[0]-wid, c[1]+wid, c[2]+wid])
    p.append([c[0]-wid, c[1]+wid, c[2]-wid])
    p.append([c[0]+wid, c[1]+wid, c[2]-wid])
    p.append([c[0]+wid, c[1]+wid, c[2]+wid])
    p.append([c[0]+wid, c[1]-wid, c[2]+wid])
    p.append([c[0]+wid, c[1]-wid, c[2]-wid])
    
    return array(p).T


def my_calibration(sz):
    """
    Calibration function for the camera (iPhone4) used in this example.
    """
    row, col = sz
    fx = 2555*col/2592
    fy = 2586*row/1936
    K = diag([fx, fy, 1])
    K[0, 2] = 0.5*col
    K[1, 2] = 0.5*row
    return K



# compute features
sift.process_image('../data/book_frontal.JPG', 'im0.sift')
l0, d0 = sift.read_features_from_file('im0.sift')

sift.process_image('../data/book_perspective.JPG', 'im1.sift')
l1, d1 = sift.read_features_from_file('im1.sift')


# match features and estimate homography
matches = sift.match_twosided(d0, d1)
ndx = matches.nonzero()[0]
fp = homography.make_homog(l0[ndx, :2].T)
ndx2 = [int(matches[i]) for i in ndx]
tp = homography.make_homog(l1[ndx2, :2].T)

model = homography.RansacModel()
H, inliers = homography.H_from_ransac(fp, tp, model)

# camera calibration
K = my_calibration((747, 1000))

# 3D points at plane z=0 with sides of length 0.2
box = cube_points([0, 0, 0.1], 0.1)

# project bottom square in first image
cam1 = camera.Camera(hstack((K, dot(K, array([[0], [0], [-1]])))))
# first points are the bottom square
box_cam1 = cam1.project(homography.make_homog(box[:, :5]))


# use H to transfer points to the second image
box_trans = homography.normalize(dot(H,box_cam1))

# compute second camera matrix from cam1 and H
cam2 = camera.Camera(dot(H, cam1.P))
A = dot(linalg.inv(K), cam2.P[:, :3])
A = array([A[:, 0], A[:, 1], cross(A[:, 0], A[:, 1])]).T
cam2.P[:, :3] = dot(K, A)

# project with the second camera
box_cam2 = cam2.project(homography.make_homog(box))



# plotting
im0 = array(Image.open('book_frontal.JPG'))
im1 = array(Image.open('book_perspective.JPG'))

figure()
imshow(im0)
plot(box_cam1[0, :], box_cam1[1, :], linewidth=3)
title('2D projection of bottom square')
axis('off')

figure()
imshow(im1)
plot(box_trans[0, :], box_trans[1, :], linewidth=3)
title('2D projection transfered with H')
axis('off')

figure()
imshow(im1)
plot(box_cam2[0, :], box_cam2[1, :], linewidth=3)
title('3D points projected in second image')
axis('off')

show()
~~~
代码的读入文件为下图，分别为一本书的平面俯视图以及任意视角的透视图：  
![image](https://github.com/Nocami/PythonComputerVision-5-AR/blob/master/images/book_frontal.JPG)  
![image](https://github.com/Nocami/PythonComputerVision-5-AR/blob/master/images/book_perspective.JPG)  
运行代码后，我们可以绘制出漂亮的立方体。使用平面物体作为标记物，来计算用于新视图投影矩阵的例子。将图像的特征和对齐后的标记匹配，计算出单应性矩阵，然后用于计算照相机的姿态。带有一个灰色正方形区域的模板图像(**左图**)  。从未知视角拍摄的一幅图像，该图像包含同一个正方形，该正方形已经经过估计的单应性矩阵进行了变换(**右图**) 。
![image](https://github.com/Nocami/PythonComputerVision-5-AR/blob/master/images/1111.jpg)  
使用计算出的照相机矩阵变换立方体如下：  
![image](https://github.com/Nocami/PythonComputerVision-5-AR/blob/master/images/QQ%E6%88%AA%E5%9B%BE20190402151145.jpg)  
