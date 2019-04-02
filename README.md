# PythonComputerVision-5-AR
借助pygame和openGL在平面内实现简单的AR例子
## 一.以平面和标记物进行姿态评估
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
### A组
![image](https://github.com/Nocami/PythonComputerVision-5-AR/blob/master/images/book_frontal.JPG)  
![image](https://github.com/Nocami/PythonComputerVision-5-AR/blob/master/images/book_perspective.JPG)  
### B组
![image](https://github.com/Nocami/PythonComputerVision-5-AR/blob/master/images/11111.JPG)  
![image](https://github.com/Nocami/PythonComputerVision-5-AR/blob/master/images/2222.JPG)  
运行代码后，我们可以绘制出漂亮的立方体。使用平面物体作为标记物，来计算用于新视图投影矩阵的例子。将图像的特征和对齐后的标记匹配，计算出单应性矩阵，然后用于计算照相机的姿态。带有一个灰色正方形区域的模板图像(**左图**)  。从未知视角拍摄的一幅图像，该图像包含同一个正方形，该正方形已经经过估计的单应性矩阵进行了变换(**右图**) 。  
### A组
![image](https://github.com/Nocami/PythonComputerVision-5-AR/blob/master/images/1111.jpg)  
### B组
![image](https://github.com/Nocami/PythonComputerVision-5-AR/blob/master/images/3333.jpg)  
使用计算出的照相机矩阵变换立方体如下：  
### A组
![image](https://github.com/Nocami/PythonComputerVision-5-AR/blob/master/images/QQ%E6%88%AA%E5%9B%BE20190402151145.jpg)  
### B组
![image](https://github.com/Nocami/PythonComputerVision-5-AR/blob/master/images/4444.jpg) 
## 二.在图像中放置虚拟物体
### 增强现实
增强现实（Augmented Reality,AR）是将物体和相应信息放置在图像数据上的一系列的总称。最经典的例子是放置一个三维计算机图形学模型，使其看起来属于该场景；如果在视频中，该模型会随着照相机的运动很自然的移动。这里介绍一个简单的例子，即在图像中放置虚拟物品。我们要用到两个工具包：**PyGame和PyOpenGL**。  
安装方式：  
~~~
pip install pygame
~~~
~~~
pip install PyOpenGL PyOpenGL_accelerate
~~~
如果pip源下载缓慢，可以自行去第三方站点，如https://www.lfd.uci.edu/~gohlke/pythonlibs/ 
运行如下源码：  
~~~python
# import math
# import pickle
from pylab import *
from OpenGL.GL import * 
from OpenGL.GLU import * 
from OpenGL.GLUT import * 
import pygame, pygame.image 
from pygame.locals import *
from PCV.geometry import homography, camera
from PCV.localdescriptors import sift

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
    row, col = sz
    fx = 2555*col/2592
    fy = 2586*row/1936
    K = diag([fx, fy, 1])
    K[0, 2] = 0.5*col
    K[1, 2] = 0.5*row
    return K

def set_projection_from_camera(K): 
	glMatrixMode(GL_PROJECTION) 
	glLoadIdentity()
	fx = K[0,0] 
	fy = K[1,1] 
	fovy = 2*math.atan(0.5*height/fy)*180/math.pi 
	aspect = (width*fy)/(height*fx)
	near = 0.1 
	far = 100.0
	gluPerspective(fovy,aspect,near,far) 
	glViewport(0,0,width,height)

def set_modelview_from_camera(Rt): 
	glMatrixMode(GL_MODELVIEW) 
	glLoadIdentity()
	Rx = np.array([[1,0,0],[0,0,-1],[0,1,0]])
	R = Rt[:,:3] 
	U,S,V = np.linalg.svd(R) 
	R = np.dot(U,V) 
	R[0,:] = -R[0,:]
	t = Rt[:,3]
	M = np.eye(4) 
	M[:3,:3] = np.dot(R,Rx) 
	M[:3,3] = t
	M = M.T
	m = M.flatten()
	glLoadMatrixf(m)

def draw_background(imname):
	bg_image = pygame.image.load(imname).convert() 
	bg_data = pygame.image.tostring(bg_image,"RGBX",1)
	glMatrixMode(GL_MODELVIEW) 
	glLoadIdentity()

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
	glEnable(GL_TEXTURE_2D) 
	glBindTexture(GL_TEXTURE_2D,glGenTextures(1)) 
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,width,height,0,GL_RGBA,GL_UNSIGNED_BYTE,bg_data) 
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST) 
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST)
	glBegin(GL_QUADS) 
	glTexCoord2f(0.0,0.0); glVertex3f(-1.0,-1.0,-1.0) 
	glTexCoord2f(1.0,0.0); glVertex3f( 1.0,-1.0,-1.0) 
	glTexCoord2f(1.0,1.0); glVertex3f( 1.0, 1.0,-1.0) 
	glTexCoord2f(0.0,1.0); glVertex3f(-1.0, 1.0,-1.0) 
	glEnd()
	glDeleteTextures(1)


def draw_teapot(size):
	glEnable(GL_LIGHTING) 
	glEnable(GL_LIGHT0) 
	glEnable(GL_DEPTH_TEST) 
	glClear(GL_DEPTH_BUFFER_BIT)
	glMaterialfv(GL_FRONT,GL_AMBIENT,[0,0,0,0]) 
	glMaterialfv(GL_FRONT,GL_DIFFUSE,[0.5,0.0,0.0,0.0]) 
	glMaterialfv(GL_FRONT,GL_SPECULAR,[0.7,0.6,0.6,0.0]) 
	glMaterialf(GL_FRONT,GL_SHININESS,0.25*128.0) 
	glutSolidTeapot(size)

width,height = 1000,747
def setup():
	pygame.init() 
	pygame.display.set_mode((width,height),OPENGL | DOUBLEBUF) 
	pygame.display.set_caption("OpenGL AR demo")    

# compute features
sift.process_image('book_frontal.JPG', 'im0.sift')
l0, d0 = sift.read_features_from_file('im0.sift')

sift.process_image('book_perspective.JPG', 'im1.sift')
l1, d1 = sift.read_features_from_file('im1.sift')

# match features and estimate homography
matches = sift.match_twosided(d0, d1)
ndx = matches.nonzero()[0]
fp = homography.make_homog(l0[ndx, :2].T)
ndx2 = [int(matches[i]) for i in ndx]
tp = homography.make_homog(l1[ndx2, :2].T)

model = homography.RansacModel()
H, inliers = homography.H_from_ransac(fp, tp, model)

K = my_calibration((747, 1000))
cam1 = camera.Camera(hstack((K, dot(K, array([[0], [0], [-1]])))))
box = cube_points([0, 0, 0.1], 0.1)
box_cam1 = cam1.project(homography.make_homog(box[:, :5]))
box_trans = homography.normalize(dot(H,box_cam1))
cam2 = camera.Camera(dot(H, cam1.P))
A = dot(linalg.inv(K), cam2.P[:, :3])
A = array([A[:, 0], A[:, 1], cross(A[:, 0], A[:, 1])]).T
cam2.P[:, :3] = dot(K, A)

Rt=dot(linalg.inv(K),cam2.P)
 
setup() 
draw_background("book_perspective.bmp") 
set_projection_from_camera(K) 
set_modelview_from_camera(Rt)
draw_teapot(0.05)

pygame.display.flip()
while True: 
	for event in pygame.event.get():
		if event.type==pygame.QUIT:
			sys.exit()

~~~
### 运行结果
![image]()  
![image]()  
增强显示，使用由特征匹配计算出的照相机参数，将一个计算机图形学模型放置在场景中的书本上：将茶壶按照坐标轴对齐方式显示出来。  
