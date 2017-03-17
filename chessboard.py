
import cv2 
import numpy as np
from scipy.misc import imread

class ChessBoard:
    
  def __init__(self, i, path, nx = 9, ny = 6): 
    
    self.i = i
    self.path = path
    self.nx, self.ny = nx, ny
    self.n = (self.nx, self.ny)
    
    temp_image = imread(self.path)
    temp_gray = cv2.cvtColor(temp_image, cv2.COLOR_RGB2GRAY)
    
    self.rows, self.cols, self.channels = temp_image.shape
    self.dimensions = (self.rows, self.cols)
    
    self.has_corners, self.corners = cv2.findChessboardCorners(temp_gray, self.n, None)
    self.object_points = self.get_object_points()
    self.matrix, self.distortion, self.can_undistort = None, None, False

  def get_object_points(self):
    # (0, 0 ,0), (0, 1, 0)... (8, 5, 0)
    number_of_points = self.nx * self.ny
    points = np.zeros((number_of_points, 3), np.float32)
    points[:, :2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1, 2)
    return points
    
  def image(self):
    temp_image = imread(self.path)
    return temp_image

  def image_with_corners(self):
    ''' if this image doesn't have calculated corners, return raw image '''
    temp_image = imread(self.path)
    if self.has_corners:
      cv2.drawChessboardCorners(temp_image, self.n, self.corners, self.has_corners)
    return temp_image

  def undistorted_image(self):
    '''if camera parameters is not initialized, return None '''
    temp_image = None
    if self.can_undistort:
      temp_image = imread(self.path)
      temp_image = cv2.undistort(temp_image, self.matrix, self.distortion, None, self.matrix)
    return temp_image
  
  def load_undistort_params(self, camera_matrix, distortion):
    self.distortion = distortion
    self.matrix = camera_matrix
    self.can_undistort = True