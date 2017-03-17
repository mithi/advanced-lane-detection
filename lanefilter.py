import cv2 
import numpy as np
from helpers import roi, scale_abs

class LaneFilter:
  def __init__(self, p):
    self.sat_thresh = p['sat_thresh']
    self.light_thresh = p['light_thresh'] 
    self.light_thresh_agr = p['light_thresh_agr']
    self.grad_min, self.grad_max = p['grad_thresh']
    self.mag_thresh, self.x_thresh = p['mag_thresh'], p['x_thresh']
    self.hls, self.l, self.s, self.z  = None, None, None, None
    self.color_cond1, self.color_cond2 = None, None
    self.sobel_cond1, self.sobel_cond2, self.sobel_cond3 = None, None, None 

  def sobel_breakdown(self, img):
    self.apply(img)
    b1, b2, b3 = self.z.copy(), self.z.copy(), self.z.copy()
    b1[(self.sobel_cond1)] = 255
    b2[(self.sobel_cond2)] = 255
    b3[(self.sobel_cond3)] = 255
    return np.dstack((b1, b2,b3))

  def color_breakdown(self, img):
    self.apply(img)
    b1, b2 = self.z.copy(), self.z.copy()
    b1[(self.color_cond1)] = 255
    b2[(self.color_cond2)] = 255
    return np.dstack((b1, b2, self.z))

  def apply(self, rgb_image):    
    self.hls = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HLS)
    self.l = self.hls[:, :, 1]
    self.s = self.hls[:, :, 2]
    self.z = np.zeros_like(self.s)
    color_img = self.apply_color_mask()
    sobel_img = self.apply_sobel_mask()
    filtered_img = cv2.bitwise_or(sobel_img, color_img)
    return filtered_img

  def apply_color_mask(self):   
    self.color_cond1 = (self.s > self.sat_thresh) & (self.l > self.light_thresh)
    self.color_cond2 = self.l > self.light_thresh_agr
    b = self.z.copy()
    b[(self.color_cond1 | self.color_cond2)] = 1
    return b

  def apply_sobel_mask(self):       
    lx = cv2.Sobel(self.l, cv2.CV_64F, 1, 0, ksize = 5)
    ly = cv2.Sobel(self.l, cv2.CV_64F, 0, 1, ksize = 5)
    gradl = np.arctan2(np.absolute(ly), np.absolute(lx))
    l_mag = np.sqrt(lx**2 + ly**2)
    slm, slx, sly = scale_abs(l_mag), scale_abs(lx), scale_abs(ly)
    b = self.z.copy()
    self.sobel_cond1 = slm > self.mag_thresh
    self.sobel_cond2 = slx > self.x_thresh
    self.sobel_cond3 = (gradl > self.grad_min) & (gradl < self.grad_max)
    b[(self.sobel_cond1 & self.sobel_cond2 & self.sobel_cond3)] = 1  
    return b 