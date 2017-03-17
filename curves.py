import cv2 
import numpy as np

class Curves:
  def __init__(self, number_of_windows, margin, minimum_pixels, ym_per_pix, xm_per_pix):
    
    self.min_pix = minimum_pixels
    self.margin = margin
    self.n = number_of_windows
    self.ky, self.kx = ym_per_pix, xm_per_pix

    self.binary, self.h, self.w, self.window_height = None, None, None, None
    self.all_pixels_x, self.all_pixels_y = None, None
    self.left_pixels_indices, self.right_pixels_indices = [], []
    self.left_pixels_x, self.left_pixels_y = None, None
    self.right_pixels_x, self.right_pixels_y = None, None
    self.out_img = None 
    self.left_fit_curve_pix, self.right_fit_curve_pix = None, None
    self.left_fit_curve_f, self.right_fit_curve_f = None, None
    self.left_radius, self.right_radius = None, None
    self.vehicle_position, self.vehicle_position_words = None, None
    self.result = {}
    
  def store_details(self, binary):
    self.out_img = np.dstack((binary, binary, binary)) * 255
    self.binary = binary
    self.h, self.w = binary.shape[0], binary.shape[1]
    self.mid = self.h / 2
    self.window_height = np.int(self.h / self.n)  
    self.all_pixels_x = np.array(binary.nonzero()[1])
    self.all_pixels_y = np.array(binary.nonzero()[0])
    
  def start(self, binary):
    hist = np.sum(binary[np.int(self.h / 2):, :], axis = 0)
    mid = np.int(hist.shape[0] / 2)
    current_leftx = np.argmax(hist[:mid])
    current_rightx = np.argmax(hist[mid:]) + mid
    return current_leftx, current_rightx

  def next_y(self, w):
    y_lo = self.h - (w + 1) * self.window_height
    y_hi = self.h - w * self.window_height 
    return y_lo, y_hi

  def next_x(self, current):
    x_left = current - self.margin
    x_right = current + self.margin
    return x_left, x_right
  
  def next_midx(self, current, pixel_indices):
    if len(pixel_indices) > self.min_pix:
      current = np.int(np.mean(self.all_pixels_x[pixel_indices]))
    return current

  def draw_boundaries(self, p1, p2, color, thickness = 5):
    cv2.rectangle(self.out_img, p1, p2, color, thickness)
  
  def indices_within_boundary(self, y_lo, y_hi, x_left, x_right):
    cond1 = (self.all_pixels_y >= y_lo)
    cond2 = (self.all_pixels_y < y_hi)
    cond3 = (self.all_pixels_x >= x_left)
    cond4 = (self.all_pixels_x < x_right)
    return (cond1 & cond2 & cond3 & cond4 ).nonzero()[0]

  def pixel_locations(self, indices):
    return self.all_pixels_x[indices], self.all_pixels_y[indices]
  
  def plot(self, t = 4):
  
    self.out_img[self.left_pixels_y, self.left_pixels_x] = [255, 0, 255]
    self.out_img[self.right_pixels_y, self.right_pixels_x] = [0, 255, 255]

    self.left_fit_curve_pix = np.polyfit(self.left_pixels_y, self.left_pixels_x, 2)
    self.right_fit_curve_pix = np.polyfit(self.right_pixels_y, self.right_pixels_x, 2)

    kl, kr = self.left_fit_curve_pix, self.right_fit_curve_pix
    ys = np.linspace(0, self.h - 1, self.h)
    
    left_xs = kl[0] * (ys**2) + kl[1] * ys + kl[2]
    right_xs = kr[0] * (ys**2) + kr[1] * ys + kr[2]
    
    xls, xrs, ys = left_xs.astype(np.uint32), right_xs.astype(np.uint32), ys.astype(np.uint32)
    
    for xl, xr, y in zip(xls, xrs, ys):
      cv2.line(self.out_img, (xl - t, y), (xl + t, y), (255, 255, 0), int(t / 2))
      cv2.line(self.out_img, (xr - t, y), (xr + t, y), (0, 0, 255), int(t / 2))
  
  def get_real_curvature(self, xs, ys):
    return np.polyfit(ys * self.ky, xs * self.kx, 2)
  
  def radius_of_curvature(self, y, f):
    return ((1 + (2 * f[0] * y + f[1])**2)**(1.5)) / np.absolute(2 * f[0])

  def update_vehicle_position(self):
    y = self.h
    mid = self.w / 2
    kl, kr = self.left_fit_curve_pix, self.right_fit_curve_pix
    xl = kl[0] * (y**2) + kl[1]* y + kl[2]
    xr = kr[0] * (y**2) + kr[1]* y + kr[2]
    pix_pos = xl + (xr - xl) / 2
    self.vehicle_position = (pix_pos - mid) * self.kx 

    if self.vehicle_position < 0:
      self.vehicle_position_words = str(np.absolute(np.round(self.vehicle_position, 2))) + " m left of center"
    elif self.vehicle_position > 0:
      self.vehicle_position_words = str(np.absolute(np.round(self.vehicle_position, 2))) + " m right of center"
    else:
      self.vehicle_position_words = "at the center"

  def fit(self, binary):
    
    self.store_details(binary)
    mid_leftx, mid_rightx = self.start(binary)

    left_pixels_indices, right_pixels_indices = [], []
    x, y = [None, None, None, None], [None, None]
    
    for w in range(self.n):
      
      y[0], y[1] = self.next_y(w)
      x[0], x[1] = self.next_x(mid_leftx) 
      x[2], x[3] = self.next_x(mid_rightx)
        
      self.draw_boundaries((x[0], y[0]), (x[1], y[1]), (255, 0, 0))
      self.draw_boundaries((x[2], y[0]), (x[3], y[1]), (0, 255, 0))
      
      curr_left_pixels_indices = self.indices_within_boundary(y[0], y[1], x[0], x[1])
      curr_right_pixels_indices = self.indices_within_boundary(y[0], y[1], x[2], x[3])
      
      left_pixels_indices.append(curr_left_pixels_indices)
      right_pixels_indices.append(curr_right_pixels_indices)
      
      mid_leftx = self.next_midx(mid_leftx, curr_left_pixels_indices)
      mid_rightx = self.next_midx(mid_rightx, curr_right_pixels_indices)
    
    self.left_pixels_indices = np.concatenate(left_pixels_indices)
    self.right_pixels_indices = np.concatenate(right_pixels_indices)
    
    self.left_pixels_x, self.left_pixels_y = self.pixel_locations(self.left_pixels_indices)
    self.right_pixels_x, self.right_pixels_y = self.pixel_locations(self.right_pixels_indices)

    self.left_fit_curve_f = self.get_real_curvature(self.left_pixels_x, self.left_pixels_y)
    self.right_fit_curve_f = self.get_real_curvature(self.right_pixels_x, self.right_pixels_y)
    
    self.left_radius = self.radius_of_curvature(self.h * self.ky, self.left_fit_curve_f)
    self.right_radius = self.radius_of_curvature(self.h *  self.ky, self.right_fit_curve_f)

    self.plot()
    self.update_vehicle_position()

    self.result = {
      'image': self.out_img,
      'left_radius': self.left_radius,
      'right_radius': self.right_radius,
      'real_left_best_fit_curve': self.left_fit_curve_f,
      'real_right_best_fit_curve': self.right_fit_curve_f, 
      'pixel_left_best_fit_curve': self.left_fit_curve_pix,
      'pixel_right_best_fit_curve': self.right_fit_curve_pix, 
      'vehicle_position': self.vehicle_position, 
      'vehicle_position_words': self.vehicle_position_words
    }

    return self.result