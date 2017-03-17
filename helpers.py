import matplotlib.pyplot as plt
import cv2 
import numpy as np
from matplotlib.patches import Circle
from scipy.misc import imsave

def scale_abs(x, m = 255):
  x = np.absolute(x)
  x = np.uint8(m * x / np.max(x))
  return x 

def roi(gray, mn = 125, mx = 1200):
  m = np.copy(gray) + 1
  m[:, :mn] = 0 
  m[:, mx:] = 0 
  return m 

def save_image(img, name, i):
  path =  "output_images/" + name + str(i) + ".jpg"
  imsave(path, img)


def show_images(imgs, per_row = 3, per_col = 2, W = 10, H = 5, tdpi = 80):
      
  fig, ax = plt.subplots(per_col, per_row, figsize = (W, H), dpi = tdpi)
  ax = ax.ravel()
  
  for i in range(len(imgs)):
    img = imgs[i]
    ax[i].imshow(img)
  
  for i in range(per_row * per_col):
    ax[i].axis('off')


def show_dotted_image(this_image, points, thickness = 5, color = [255, 0, 255 ], d = 15):
  
  image = this_image.copy()
  
  cv2.line(image, points[0], points[1], color, thickness)
  cv2.line(image, points[2], points[3], color, thickness)

  fig, ax = plt.subplots(1)
  ax.set_aspect('equal')
  ax.imshow(image)

  for (x, y) in points:
    dot = Circle((x, y), d)
    ax.add_patch(dot)
  
  plt.show()

