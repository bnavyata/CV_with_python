from PIL import Image
from numpy import *
from scipy.ndimage import filters

im = array(Image.open('empire.jpg'))
im1 = array(Image.open('empire.jpg').convert('L'))
im2 = filters.gaussian_filter(im1, 50)
Image.fromarray(im).show()
Image.fromarray(im1).show()
Image.fromarray(im2).show()