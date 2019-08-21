from PIL import Image
from numpy import *
from scipy.ndimage import filters
import matplotlib.pyplot as plt

original = array(Image.open('empire.jpg'))
gray = array(Image.open('empire.jpg').convert('L'))
gray_blurred = filters.gaussian_filter(gray, 10)

colored_blurred = zeros(original.shape)#declare empty 
for i in range(3):
    colored_blurred[:,:,i] = filters.gaussian_filter(original[:,:,i],10)#fill values after gaussian filter
colored_blurred = uint8(colored_blurred)#convert to array

fig = plt.figure()
plt.gray()  # show the filtered result in grayscale
ax1 = fig.add_subplot(141)  # 1st
ax2 = fig.add_subplot(142)  # 2nd
ax3 = fig.add_subplot(143)  # 3rd
ax4 = fig.add_subplot(144)  # 4th

ax1.imshow(original)
ax2.imshow(gray)
ax3.imshow(gray_blurred)
ax4.imshow(colored_blurred)
plt.show()

