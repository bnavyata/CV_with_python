im = array(Image.open('empire.jpg').convert('L'))

#Sobel derivative - x derivative
imx = zeros(im.shape)
filters.sobel(im,1,imx)
# y derivative
imy = zeros(im.shape)
filters.sobel(im,0,imy)
#magnitude
magnitude = sqrt(imx**2 + imy**2)


# Gaussian derivative - x derivative
imx_g = zeros(im.shape)
filters.gaussian_filter(im, (5,5),(0,1), imx_g)
# y derivative
imy_g = zeros(im.shape)
filters.gaussian_filter(im, (5,5),(1,0), imy_g)

fig = plt.figure()
plt.gray()  # show the filtered result in grayscale
ax1 = fig.add_subplot(151)
ax2 = fig.add_subplot(152) 
ax3 = fig.add_subplot(153)  
ax4 = fig.add_subplot(154) 
ax5 = fig.add_subplot(155) 

ax1.imshow(imx)
ax2.imshow(imy)
ax3.imshow(magnitude)
ax4.imshow(imx_g)
ax5.imshow(imy_g)
plt.show()