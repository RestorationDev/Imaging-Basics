import cv2
import matplotlib.pyplot as plt
import numpy as np


image = cv2.imread("/Imaging/histogram_eq_img.jpg", cv2.IMREAD_GRAYSCALE)

def histogram_equalization_transformation(img):
    """
    returns histogram equalization transformed image 

    Args:
        img jpg

    Returns:
        img jpg: evenly distributed contrast img
    """
    
    #Norm factor
    M, N = img.shape
    
    
    histogram, bins = np.histogram(img.flatten(), bins=256, range=[0, 256])
    
    hist_sum = np.cumsum(histogram)
    
    normal_cdf = hist_sum * float(histogram.max()) / hist_sum.max()
    
    T = np.uint(255*hist_sum/hist_sum[-1])
    
    return T, T[image]

#original image

transformation, enhanced_img = histogram_equalization_transformation(image)

plt.subplot(2,3,1)
plt.hist(image.ravel(), 256, [0,256])
plt.title('Original Histogram')

plt.subplot(2,3,2)
plt.plot(transformation, color='black')
plt.title('Histogram Equalization Transformation')

plt.subplot(2,3,3)
plt.hist(enhanced_img.ravel(), 256, [0,256])
plt.title("Equalized histogram")

plt.subplot(2,3,4)
plt.imshow(image, cmap = 'gray')
plt.title('Original')

plt.subplot(2,3,5)
plt.imshow(enhanced_img, cmap = 'gray')
plt.title('Enhanced')



plt.show()


