import cv2
import matplotlib.pyplot as plt
import numpy as np


image = cv2.imread("/Imaging/spatial_filt_img.jpg", cv2.IMREAD_GRAYSCALE)


def linear_avg_filter(img, mask_size):
    """ 
    returns linear avg filtered img

    Args:
        img jpg
        mask_size int

    Returns:
        blurred img jpg
    """
    avg_filt = np.ones([mask_size,mask_size])
    normalization = 1 / (mask_size**2)
    avg_filt = normalization * avg_filt
    output_img = np.zeros([img.shape[0],img.shape[1]])
    
    a = mask_size - 1
    a /= 2
    
    for i in range(1, img.shape[0]):
        for j in range(1,img.shape[1]):
            neighborhood = img[i - int(a) :i + int(a) + 1, j - int(a) : j + int(a) + 1]
            output_img[i- int(a) ,j - int(a)] = np.mean(neighborhood)
    
    return output_img
            
    

blurred_img = linear_avg_filter(image, 3)
blurred_img_2 = linear_avg_filter(image, 9)

# print(array)

plt.subplot(2,3,1)
plt.imshow(image, cmap = 'gray')
plt.title('Original')

plt.subplot(2,3,2)
plt.imshow(blurred_img, cmap = 'gray')
plt.title('Blurred')

plt.subplot(2,3,3)
plt.imshow(blurred_img_2, cmap = 'gray')
plt.title('Heavily Blurred')


plt.show()

