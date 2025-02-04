import cv2
import matplotlib.pyplot as plt
import numpy as np

image_original = cv2.imread("/Imaging/med_filt.jpg", cv2.IMREAD_GRAYSCALE)


def median_filter(image):
    """
    returns median filtered img for noise reduction

    Args:
        image jpg

    Returns:
        img jpg noise reduced
    """
    med_image = np.zeros_like(image)
    filter_size = 3
    padding = 1
    
    padded_image = np.pad(med_image, (padding,padding), 'constant', constant_values=0)
    
    for i in range(1, image.shape[0]):
        for j in range(1, image.shape[1]):
            neighborhood = image[i:i + filter_size, j : j + filter_size]
            padded_image [i,j] = np.median(neighborhood)
    
    return padded_image


median_image = median_filter(image_original)

plt.subplot(2,3,1)
plt.imshow(image_original, cmap = 'gray')
plt.title('Original')

plt.subplot(2,3,2)
plt.imshow(median_image, cmap = 'gray')
plt.title('Median Image')

plt.show()