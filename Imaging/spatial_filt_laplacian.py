import cv2
import matplotlib.pyplot as plt
import numpy as np


image = cv2.imread("/Imaging/sp_filt_laplacian.jpg", cv2.IMREAD_GRAYSCALE)
            
    
def five_pt_laplacian(img):
    """
    five point laplacian, edge sharpening

    Args:
        img jpg

    Returns:
        img jpg
    """
    laplacian_arr = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    output_img = np.zeros_like(img, dtype=np.float32)
    
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            neighborhood = img[i - 1 :i + 2, j - 1 : j + 2]
            output_img[i, j] = np.sum(laplacian_arr * neighborhood)
            
    return output_img.astype(np.uint8)


output_img = five_pt_laplacian(image)

plt.subplot(2,3,1)
plt.imshow(image, cmap = 'gray')
plt.title('Original')

plt.subplot(2,3,2)
plt.imshow(output_img, cmap = 'gray')
plt.title('Laplacian')


plt.show()