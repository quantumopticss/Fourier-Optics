import numpy as np
import cv2 as cv

path = "test_img2.jpg"
img = cv.imread(path)

n,m,k = img.shape
print(n,m,k)

E_img = np.sqrt(img)

## coordinates
y = np.arange(-n//2,n//2)
x = np.arange(-m//2,m//2)

xx,yy = np.meshgrid(x,y)
rho_r = np.sqrt(xx**2 + yy**2)

## mask 1
theta = np.arctan2(yy,xx) ## helitical phase
L = -1 # topological core 
mask_1 = np.exp(1j*L*theta)

## mask 2
a = 30
tau = 20
#mask_2 = (rho_r >= a) + (rho_r <= a)*np.exp(-(rho_r-a)**2/tau**2) # high pass
mask_2 = (rho_r <= a) + (rho_r > a)*np.exp(-(rho_r-a)**2/tau**2) # low pass
print(mask_2)

E_processed_img = np.empty_like(E_img,dtype = complex)
for i in range(k):
    E_img_i = E_img[:,:,i]
    f_E_img = np.fft.fftshift(np.fft.fft2(E_img_i),axes = (0,1))
    
    f_E_img = f_E_img*mask_2#*mask_1
    
    E_processed_img[:,:,i] = np.fft.ifft2(f_E_img)
    #processed_img[:,:,i] = processed_img[:,:,i]/np.max(processed_img[:,:,i],axis = None)
    
# figrue
processed_img = np.abs(E_processed_img)**2
processed_img = np.uint8(255*processed_img/np.max(processed_img,axis = None))
cv.namedWindow("processed img", cv.WINDOW_NORMAL)
cv.imshow("processed img",processed_img)
cv.imwrite("output_image.png", processed_img)  # Saves in PNG format
k = cv.waitKey(0) # Wait for a keystroke in the window

import matplotlib.pyplot as plt
plt.imshow(processed_img)
plt.show()