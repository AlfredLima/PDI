# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2


#%%

cv2.__version__

img = cv2.imread('/home/alunoic/bb8.jpg', cv2.IMREAD_GRAYSCALE)


#cv2.imshow('img',img)
#cv2.waitkey(0)



plt.subplot('211') 
plt.title('Original')
plt.imshow(img, 'gray')
plt.subplot('212')
plt.title('Histo')
plt.hist(img.ravel(), 256, [0, 256])




#cv2.destroyAllWindows()
