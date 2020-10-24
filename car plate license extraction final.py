'''
AIM: CAR LICENSE PLATE EXTRACTION USING DIGITAL IMAGE PROCESSING

Author: DIVYA AGARWAL
'''

import cv2
import numpy as np

  
#input image
image = cv2.imread('img1.jpg')

#convert to gray scale image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Otsu's thresholding
#converts grey image to binary

#ret2,th2 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  #binary edge image
ret2,th2 = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)  #binary edge image


vertical = np.copy(th2)
horizontal = np.copy(th2)
#print(vertical)


#vertical edge detection
#sobel_vertical = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=9)
sobel_vertical = cv2.Sobel(th2, cv2.CV_64F, 1, 0, ksize=3)


#strucuring element
#To ensure that the license plate is not cropped (i.e., enclose all the license plate characters within a region), the maximum space between characters
kernel = np.ones((22,22),np.uint8)

'''
kernel =  cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
print(kernel)
'''

closing = cv2.morphologyEx(sobel_vertical, cv2.MORPH_CLOSE, kernel)

cv2.imshow('Original image',image)
cv2.imshow('Gray image', gray)
cv2.imshow('Sobel vertical', sobel_vertical)

#fig7
cv2.imshow('closing',closing)


#-----------------------------------------------


# Specify size on vertical axis
rows = vertical.shape[0]
#print(rows)
verticalsize = 15 

# Create structure element for extracting vertical lines through morphology operations
verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
#print(verticalStructure)

opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, verticalStructure)

#fig8 eliminates the regions whose height is less than the minimum character height.
cv2.imshow('opening',opening)


#-----------------------------------------------


#fig9 eliminate region with maximum character height
# Noise blobs taller than the plate
verticalsize = 50
verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
opening2 = cv2.morphologyEx(opening, cv2.MORPH_OPEN, verticalStructure)
cv2.imshow('opening2',opening2)


#----------------------------------------------


#fig10
#Taller noise blobs are eliminated
#This results in the elimination of regions with height greater than the maximum license plate height.
res = cv2.subtract(opening,opening2)
cv2.imshow('res',res)


#---------------------------------------------


# Specify size on horizontal axis
cols = horizontal.shape[1]
#print(cols)
horizontal_size = 50 
# Create structure element for extracting horizontal lines through morphology operations
horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

opening3 = cv2.morphologyEx(res, cv2.MORPH_OPEN, horizontalStructure)
#fig11
#horizonital SE (SE width is less thani minimum license plate width) eliminiates the noise blobs whose width is less than minimum width of license plate.
cv2.imshow('opening3',opening3)


#---------------------------------------------


#fig12
dest_and = cv2.bitwise_and(np.uint8(gray), np.uint8(opening3)) 
#print(dest_and)
cv2.imshow('AND',dest_and)


#-----------------------------------------------



cv2.waitKey(0)
cv2.destroyAllWindows()
