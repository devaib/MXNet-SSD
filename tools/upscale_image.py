import cv2
import os

img = cv2.imread(os.path.join(os.getcwd(), '../data', 'kitti',
                              'data_object_image_2', 'training',
                              'image_2', '000025.png'))
resized1 = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
size = resized1.shape
cropped1 = resized1[size[0]/4:size[0]/4*3, size[1]/4:size[1]/4*3]
resized2 = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
size = resized2.shape
cropped2 = resized2[size[0]/8*3:size[0]/8*5, size[1]/8*3:size[1]/8*5]

cv2.imshow('Original image', img)
cv2.imshow('Upscale image1', cropped1)
cv2.imshow('Upscale image2', cropped2)

cv2.waitKey(0)
cv2.imwrite(os.path.join(os.getcwd(), '../data', 'test', 'original.png'), img)
cv2.imwrite(os.path.join(os.getcwd(), '../data', 'test', 'cropped1.png'), cropped1)
cv2.imwrite(os.path.join(os.getcwd(), '../data', 'test', 'cropped2.png'), cropped2)
cv2.destroyAllWindows()
