import cv2
import numpy as np
from skimage.exposure import rescale_intensity


directory = 'data/task2'

def mean_filter(image):
	kernel = np.array((
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]), dtype="float32")
	
	(iH, iW) = image.shape[:2]
	(kH, kW) = kernel.shape[:2]

	pad = (kW - 1) // 2
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
	output = np.zeros((iH, iW, 3), dtype="float32")
    
	for z in range(0,3):
		for y in np.arange(pad, iH + pad):
			for x in np.arange(pad, iW + pad):
				roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1, z]

				k = (roi * kernel).sum()

				output[y - pad, x - pad, z] = k	

	output = rescale_intensity(output, in_range=(0, 255))
	output = (output * 255).astype("uint8")

	return output

def gaussian_smoothing(image):
	kernel = np.array((
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]), dtype="float32")
	
	(iH, iW) = image.shape[:2]
	(kH, kW) = kernel.shape[:2]

	pad = (kW - 1) // 2
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
	output = np.zeros((iH, iW, 3), dtype="float32")
    
	for z in range(0,3):
		for y in np.arange(pad, iH + pad):
			for x in np.arange(pad, iW + pad):
				roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1, z]

				k = (roi * kernel).sum()

				output[y - pad, x - pad, z] = k	

	output = rescale_intensity(output, in_range=(0, 255))
	output = (output * 255).astype("uint8")

	return output

def median_filter(image):	
	(iH, iW) = image.shape[:2]

	pad = 1
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
	output = np.zeros((iH, iW, 3), dtype="float32")
    
	for z in range(0,3):
		for y in np.arange(pad, iH + pad):
			for x in np.arange(pad, iW + pad):
				roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1, z]

				k = roi.median()

				output[y - pad, x - pad, z] = k	

	output = rescale_intensity(output, in_range=(0, 255))
	output = (output * 255).astype("uint8")

	return output