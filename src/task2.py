import cv2
import numpy as np
from skimage.exposure import rescale_intensity
import os

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
        [1/16, 1/8, 1/16],
        [1/8, 1/4, 1/8],
        [1/16, 1/8, 1/16]), dtype="float32")
	
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

				k = np.median(roi)

				output[y - pad, x - pad, z] = k	

	output = rescale_intensity(output, in_range=(0, 255))
	output = (output * 255).astype("uint8")
	return output

if __name__ == "__main__":
	directory = 'data/task2'
	
	for idx, filename in enumerate(os.listdir('data/task2')):
		filepath = os.path.join(directory, filename)
		img = cv2.imread(filepath)
		if img is None:
			print(f"Error reading image: {filepath}")
			continue  # Skip to the next iteration
		if 'gb' in filename:
			smooth_image = gaussian_smoothing(img)
		elif 'sp' in filename:
			smooth_image = median_filter(img)

		cv2.imshow("original", img)
		cv2.imshow("{} - smooth".format('final'), smooth_image)
		# fn = 'output/task1/' + str(idx+1) + '.jpg'
		# cv2.imwrite(fn, convolveOutput)

		cv2.waitKey(0)
		cv2.destroyAllWindows()