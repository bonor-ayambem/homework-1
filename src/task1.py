from skimage.exposure import rescale_intensity
import numpy as np
import cv2
import os


def convolve(image, kernel):
    # grab the spatial dimensions of the image, along with
	# the spatial dimensions of the kernel
	(iH, iW) = image.shape[:2]
	(kH, kW) = kernel.shape[:2]
	# print(image.shape)
	# print(kernel.shape)

	# allocate memory for the output image, taking care to
	# "pad" the borders of the  input image so the spatial
	# size (i.e., width and height) are not reduced
	pad = (kW - 1) // 2
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
	output = np.zeros((iH, iW, 3), dtype="float32")
    
    # loop over the input image, "sliding" the kernel across
	# each (x, y)-coordinate from left-to-right and top to
	# bottom
	for z in range(0,3):
		for y in np.arange(pad, iH + pad):
			for x in np.arange(pad, iW + pad):
				# extract the ROI of the image by extracting the
				# *center* region of the current (x, y)-coordinates
				# dimensions
				roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1, z]
				# perform the actual convolution by taking the
				# element-wise multiplicate between the ROI and
				# the kernel, then summing the matrix
				k = (roi * kernel).sum()
				# store the convolved value in the output (x,y)-
				# coordinate of the output image
				output[y - pad, x - pad, z] = k	
    # rescale the output image to be in the range [0, 255]
	output = rescale_intensity(output, in_range=(0, 255))
	output = (output * 255).astype("uint8")
	# return the output image
	return output

if __name__ == "__main__":
	directory = 'data/task1'

	# sharpen = np.array((
	# [0, -1, 0],
	# [-1, 5, -1],
	# [0, -1, 0]), dtype="int")

	gaussian = np.array((
        [1/16, 1/8, 1/16],
        [1/8, 1/4, 1/8],
        [1/16, 1/8, 1/16]), dtype="float32")

	# construct average blurring kernels used to smooth an image
	smallBlur = np.ones((3, 3), dtype="float") * (1.0 / (3 * 3))
	largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

	for idx, filename in enumerate(os.listdir('data/task1')):
		filepath = os.path.join(directory, filename)
		img = cv2.imread(filepath)
		
		convolveOutput = convolve(img, gaussian)
		cv2.imshow("{} - convole".format('convolve output'), convolveOutput)
		print(convolveOutput.shape)
		# convolveOutputRGB = cv2.cvtColor(convolveOutput, cv2.COLOR_GRAY2BGR)

		di = img - convolveOutput
		cv2.imshow("{} - convole".format('di'), di)
		print(di.shape)

		final = img + di
		# cv2.imshow("{} - convole".format('final'), final)

		cv2.imshow("original", img)
		cv2.imshow("{} - convole".format('final'), convolveOutput)
		fn = 'output/task1/' + str(idx+1) + '.jpg'
		cv2.imwrite(fn, convolveOutput)

		# print(f"Number of channels after convolution: {convolveOutputRGB.shape[2] if len(convolveOutputRGB.shape) == 3 else 1}")

		cv2.waitKey(0)
		cv2.destroyAllWindows()
