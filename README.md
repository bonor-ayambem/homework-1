# Image Enhancement

## Introduction

Image enhancement is a process in Computer Vision that deals with the modification of images, to 
improve their quality and render them more viable and effective for computer vision tasks.

This project explores image enhancement, particularly image sharpening and image denoising, for the 
purpose of understanding the different techniques that are involved in carrying them out effectively.

## Implementation
### Task 1 - Image Sharpening

Task 1 develops a program to perform _unsharp masking_ on input images, in order to enhance their major
edges.

Unsharp masking is performed in two steps, by following the folowing formula:

- **Step 1: I - G(I) = D(I)**: here G(I) is the blurred image using a Gaussian kernel/filter of choice,
and D(I) is the residual image containing edge details.

- **Step 2: I + D(I) = Final**: adding the details back to image reinforces the edge information creating a 
sharper image.

**Final** is the final sharpened image

To achieve this, a method called `convolve(image, kernel)` was implemented.
`convolve()` performs a sliding convolution of `image` and `kernel`, as is common in Computer Vision operations.

For the purpose of this task, a Gaussian filter is convolved with the original image, and the result from that 
operation is subtracted from the original image. We add the difference back to the original image to generate 
the final sharpened image. The Gaussian filter used is defined as follows:

```
gaussian = np.array((
        [1/16, 1/8, 1/16],
        [1/8, 1/4, 1/8],
        [1/16, 1/8, 1/16]), dtype="float32")
```
The `convolve()` function was taken from [Adrian Rosebrock] (https://pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/),
however, the following changes were made:

- To allow the function to process color images, the convolution process takes place in all 3 dimensions. This was not previously the case, and resulted in the function producing only greyscale images
- After much experimentation and difficulties, a kernel was found which worked well for this task. This kernel is different from any of those used in the original `convolve()` function

### Task 2 - Image Denoising



## 498 Graduate Level Additional Questions

## References
- [Convolutions with OpenCV and Python] (https://pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/)
