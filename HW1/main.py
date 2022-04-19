import numpy as np
from numpy import asarray
from numpy import savetxt
import scipy as sc
from scipy import signal
from skimage import filters
from PIL import Image
from matplotlib import pyplot as plt
import cv2


# Took this function from supplied material


def gaussian(sigma):
    filter_size = 2 * int(sigma * 4 + 0.5) + 1
    filter = np.zeros((filter_size, filter_size), dtype=np.float32)

    for i in range(filter_size):
        for j in range(filter_size):
            x = i - filter_size // 2
            y = j - filter_size // 2
            # gaussian equation:
            filter[i, j] = 1.0 / (2 * np.pi * sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    return filter


def separable_gaussian(sigma):
    filter_size = 2 * int(sigma * 4 + 0.5) + 1
    xfilter = np.zeros((filter_size), dtype=np.float32)
    yfilter = np.zeros((filter_size, 1), dtype=np.float32)
    for i in range(filter_size):
        xy = i - filter_size // 2
        xfilter[i] = 1.0 / np.sqrt((2 * np.pi * sigma ** 2)) * np.exp(-(xy ** 2) / (2 * sigma ** 2))
        yfilter[i,0] = 1.0 / np.sqrt((2 * np.pi * sigma ** 2)) * np.exp(-(xy ** 2) / (2 * sigma ** 2))
    return xfilter, yfilter


def convolution1(image, filter):
    # Obtain image dimensions and filter dimension/Kernel
    dimensionsImg = image.shape
    dimensionsFltr = filter.shape
    height = dimensionsImg[0]
    length = dimensionsImg[1]
    kernel = dimensionsFltr[0]

    result = np.zeros((height - kernel, length - kernel, 3))
    for heightIndex in range(height - kernel - 1):
        for lenIndex in range(length - kernel - 1):
            # Since we're dealing with a RGB image (3d array), we convolve each array separately
            subset1 = np.array(image)[heightIndex:heightIndex + kernel, lenIndex:lenIndex + kernel, 0]
            subset2 = np.array(image)[heightIndex:heightIndex + kernel, lenIndex:lenIndex + kernel, 1]
            subset3 = np.array(image)[heightIndex:heightIndex + kernel, lenIndex:lenIndex + kernel, 2]
            # I'm not sure why I have to assign the values in reverse, but it seems to fix issues so...
            result[heightIndex, lenIndex, 2] = np.sum(np.multiply(subset1, filter))
            result[heightIndex, lenIndex, 1] = np.sum(np.multiply(subset2, filter))
            result[heightIndex, lenIndex, 0] = np.sum(np.multiply(subset3, filter))

    return result


def convolution2(image, filter, dir):
    # Obtain image dimensions and filter dimension/Kernel
    dimensionsImg = image.shape
    dimensionsFltr = filter.shape
    height = dimensionsImg[0]
    length = dimensionsImg[1]
    kernelx = 1
    kernely = 1

    if dir == "x":
        kernelx = dimensionsFltr[0]
    else:
        kernely = dimensionsFltr[0]

    result = np.zeros((height - kernelx, length - kernely, 3))
    for heightIndex in range(height - max(kernelx, kernely) - 1):
        for lenIndex in range(length - max(kernelx, kernely) - 1):
            # Since we're dealing with a RGB image (3d array), we convolve each array separately
            subset1 = np.array(image)[heightIndex:heightIndex + kernely, lenIndex:lenIndex + kernelx, 0]
            subset2 = np.array(image)[heightIndex:heightIndex + kernely, lenIndex:lenIndex + kernelx, 1]
            subset3 = np.array(image)[heightIndex:heightIndex + kernely, lenIndex:lenIndex + kernelx, 2]

            result[heightIndex, lenIndex, 2] = np.sum(np.multiply(subset1, filter))
            result[heightIndex, lenIndex, 1] = np.sum(np.multiply(subset2, filter))
            result[heightIndex, lenIndex, 0] = np.sum(np.multiply(subset3, filter))

    return result


def HW1_1(name="Seattle.jpg"):
    img = cv2.imread(name)
    gFilter = gaussian(4)
    image_array = convolution1(img, gFilter)
    image = Image.fromarray(np.uint8(image_array))
    image.save("1.png")
    print("Completed 1_1")

def HW1_2(name="Seattle.jpg"):
    img = cv2.imread(name)
    [xG, yG] = separable_gaussian(4)
    xImage = convolution2(img, xG, "x")
    #yImage = convolution2(img, yG, "y") #
    #image = Image.fromarray(np.uint8(yImage)) #
    #image2 = Image.fromarray(np.uint8(xImage)) #
    #image.save("y.png") #
    #image2.save("x.png") #
    yImage = convolution2(xImage, yG, "y")
    dim = yImage.shape
    # i dont know if the program im using is fualty or what, but it keeps switching around some values when saving.
    # so im manually swithcing them back :/
    temp = np.zeros((dim[0], dim[1], dim[2]))
    temp[0:dim[0], 0:dim[1], 0] = yImage[0:dim[0], 0:dim[1], 0]
    yImage[0:dim[0]-1, 0:dim[1]-1, 0] = yImage[0:dim[0]-1, 0:dim[1]-1, 2]
    yImage[0:dim[0]-1, 0:dim[1]-1, 2] = temp[0:dim[0]-1, 0:dim[1]-1, 0]
    image = Image.fromarray(np.uint8(yImage))
    image.save("2.png")
    print("Completed 1_2")


#HW1_1("img.jpg")
HW1_2("img.jpg")
