import numpy as np
from numpy import asarray
from numpy import savetxt
# from PIL import Image
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
        yfilter[i, 0] = 1.0 / np.sqrt((2 * np.pi * sigma ** 2)) * np.exp(-(xy ** 2) / (2 * sigma ** 2))
    return xfilter, yfilter


def sobelfilter():
    gx = np.array([-1, 0, 1], [-2, 0, 2], [-1, 0, 1])
    gy = np.array([-1, -2, -1], [0, 0, 0], [-1, 2, 1])
    return gx, gy


def convolution1(image, filter):
    # Obtain image dimensions and filter dimension/Kernel
    dimensionsImg = image.shape
    dimensionsFltr = filter.shape
    height = dimensionsImg[0]
    length = dimensionsImg[1]
    kernel = dimensionsFltr[0]

    result = np.zeros((height - kernel + 1, length - kernel + 1, 3))
    for heightIndex in range(height - kernel + 1):
        for lenIndex in range(length - kernel + 1):
            # Since we're dealing with a RGB image (3d array), we convolve each array separately
            subsetR = np.array(image)[heightIndex:(heightIndex + kernel), lenIndex:(lenIndex + kernel), 0]
            subsetG = np.array(image)[heightIndex:(heightIndex + kernel), lenIndex:(lenIndex + kernel), 1]
            subsetB = np.array(image)[heightIndex:(heightIndex + kernel), lenIndex:(lenIndex + kernel), 2]
            # I'm not sure why I have to assign the values in reverse, but it seems to fix issues so...
            result[heightIndex, lenIndex, 0] = np.sum(np.multiply(subsetR, filter))
            result[heightIndex, lenIndex, 1] = np.sum(np.multiply(subsetG, filter))
            result[heightIndex, lenIndex, 2] = np.sum(np.multiply(subsetB, filter))

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

    result = np.zeros((height - kernely + 1, length - kernelx + 1, 3))
    d = result.shape
    for heightIndex in range(height - kernely + 1):
        for lenIndex in range(length - kernelx + 1):
            # Since we're dealing with a RGB image (3d array), we convolve each array separately
            subsetR = np.array(image)[heightIndex:(heightIndex + kernely), lenIndex:(lenIndex + kernelx), 0]
            subsetG = np.array(image)[heightIndex:(heightIndex + kernely), lenIndex:(lenIndex + kernelx), 1]
            subsetB = np.array(image)[heightIndex:(heightIndex + kernely), lenIndex:(lenIndex + kernelx), 2]

            result[heightIndex, lenIndex, 0] = np.sum(np.multiply(subsetR, filter))
            result[heightIndex, lenIndex, 1] = np.sum(np.multiply(subsetG, filter))
            result[heightIndex, lenIndex, 2] = np.sum(np.multiply(subsetB, filter))

    return result


def GaussianBlurImage(image, sigma):
    print("Running 1_1")
    img = cv2.imread(image)
    gFilter = gaussian(sigma)
    image_array = convolution1(img, gFilter)
    cv2.imwrite("1.png", image_array)
    print("Completed 1_1")


def SeparableGaussianBlurImage(image, sigma):
    print("Running 1_2")
    img = cv2.imread(image)
    [xG, yG] = separable_gaussian(sigma)
    xImage = convolution2(img, xG, "x")
    yImage = convolution2(xImage, yG, "y")
    cv2.imwrite("2.png", yImage)
    print("Completed 1_2")


def SharpenImage(image, sigma, alpha):
    print("Running 1_3")
    img = cv2.imread(image)
    gFilter = gaussian(sigma)
    dim = gFilter.shape  # need to know how much smaller the new image should be
    dimImg = img.shape
    blur = convolution1(img, gFilter)
    subset = img[(dim[0] // 2):(dimImg[0] - (dim[0] // 2)), (dim[1] // 2):(dimImg[1] - (dim[1] // 2))]
    sharpness = np.multiply(alpha, np.subtract(subset, blur))
    sharpImg = np.add(subset, sharpness)
    cv2.imwrite("3.png", sharpImg)
    print("Completed 1_3")


def SobelImage(image):
    print("Running 1_4")
    img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY)
    gx, gy = sobelfilter()
    imgGx = convolution1(img, gx)
    imgGy = convolution1(img, gy)
    magnitude = np.sqrt(np.square(imgGx), np.square(imgGx))
    magnitude *= 255.0 / magnitude.max()
    cv2.imwrite("5a.png", magnitude)
    cv2.imwrite("5b.png", np.arctan(np.divide(imgGy, imgGx)))
    print("Completed 1_4")


# def BilinearInterpolation(image, x, y):


def nearestNeighbor(image, scale):
    img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY)
    dim = img.shape
    result = np.zeros((dim[0] * scale, dim[1] * scale))
    for xImgIndex in range(dim[0]):
        for yImgIndex in range(dim[1]):
            for i in range(scale):
                for j in range(scale):
                    result[xImgIndex * scale + i, yImgIndex * scale + j] = img[xImgIndex, yImgIndex]


    cv2.imwrite("5a.png", result)

# GaussianBlurImage("Seattle.jpg", 4)  # This is working
SeparableGaussianBlurImage("Seattle.jpg", 4)  # This is working

# SharpenImage("Yosemite.png", 1, 5)  # This is also working
#nearestNeighbor("moire_small.jpg", 2)
