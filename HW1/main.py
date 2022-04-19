import numpy as np
from numpy import asarray
from numpy import savetxt
import os
from os.path import exists
import cv2
import math


# Create result directory
path_to_script = os.path.dirname(os.path.abspath(__file__))
result_path = os.path.join(path_to_script, "Result")
if not (exists(result_path)):
    os.mkdir(result_path)


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
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gy = np.array([[-1, -2, -1], [0, 0, 0], [-1, 2, 1]])
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


def convolution3(image, filter):
    # Obtain image dimensions and filter dimension/Kernel
    dimensionsImg = image.shape
    dimensionsFltr = filter.shape
    height = dimensionsImg[0]
    length = dimensionsImg[1]
    kernel = dimensionsFltr[0]

    result = np.zeros((height - kernel + 1, length - kernel + 1))
    for heightIndex in range(height - kernel + 1):
        for lenIndex in range(length - kernel + 1):
            # Since we're dealing with a RGB image (3d array), we convolve each array separately
            subsetR = np.array(image)[heightIndex:(heightIndex + kernel), lenIndex:(lenIndex + kernel)]
            # I'm not sure why I have to assign the values in reverse, but it seems to fix issues so...
            result[heightIndex, lenIndex] = np.sum(np.multiply(subsetR, filter))

    return result


def GaussianBlurImage(image, sigma):
    print("Running 1_1")
    img = cv2.imread(image)
    gFilter = gaussian(sigma)
    image_array = convolution1(img, gFilter)
    cv2.imwrite("./Result/1.png", image_array)
    print("Completed 1_1")


def SeparableGaussianBlurImage(image, sigma):
    print("Running 1_2")
    img = cv2.imread(image)
    [xG, yG] = separable_gaussian(sigma)
    # saving to file and reloading dramatically imrpoves runtime speed
    cv2.imwrite("2.png", convolution2(img, xG, "x"))
    img = cv2.imread("2.png")
    yImage = convolution2(img, yG, "y")
    cv2.imwrite("./Result/2.png", yImage)
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
    cv2.imwrite("./Result/4.png", sharpImg)
    print("Completed 1_3")


def sobelImage(image, save):
    print("Running 1_4")
    img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY)
    gx, gy = sobelfilter()
    imgGx = convolution3(img, gx)
    imgGy = convolution3(img, gy)
    magnitude = np.sqrt(np.square(imgGx), np.square(imgGx))
    magnitude *= 255.0 / magnitude.max()
    direction = np.arctan(np.divide(imgGy, imgGx))
    directionR = np.arctan(np.divide(imgGy, imgGx))
    direction = direction - direction.min()  # Now between 0 and 8674
    direction = direction / direction.max() * 255
    orientation = cv2.applyColorMap(np.uint8(direction), cv2.COLORMAP_JET)
    if save:
        cv2.imwrite("./Result/5a.png", magnitude)
        cv2.imwrite("./Result/5b.png", orientation)
    print("Completed 1_4")
    return directionR, magnitude


def BilinearInterpolation(image, x, y):


    a = x - int(x)
    b = y - int(y)
    x -= a
    y -= b
    x = int(x)
    y = int(y)

    val = 0

    try:
        val = image[x,y] * (1 - a) * (1 - b) + a * (1 - b) * image[x+1, y] + (1 - a) * b * image[x, y+1] + a * b * image[x+1, y+1]
    except IndexError:
        pass
    return val


def upscale(image, scale):
    print("Running 1_5b")
    img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY)
    dim = img.shape
    result = np.zeros((dim[0] * scale, dim[1] * scale))
    for x in range(dim[0] * scale):
        for y in range(dim[1] * scale):
            result[x, y] = BilinearInterpolation(img, x / scale, y / scale)

    cv2.imwrite("./Result/6b.png", result)
    print("Completed 1_5b")


def FindPeaksImage(image, thresh):
    img = cv2.imread(image)
    direction, magnitude = sobelImage(image, False)
    dim = direction.shape
    result = np.zeros((dim[0], dim[1]))
    for i in range(dim[0]):
        for j in range(dim[1]):
            if math.isnan(direction[i,j]):
                result[i,j] = 0
                continue
            a = np.cos(direction[i,j])
            b = np.sin(direction[i,j])
            xp = i + a
            yp = j + b
            xn = i - a
            yn = j - b
            e0 = BilinearInterpolation(magnitude, xp, yp)
            e1 = BilinearInterpolation(magnitude, xn, yn)

            if magnitude[i, j] > e0 and magnitude[i, j] > e1 and magnitude[i, j] > thresh:
                result[i,j] = 255
            else:
                result[i,j] = 0

    cv2.imwrite("./Result/7.png", result)

def nearestNeighbor(image, scale):
    print("Running 1_5a")
    img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY)
    dim = img.shape
    result = np.zeros((dim[0] * scale, dim[1] * scale))
    for xImgIndex in range(dim[0]):
        for yImgIndex in range(dim[1]):
            for i in range(scale):
                for j in range(scale):
                    result[xImgIndex * scale + i, yImgIndex * scale + j] = img[xImgIndex, yImgIndex]


    cv2.imwrite("./Result/6a.png", result)
    print("Completed 1_5a")

#GaussianBlurImage("Seattle.jpg", 4)  # This is working
#SeparableGaussianBlurImage("Seattle.jpg", 4)  # This is working
#SharpenImage("Yosemite.png", 1, 5)  # This is also working
#sobelImage("LadyBug.jpg", True)  # this is working
#nearestNeighbor("moire_small.jpg", 4)  # this is working
#upscale("moire_small.jpg", 4)  # this is working (bilinear interpolation)
FindPeaksImage("Circle.png", 40.0)