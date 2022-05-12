import numpy as np
from numpy import asarray
from numpy import savetxt
import os
from os.path import exists
import cv2
import math
import time

# Create result directory
path_to_script = os.path.dirname(os.path.abspath(__file__))
result_path = os.path.join(path_to_script, "Result")
if not (exists(result_path)):
    os.mkdir(result_path)


def gaussian(sigma):
    # took this code from course website
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
    # creates 2 filters, one along x axis and one along y axis. took it from lecture slides
    filter_size = 2 * int(sigma * 4 + 0.5) + 1
    xfilter = np.zeros((filter_size), dtype=np.float32)
    yfilter = np.zeros((filter_size, 1), dtype=np.float32)
    for i in range(filter_size):
        xy = i - filter_size // 2
        xfilter[i] = 1.0 / np.sqrt((2 * np.pi * sigma ** 2)) * np.exp(-(xy ** 2) / (2 * sigma ** 2))
        yfilter[i, 0] = 1.0 / np.sqrt((2 * np.pi * sigma ** 2)) * np.exp(-(xy ** 2) / (2 * sigma ** 2))
    return xfilter, yfilter


def sobelfilter():
    # took this from lecture slides
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
    # initialize result array
    result = np.zeros((height - kernel + 1, length - kernel + 1, 3))
    for heightIndex in range(height - kernel + 1):
        for lenIndex in range(length - kernel + 1):
            # Since we're dealing with a RGB image (3d array), we convolve each array separately
            subsetR = np.array(image)[heightIndex:(heightIndex + kernel), lenIndex:(lenIndex + kernel), 0]
            subsetG = np.array(image)[heightIndex:(heightIndex + kernel), lenIndex:(lenIndex + kernel), 1]
            subsetB = np.array(image)[heightIndex:(heightIndex + kernel), lenIndex:(lenIndex + kernel), 2]
            result[heightIndex, lenIndex, 0] = np.sum(np.multiply(subsetR, filter))
            result[heightIndex, lenIndex, 1] = np.sum(np.multiply(subsetG, filter))
            result[heightIndex, lenIndex, 2] = np.sum(np.multiply(subsetB, filter))

    return result


# second convolution function to deal with non square filter matrices. I initially was editing the original function to
# be able to deal with both. But it took too long so I just made  a new one
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

    # convolution3 is used for grayscale images


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
    img = cv2.imread(image)
    gFilter = gaussian(sigma)
    image_array = convolution1(img, gFilter)
    return image_array


def SeparableGaussianBlurImage(image, sigma):
    img = cv2.imread(image)
    [xG, yG] = separable_gaussian(sigma)
    # saving to file and reloading dramatically improves runtime speed
    cv2.imwrite("2.png", convolution2(img, xG, "x"))
    img = cv2.imread("2.png")
    yImage = convolution2(img, yG, "y")

    return yImage


def SharpenImage(image, sigma, alpha):
    img = cv2.imread(image)
    gFilter = gaussian(sigma)
    dim = gFilter.shape  # need to know how much smaller the new image should be
    dimImg = img.shape
    blur = convolution1(img, gFilter)
    # subset has the new smaller image
    subset = img[(dim[0] // 2):(dimImg[0] - (dim[0] // 2)), (dim[1] // 2):(dimImg[1] - (dim[1] // 2))]
    # result = image + (alpha * (image - blur))
    sharpness = np.multiply(alpha, np.subtract(subset, blur))
    sharpImg = np.add(subset, sharpness)

    return sharpImg


def sobelImage(image):
    # convert to greyscale
    img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY)
    gx, gy = sobelfilter()
    # convolve with each filter
    imgGx = convolution3(img, gx)
    imgGy = convolution3(img, gy)
    # r = sqrt(x^2 + y^2)
    magnitude = np.sqrt(np.square(imgGx), np.square(imgGx))
    magnitude *= 255.0 / magnitude.max()
    direction = np.arctan(np.divide(imgGy, imgGx))

    return direction, magnitude


def BilinearInterpolation(image, x, y):
    # taken from lecture slides
    # this seperates a decimal like 1.2 into 1 and 0.2 (x.a into x and a)
    a = x - int(x)
    b = y - int(y)
    x -= a
    y -= b
    x = int(x)
    y = int(y)
    val = 0

    # indexing is hard here but ultimately not very important because python is nice :)
    try:
        val = image[x, y] * (1 - a) * (1 - b) + a * (1 - b) * image[x + 1, y] + (1 - a) * b * image[x, y + 1] + a * b * \
              image[x + 1, y + 1]
    except IndexError:
        pass
    return val


def upscale(image, scale):
    # upscales an image using bilinear interpolation
    img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY)
    dim = img.shape
    result = np.zeros((dim[0] * scale, dim[1] * scale))
    for x in range(dim[0] * scale):
        for y in range(dim[1] * scale):
            result[x, y] = BilinearInterpolation(img, x / scale, y / scale)

    return result


def FindPeaksImage(image, thresh):
    direction, magnitude = sobelImage(image)
    dim = direction.shape
    result = np.zeros((dim[0], dim[1]))
    for i in range(dim[0]):
        for j in range(dim[1]):
            # sometimes the values are just NULL, because there is no angle its a solid area.
            # so in that case just set it to 0
            if math.isnan(direction[i, j]):
                result[i, j] = 0
                continue
            # using math to find a and b (which are decimals)
            a = np.cos(direction[i, j])
            b = np.sin(direction[i, j])
            # within a value of a or b
            xp = i + a
            yp = j + b
            xn = i - a
            yn = j - b
            e0 = BilinearInterpolation(magnitude, xp, yp)
            e1 = BilinearInterpolation(magnitude, xn, yn)

            if magnitude[i, j] > e0 and magnitude[i, j] > e1 and magnitude[i, j] > thresh:
                result[i, j] = 255
            else:
                result[i, j] = 0

    return result


def nearestNeighbor(image, scale):
    # upscaling using nearest neighbor. kind of self explanatory?
    img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY)
    dim = img.shape
    result = np.zeros((dim[0] * scale, dim[1] * scale))
    for xImgIndex in range(dim[0]):
        for yImgIndex in range(dim[1]):
            for i in range(scale):
                for j in range(scale):
                    result[xImgIndex * scale + i, yImgIndex * scale + j] = img[xImgIndex, yImgIndex]

    return result


# run all will run all the homework questions 1 through 6. it will update the user when each is done.
# and it will save the results in a file called result created in the local program directory
def runAll():
    print("Running 1.1...")
    out = GaussianBlurImage("Seattle.jpg", 4)  # This is working
    cv2.imwrite("./Result/1.png", out)
    print("Done!")

    print("Running 1.2...")
    out = SeparableGaussianBlurImage("Seattle.jpg", 8)  # This is working
    cv2.imwrite("./Result/2.png", out)
    print("Done!")

    print("Running 1.3...")
    out = SharpenImage("Yosemite.png", 1, 5)  # This is also working
    cv2.imwrite("./Result/4.png", out)
    print("Done!")

    print("Running 1.4...")
    dir, mag = sobelImage("LadyBug.jpg")  # this is working
    dir = dir - dir.min()  # Now between 0 and 8674
    dir = dir / dir.max() * 255
    orientation = cv2.applyColorMap(np.uint8(dir), cv2.COLORMAP_JET)
    cv2.imwrite("./Result/5a.png", mag)
    cv2.imwrite("./Result/5b.png", orientation)
    print("Done!")

    print("Running 1.5...")
    out1 = nearestNeighbor("moire_small.jpg", 4)  # this is working
    out2 = upscale("moire_small.jpg", 4)  # this is working (bilinear interpolation)
    cv2.imwrite("./Result/6b.png", out2)
    cv2.imwrite("./Result/6a.png", out1)
    print("Done!")

    print("Running 1.6...")
    out = FindPeaksImage("Circle.png", 40.0)
    cv2.imwrite("./Result/7.png", out)
    print("Done!")


runAll()
