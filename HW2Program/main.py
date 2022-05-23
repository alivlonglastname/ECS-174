# ECS 174, Spring 2022
# Homework 2: Image stitching
import numpy as np
import cv2
import utils
import matplotlib.pyplot as plt
import imageio
import random
import csv

# algorithms and parameters
extraction_algorithm = 'sift'  # only sift is supported for now
feature_to_match = 'bf'  # only Brute force is supported for now
THRESHOLD = 0.6
NUM_ITERS = 1000


# *** Beginng of comon functions *************


def calc_Homography(pairs):
    A = []
    for x1, y1, x2, y2 in pairs:
        '''changed this'''
        A.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
        A.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])
    A = np.array(A)

    # Singular Value Decomposition (SVD)
    U, S, V = np.linalg.svd(A)
    H = np.reshape(V[-1], (3, 3))
    # V has shape (9, 9) for any number of input pairs. V[-1] is the eigenvector
    # of (A^T)A with the smalles eigenvalue. Reshape into 3x3 matrix.


    # Normalization
    '''changed this'''
    H = (1 / H.item(8)) * H
    return H


def dist(pair, H):
    # points in homogeneous coordinates
    p1 = np.array([pair[0], pair[1], 1])
    p2 = np.array([pair[2], pair[3], 1])

    p2_estimate = np.dot(H, np.transpose(p1))
    p2_estimate = (1 / p2_estimate[2]) * p2_estimate

    return np.linalg.norm(np.transpose(p2) - p2_estimate)


def RANSAC(point_map, threshold=THRESHOLD):
    bestInliers = set()
    homography = None
    for i in range(NUM_ITERS):
        # choose 4 random points from the matrix to calculate the homography
        pairs = [point_map[i] for i in np.random.choice(len(point_map), 4)]

        H = calc_Homography(pairs)

        inliers = {(c[0], c[1], c[2], c[3]) for c in point_map if dist(c, H) < 5}

        if len(inliers) > len(bestInliers):
            bestInliers = inliers
            homography = H

    return homography, bestInliers


def myfindHomography(points_right, points_left, method, reprojThresh):
    if method == 1:
        point_map = np.concatenate((points_right, points_left), axis=1)
        H, status = RANSAC(point_map)

    else:
        (H, status) = cv2.findHomography(points_right, points_left, cv2.RANSAC, reprojThresh)

    return H, status


def descriptor(img):
    descriptor = cv2.SIFT_create()
    (keypoints, features) = descriptor.detectAndCompute(img, None)

    return (keypoints, features)


def matching_object(method, crossCheck):
    "Create and return a Matcher Object"
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    return bf


def key_points_matching(features_right_img, features_left_img, method):
    bf = matching_object(method, crossCheck=True)
    # Match descriptors.

    best_matches = bf.match(features_right_img, features_left_img)

    # Sort the features in order of distance.
    # The points with small distance (more similarity) are ordered first in the vector
    rawMatches = sorted(best_matches, key=lambda x: x.distance)
    print("Brute force matches:", len(rawMatches))
    return rawMatches


def homography_mapping(keypoints_right_img, keypoints_left_img, matches, reprojThresh):
    keypoints_right_img = np.float32([keypoint.pt for keypoint in keypoints_right_img])
    keypoints_left_img = np.float32([keypoint.pt for keypoint in keypoints_left_img])

    if len(matches) > 4:
        # construct the two sets of points

        points_right = np.float32([keypoints_right_img[m.queryIdx] for m in matches])
        points_left = np.float32([keypoints_left_img[m.trainIdx] for m in matches])

        # Calculate the homography between the sets of points

        (H, status) = myfindHomography(points_right, points_left, 1, reprojThresh)

        return (matches, H, status)
    else:
        return None


# *** End of comon functions *************


# ***** Task 1: Load both images, convert to double and to grayscale. *****
imgLN = "uttower_left.JPG"
imgRN = "uttower_right.JPG"
left_photo = cv2.imread(imgLN)  # Load keft image
right_photo = cv2.imread(imgRN)  # Load right image
left_photo = cv2.cvtColor(left_photo, cv2.COLOR_BGR2RGB)  # convert to RGB for Matplotlib for ploting
right_photo = cv2.cvtColor(right_photo, cv2.COLOR_BGR2RGB)  # convert to RGB for Matplotlib for ploting
left_photo_gray = cv2.cvtColor(left_photo, cv2.COLOR_RGB2GRAY)  # convert left image to gray
right_photo_gray = cv2.cvtColor(right_photo, cv2.COLOR_RGB2GRAY)  # convert right image to gray
left_photo_gray_d = np.float32(left_photo_gray)  # convert left image to double
right_photo_gray_d = np.float32(right_photo_gray)  # convert right image to double

# Show orignal images before processing

fig, (ph1, ph2) = plt.subplots(nrows=1, ncols=2, constrained_layout=False, figsize=(16, 9))
ph1.imshow(left_photo, cmap="gray")
ph1.set_xlabel("Left image", fontsize=16)
ph2.imshow(right_photo, cmap="gray")
ph2.set_xlabel("Right image", fontsize=16)
fig.suptitle('Orignal images before processing (Press any key to skip)', fontsize=16)
plt.waitforbuttonpress(0)
plt.close(fig)



# ***** Task 2: Detect feature points in both images using Harris corner detection. ******
imgL = left_photo.copy()
imgR = right_photo.copy()
dst1 = cv2.cornerHarris(left_photo_gray_d, 2, 3, 0.04)  # Detect feature points for left image
dst1 = cv2.dilate(dst1, None)
imgL[dst1 > 0.01 * dst1.max()] = [0, 0, 255]

dst2 = cv2.cornerHarris(right_photo_gray_d, 2, 3, 0.04)  # Detect feature points for right image
dst2 = cv2.dilate(dst2, None)
imgR[dst2 > 0.01 * dst2.max()] = [0, 0, 255]

# Show images after Haris corner processing
fig, (ph1, ph2) = plt.subplots(nrows=1, ncols=2, constrained_layout=False, figsize=(16, 9))
ph1.imshow(imgL, cmap="gray")
ph1.set_xlabel("Left image Haris corners", fontsize=16)
ph2.imshow(imgR, cmap="gray")
ph2.set_xlabel("Right image Haris corners", fontsize=16)
fig.suptitle('Images after Haris corner algorithm processing (Press any key to skip)', fontsize=16)
plt.waitforbuttonpress(0)
plt.close(fig)

# ****** Task 3 & 4  *******
# Extract local neighborhoods around every keypoint in both images and
# Compute distances between every descriptor in one image and every descriptor in the other image using SIFT

keypoints_right_img, features_right_img = descriptor(right_photo_gray)
keypoints_left_img, features_left_img = descriptor(left_photo_gray)

for keypoint in keypoints_left_img:
    x, y = keypoint.pt
    size = keypoint.size
    orientation = keypoint.angle
    response = keypoint.response
    octave = keypoint.octave
    class_id = keypoint.class_id

features_left_img.shape

# Show keypoints and features detected on both images
fig, (ph1, ph2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 8), constrained_layout=False)
ph1.imshow(cv2.drawKeypoints(right_photo_gray, keypoints_right_img, None, color=(0, 255, 0)))
ph1.set_xlabel("detected features on left image", fontsize=14)
ph2.imshow(cv2.drawKeypoints(left_photo_gray, keypoints_left_img, None, color=(0, 255, 0)))
ph2.set_xlabel("detected features on right image", fontsize=14)
fig.suptitle('Images after SIFT feature detection processing (Press any key to skip)', fontsize=16)
plt.waitforbuttonpress(0)
plt.close(fig)

# ****** Task 5 *******
# Select putative matches based on the matrix of pairwise descriptor distances obtained above.

# print("Drawing: {} matched features Lines".format(feature_to_match))

matches = key_points_matching(features_right_img, features_left_img,
                              method=extraction_algorithm)  # using SIFT algorithm
mapped_features_image = cv2.drawMatches(right_photo, keypoints_right_img, left_photo, keypoints_left_img, matches[:100],
                                        None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

matches = key_points_matching(features_right_img, features_left_img, method=extraction_algorithm)
mapped_features_image = cv2.drawMatches(right_photo, keypoints_right_img, left_photo, keypoints_left_img, matches[:100],
                                        None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# ***** Task 6: Run RANSAC to estimate a homography mapping one image onto the other. ******
# for details of homography mapping and RANSAC implementation, please see above common functions section

M = homography_mapping(keypoints_right_img, keypoints_left_img, matches, reprojThresh=4)
(matches, Homography_Matrix, status) = M

# display the locations of inlier matches in both images.
plt.figure(figsize=(20,10))
fig = plt.figure(figsize=(20, 10))
plt.axis('off')
plt.imshow(mapped_features_image)
fig.suptitle('locations of inlier matches in both images (Press any key to skip)', fontsize=16)
plt.waitforbuttonpress(0)
plt.close(fig)

# ***** Task 7 & 8:  ******
# Warp one image onto the other using the estimated transformation.
# Create a new image big enough to hold the panorama and composite the two images into it

width = left_photo.shape[1] + right_photo.shape[1]
# print("width ", width)


height = max(left_photo.shape[0], right_photo.shape[0])

# apply a perspective warp to stitch the images together

result = cv2.warpPerspective(right_photo, Homography_Matrix, (width, height))

result[0:left_photo.shape[0], 0:left_photo.shape[1]] = left_photo

# show the stiched image

fig = plt.figure(figsize=(20, 10))
cv2.imwrite("Result.png", result)
plt.axis('off')
plt.imshow(result)
fig.suptitle('Final stitched image (Saved in local dir as Result.png)', fontsize=16)
plt.show()










