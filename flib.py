# General system level packages
import sys
import os
import time
import glob
# For numerical and image processing
import numpy as np
import cv2
# For plotting
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# For image processing
from skimage.feature import hog
from scipy.ndimage.measurements import label
# For machine learning tasks such as standardization, splitting and linear SVM classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
# For processing video files
from moviepy.editor import VideoFileClip


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Function to draw boxes on an image using cv2.rectangle()
    # Inputs: image, list of bounding boxes, color tuple, line thickness
    # Output: image with the bboxes drawn on it
    # draws boxes using cv2.rectangle() in that color on the output
    imcopy = np.copy(img)
    for box in bboxes:
        cv2.rectangle(imcopy, box[0], box[1], color=color, thickness=thick)
    return imcopy


def color_hist(img, nbins=32):
    # Computes the histogram of each color channel, concatenate them together and
    # return the data as features for classifier
    # Inputs: image and number of bins for the histogram

    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)

    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    # Histogram function returns a tuple of two arrays.
    # first element contains the counts in each of the bins
    # second element contains the bin edges (it is one element longer than first one)
    # Generate bin centers - we can prefer to use any channel
    bin_edges = channel1_hist[1]
    bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges)-1])/2

    # Return the individual histograms, bin_centers and feature vector
    return hist_features, bin_centers, channel1_hist, channel2_hist, channel3_hist


def bin_spatial(img, size=(32, 32)):
    # Spatial binning of the image to do downsampling
    # Inputs: image and desired size
    # Output: Binned image features
    features = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST).ravel()
    # Return the feature vector
    return features


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # Function that return HOG features
    # HOG is robust to variations in shape while keeping the signature distinct enough
    # It computes the gradient of the image and creates histograms based on the cells defined
    # Histograms are constructed based on the weighted count of gradient directions in each cell
    # Here weight is the magnitude of the gradient

    # The scikit-image hog() function takes in a single color channel or grayscaled image as input,
    # as well as various other parameters.
    # These parameters include orientations, pixels_per_cell and cells_per_block.
    # The number of orientations is specified as an integer, and represents the number of orientation
    #         bins that the gradient information will be split up into in the histogram.
    #         Typical values are between 6 and 12 bins.
    # The pixels_per_cell parameter specifies the cell size over which each gradient histogram is computed.
    #         This paramater is passed as a 2-tuple so you could have different cell sizes in x and y,
    #         but cells are commonlychosen to be square.
    # The cells_per_block parameter is also passed as a 2-tuple, and specifies the local area over which the
    #         histogram counts in a given cell will be normalized. Block normalization is not necessarily
    #         required, but generally leads to a more robust feature set.

    # When we specify feature_vec=True then we will get an the feature vectors calculated for the specified
    # parameters

    # Return two outputs if visualization is True
    if vis:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Return only features when visualization is False
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features


def cs_convert_from_rgb(img, color_space='RGB'):
    # Function that converts RGB image to a desired color space
    # Options are HSV, LUV, HLS, YUV and YCrCb
    if color_space != 'RGB':
        if color_space == 'HSV':
            new_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            new_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            new_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            new_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            new_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        new_image = np.copy(img)
    return new_image


def extract_features(imgs, color_space='RGB', spatial_size=(32, 32), hist_bins=32,
                     orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    # Function that extract features from images using color histogram, binned spatial and HOG features
    # This function gets all the parameters that are used in the individual feature extraction functions
    # Calls color_hist(), bin_spatial(), get_hog_features() functions based on the parameters specified (True/False)
    # Using the same ordering of the features is important for training and testing
    # Need to provide a list of image paths and the function will loop through them
    # Each image will have a 1-D array of features. Function will append this array to features list and return it

    # Create a list to append feature vectors to
    features = []

    # Iterate through the list of images
    for file in imgs:

        file_features = []

        # Read in each one by one
        image = mpimg.imread(file)

        # apply color conversion if other than 'RGB'
        feature_image = cs_convert_from_rgb(image, color_space)

        if spatial_feat == True:
            # Get binned spatial features and update file_feature list
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)

        if hist_feat == True:
            # Get color histogram featueres and update file_feature list
            hist_features, _, _, _, _ = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)

        if hog_feat == True:
            # Get HOG features for desired channels and update file_feature list
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                                                pix_per_cell, cell_per_block,
                                                vis=False, feature_vec=True)
            file_features.append(hog_features)

        # Update main feature list by 1-D file features
        features.append(np.concatenate(file_features))

    # Return list of feature vectors
    return features


def single_img_features(img, color_space='RGB', spatial_size=(32, 32), hist_bins=32,
                        orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True, vis=False):
    # This function is same as the extract_features function but does the process for a single image
    # The image input is an image instead of list of image paths
    # This function will be used for windows that we extract from single images to classify if there is
    # a car in the window or not

    # Create a list to append feature vectors to
    img_features = []

    # apply color conversion if other than 'RGB'
    feature_image = cs_convert_from_rgb(img, color_space)

    if spatial_feat == True:
        # Get binned spatial features and update file_feature list
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        img_features.append(spatial_features)

    if hist_feat == True:
        # Get color histogram featueres and update file_feature list
        hist_features, _, _, _, _ = color_hist(feature_image, nbins=hist_bins)
        img_features.append(hist_features)

    if hog_feat == True:
        # Get HOG features for desired channels and update file_feature list
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            if vis == True:
                hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient,
                                                           pix_per_cell, cell_per_block,
                                                           vis=True, feature_vec=True)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                                                pix_per_cell, cell_per_block,
                                                vis=False, feature_vec=True)
        img_features.append(hog_features)

    # Return list of feature vectors
    if vis == True:
        return np.concatenate(img_features), hog_image
    else:
        return np.concatenate(img_features)


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):

    # Function that returns sliding window positions
    # Inputs: image, x and y starting and stopping positions, window size and fraction of overlap
    # Output: Window positions for the desired area of the image

    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]

    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]

    # Compute the span of the region to be searched
    x_span = x_start_stop[1] - x_start_stop[0]
    y_span = y_start_stop[1] - y_start_stop[0]

    # Compute the number of pixels per step in x/y
    x_pixels_per_step = np.int(xy_window[0] * (1.0 - xy_overlap[0]))
    y_pixels_per_step = np.int(xy_window[1] * (1.0 - xy_overlap[1]))

    # Compute the number of windows in x/y
    x_window_count = np.int(x_span / x_pixels_per_step) - 1
    y_window_count = np.int(y_span / y_pixels_per_step) - 1

    # Initialize a list to append window positions to
    window_list = []

    # Loop through finding x and y window positions
    for ypos in range(y_window_count):
        for xpos in range(x_window_count):

            # Calculate each window position
            window = ((xpos * x_pixels_per_step + x_start_stop[0],
                       ypos * y_pixels_per_step + y_start_stop[0]),
                      (xpos * x_pixels_per_step + x_start_stop[0] + xy_window[0],
                       ypos * y_pixels_per_step + y_start_stop[0] + xy_window[1]))

            # Append window position to list
            window_list.append(window)

    # Return the list of windows
    return window_list


def search_windows(img, windows, clf, scaler,
                   color_space='RGB', spatial_size=(32, 32), hist_bins=32,
                   orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
                   spatial_feat=True, hist_feat=True, hog_feat=True):

    # Function that gets an image and list of windows (coordinates)
    # Loops through the windows by cropping the image and resizing them to (64, 64) - same size as training images
    # Inputs: image, window list, classifier, normalization scaler, feature extraction parameters
    # Output: list of windows with cars in them

    # List for positive detecion windows - there is a car
    on_windows = []

    # Iterate over all windows in the list
    for window in windows:

        # Extract the test window from original image and resize it to 64x64 - same as training images
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))

        # Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, spatial_size=spatial_size,
                                       hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block, hog_channel=hog_channel,
                                       spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

        # Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))

        # Predict using your classifier
        prediction = clf.predict(test_features)

        # If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)

    # Return windows for positive detections
    return on_windows



def find_cars(img, scale, x_start_stop, y_start_stop, clf, scaler,
              orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space):
    # Function that efficiciently loops through windows and classify them
    # The benefit of this function is that you don't have to call HOG features function all the time
    # It gets the features once at the beginning for the entire image and then subsets the related window area
    # This way the computation time decreases significantly
    # The scale parameters scales the image so that window size does not have to be changed

    #####
    # NOTE: This function is only useful if you included HOG features during training
    ######

    # output image
    draw_img = np.copy(img)

    # Make a heatmap of zeros
    heatmap = np.zeros_like(img[:,:,0])

    # Convert the image pixel value range so that jpg images are same as png images - training was done on png
    img = img.astype(np.float32)/255

    # Using desired start/stop positions we can reduce the image size that needs to be searched
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]

    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]

    # Get the desired section of the image based on the start/stop definitions
    search_img = img[y_start_stop[0]:y_start_stop[1], x_start_stop[0]:x_start_stop[1], :]

    # Convert it to the desired color space
    search_img_color = cs_convert_from_rgb(search_img, color_space=color_space)

    # Scale the image if a scale value other than 1 is defined.
    # Rather than selecting different window sizes, we scale the image so that window size stays same
    # but spatial information corresponding to the window changes due to scaling of the image
    if scale != 1:
        img_shape = search_img_color.shape
        search_img_color = cv2.resize(search_img_color, (np.int(img_shape[1]/scale),
                                                        np.int(img_shape[0]/scale)))


    # Let's get each channel as a separate array
    ch1 = search_img_color[:,:,0]
    ch2 = search_img_color[:,:,1]
    ch3 = search_img_color[:,:,2]

    # Now since we will use the raw HOG features and subset depending on the window position
    # We need to define the parameters around getting increments

    # Number of blocks in x and y
    nx_blocks = (ch1.shape[1] // pix_per_cell) - 1
    ny_blocks = (ch1.shape[0] // pix_per_cell) - 1

    # Number of features per block
    nfeat_per_block = orient*cell_per_block**2

    # Window size
    window_px = 64

    # Number of blocks per window
    nblocks_per_window = (window_px // pix_per_cell) - 1

    # Cells per step
    # Instead of overlap, define how many cells to step
    # with pix_per_cell = 8 and cells_per_step we get 0.75 overlap
    cells_per_step = 2

    # Compute the steps in x and y directions
    # How many steps we will do across HOG array to extract features
    nxsteps = (nx_blocks - nblocks_per_window) // cells_per_step
    nysteps = (ny_blocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch2, orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):

            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            # coordinates of the top left corner
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(search_img_color[ytop:ytop+window_px, xleft:xleft+window_px],
                                (64, 64)) # we use 64x64 since training data is of this size

            # Get color features
            hist_features, _, _, _, _ = color_hist(subimg, nbins=hist_bins)

            # Get spatial features
            spatial_features = bin_spatial(subimg, size=spatial_size)

            # Combine the features and scale it with the scaler
            test_features = scaler.transform(np.hstack((spatial_features, hist_features,
                                                        hog_features)).reshape(1, -1))
            # print(test_features.shape)

            # Make prediction
            test_prediction = clf.predict(test_features)

            # If the window has a car in it then draw it on the original image
            # Also add heat to the heat map
            if test_prediction == 1:
                # Use the scale
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window_px * scale)
                cv2.rectangle(draw_img, (xbox_left + x_start_stop[0], ytop_draw + y_start_stop[0]),
                             (xbox_left + win_draw + x_start_stop[0], ytop_draw + win_draw + y_start_stop[0]),
                             (0, 0, 255), thickness=6)
                # img_boxes.append(((xbox_left + x_start_stop[0], ytop_draw + y_start_stop[0]),
                #                   (xbox_left + win_draw + x_start_stop[0], ytop_draw + win_draw + y_start_stop[0])))
                heatmap[ytop_draw + y_start_stop[0]:ytop_draw + win_draw + y_start_stop[0],
                       xbox_left + x_start_stop[0]:xbox_left + win_draw + x_start_stop[0]] += 1

    # Return the image and heatmap
    return draw_img, heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    # Reduces the false positives by removing some low value regions from the heatmap
    heatmap[heatmap <= threshold] = 0
    return heatmap


def draw_labeled_bboxes(img, labels):

    # Iterate through all detected cars and draw boxes around them

    for car_number in range(1, labels[1] +1):
        # Find pixels with each car number label value
        nonzero = (labels[0] == car_number).nonzero()

        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Define the bounding box based on min/max value of non-zero x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)

    # Return the image
    return img


def visualize(fig, rows, cols, imgs, titles):

    # A small function to visualize some results

    for i, img in enumerate(imgs):
        plt.subplot(rows, cols, i+1)
        plt.title(i+1)
        img_dim = len(img.shape)
        if img_dim < 3:
            plt.imshow(img, cmap='hot')
            plt.title(titles[i])
        else:
            plt.imshow(img)
            plt.title(titles[i])


def color_histogram(color_space, bin_ctr, chn1, chn2, chn3):

    # A function to visualize histogram of different color channels

    fig = plt.figure(figsize=(12, 3))

    plt.subplot(131)
    plt.bar(bin_ctr, chn1[0])
    plt.title(color_space[0] + ' Channel')

    plt.subplot(132)
    plt.bar(bin_ctr, chn2[0])
    plt.title(color_space[1] + ' Channel')

    plt.subplot(133)
    plt.bar(bin_ctr, chn3[0])
    plt.title(color_space[2] + ' Channel')

    fig.tight_layout()


def process_video(input_video_path, out_path, process_func):

    # A function to process video images

    clip = VideoFileClip(input_video_path)
    test_clip = clip.fl_image(process_func)
    test_clip.write_videofile(out_path, audio=False)
