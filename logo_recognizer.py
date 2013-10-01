#! /usr/bin/env python

"""This is a logo recognizer application for detecting logos in
official documentation.

Each candidate to logo is stored in its own folder:
<Regions_output_path>/<Image_name>/...  Additional intermediate files
are also stored for debugging purposes.

It uses the feature rectangles concept from
http://www.cvc.uab.es/icdar2009/papers/3725b335.pdf with some
heuristics (position of the logo in the document and vertical profile
of the image)
    
TODO: Refactor this code to be able to use it as an independent
module.

"""

import cv2
import numpy as np
from scipy.spatial import cKDTree
import os
import argparse
from multiprocessing import Process

def longest_increasing_run(blank_lines):
    """This method computes the longest increasing run in a list. That run
is the largest gap and potential separator between logo and non-logo
elements

    """
    start     = 0
    end       = 0
    maxStart  = 0
    maxEnd    = 0
    maxLength = 1

    for idx in range(1, len(blank_lines)):
        if blank_lines[idx] - blank_lines[idx-1] == 1:
            end = end + 1
        else:
            # Check if it is the longest so far
            if end - start + 1 > maxLength:
                maxLength = end - start + 1
                maxStart  = start
                maxEnd    = end
            start = end = idx
        if end - start + 1 > maxLength:
            maxLength = end - start + 1
            maxStart  = start
            maxEnd    = end
    return blank_lines[maxStart], blank_lines[maxEnd]
        
def get_boundaries(img, color_img_blanks):
    """This function calculates the upper and lower boundary of text in
the document."""
    img_height = np.shape(img)[0]
    blank_lines = []
    first_non_blank = False
    upper_boundary = 0
    lower_boundary = 0

    for idx in range(img_height/2):
        line = img[idx,:]
        if len(np.where(line == 0)[0]) >= 50 and len(np.where(line == 0)[0]) < 0.1 * len(line):
            first_non_blank = True
        if len(np.where(line == 0)[0]) < 50 and first_non_blank:
            blank_lines.append(idx)
            cv2.line(color_img_blanks, (0, idx), (np.shape(img)[1], idx), (0,255,0), 1)
            
    first_non_blank = False
    for idx in reversed(range(img_height/2, img_height)):
        line = img[idx,:]
        if len(np.where(line == 0)[0]) >= 50 and len(np.where(line == 0)[0]) < 0.1 * len(line):
            first_non_blank = True
        if len(np.where(line == 0)[0]) < 50 and first_non_blank:
            blank_lines.append(idx)
            cv2.line(color_img_blanks, (0, idx), (np.shape(img)[1], idx), (0,255,0), 1)

    upper_blank_lines = [item for item in blank_lines if item < img_height/2]
    if len(upper_blank_lines) > 0:
        upper_boundary = longest_increasing_run(upper_blank_lines)[0]
    
    lower_blank_lines = [item for item in blank_lines if item >= img_height/2]
    if len(lower_blank_lines) > 0:
        lower_boundary = longest_increasing_run(lower_blank_lines)[0]

    return upper_boundary, lower_boundary

def circle_around(seeds, center, radius):
    """This function returns points from center present in seeds,
    following a clockwise pattern of a given radius.
    
    """
    x, y = center[1], center[0]
    r = 1
    i, j = x-1, y-1
    while r < radius:
        while i < x+r:
            i += 1
            if (i,j) in seeds:
                yield (i, j)
        while j < y+r:
            j += 1
            if (i,j) in seeds:
                yield (i, j)
        while i > x-r:
            i -= 1
            if (i,j) in seeds:
                yield (i, j)
        while j > y-r:
            j -= 1
            if (i,j) in seeds:
                yield (i, j)
        r += 1
        j -= 1
        if (i,j) in seeds:
            yield (i, j)

def calculate_seeds(img):
    """This function scans the image from top to bottom, left to right
    and returns a dictionary of foreground pixels where the keys are
    the pixel coordinates and the values are 1 or 0 depending whether
    the pixel has been visited or not.

    """
    foreground = np.where(img == 0)
    foreground_length = len(foreground[0])
    pixels = zip(foreground[0], foreground[1])
    pixels_visited = zip(pixels, np.zeros(foreground_length))
    return dict(pixels_visited)

def valid_pixel(pixel, height, width, upper_boundary = 200, lower_boundary = 200):
    """ This function validates a pixel to be within the image
    boundaries."""
    if pixel[1] <= 0 or pixel[1] >= width - 1:
        return False
    if pixel[0] <= 0 or pixel[0] >= height - 1:
        return False
    if upper_boundary == -1 or lower_boundary == -1:
        return True
    return pixel[0] < upper_boundary or pixel[0] > lower_boundary

def compute_vertical_look_ahead(region):
    """This function computes the amount of vertical forward checking we
    are going to perform to the region received as parameter

    """
    return (np.shape(region)[0] / np.shape(region)[1] * 10)

def compute_horizontal_look_ahead(region):
    """This function computes the amount of horizontal forward checking we
    are going to perform to the region received as parameter

    """
    return (np.shape(region)[1] / np.shape(region)[0] * 10)

def expand_right_border(img, seeds, top_right, bottom_right, right_border):
    """This function expands a particular border by one pixel.

    """
    top_right[1]+=1
    bottom_right[1]+=1
    # Set border pixels as visited
    right_border = img[top_right[0]:bottom_right[0], top_right[1]]
    for idx, pixel in enumerate(right_border):
        if (idx + top_right[0], top_right[1]) in seeds:
            seeds[idx + top_right[0], top_right[1]] = 1

def expand_left_border(img, seeds, top_left, bottom_left, left_border):
    """This function expands a particular border by one pixel.

    """
    top_left[1]-=1
    bottom_left[1]-=1
    # Set border pixels as visited
    left_border = img[top_left[0]:bottom_left[0], top_left[1]]
    for idx, pixel in enumerate(left_border):
        if (idx + top_left[0], top_left[1]) in seeds:
            seeds[idx + top_left[0], top_left[1]] = 1

def expand_bottom_border(img, seeds, bottom_left, bottom_right, bottom_border):
    """This function expands a particular border by one pixel.

    """
    bottom_left[0]+=1
    bottom_right[0]+=1
    # Set border pixels as visited
    bottom_border = img[bottom_left[0], bottom_left[1]:bottom_right[1]]
    for idx, pixel in enumerate(bottom_border):
        if (bottom_left[0], idx + bottom_left[1]) in seeds:
            seeds[bottom_left[0], idx + bottom_left[1]] = 1

def expand_top_border(img, seeds, top_left, top_right, top_border):
    """This function expands a particular border by one pixel.

    """
    top_left[0]-=1
    top_right[0]-=1
    # Set border pixels as visited
    top_border = img[top_left[0], top_left[1]:top_right[1]]
    for idx, pixel in enumerate(top_border):
        if (top_left[0], idx + top_left[1]) in seeds:
            seeds[top_left[0], idx + top_left[1]] = 1

def expand_rectangle(img, seeds, top_right, top_left, bottom_right, bottom_left, vertical_la, horizontal_la):
    """This function expands a rectangle starting from the given
    borders. It takes into account the vertical and horizontal
    look-ahead values.

    """
    # Compute borders
    top_border    = img[top_left[0], top_left[1]:top_right[1]]
    bottom_border = img[bottom_left[0], bottom_left[1]:bottom_right[1]]
    left_border   = img[top_left[0]:bottom_left[0], top_left[1]]
    right_border  = img[top_right[0]:bottom_right[0], top_right[1]]

    # Expand the feature rectangle
    done                  = False
    num_look_ahead_top    = 0
    num_look_ahead_bottom = 0
    num_look_ahead_left   = 0
    num_look_ahead_right  = 0

    # Save the original region just in case we need to restore it later
    original_top_right    = top_right
    original_top_left     = top_left
    original_bottom_right = bottom_right
    original_bottom_left  = bottom_left

    while not done:
        if np.all(top_border == 255) and ((vertical_la == 0) or (vertical_la > 0 and num_look_ahead_top > vertical_la)):
            if vertical_la > 0:
                # Restore original values
                top_left  = original_top_left
                top_right = original_top_right
            if np.all(bottom_border == 255) and ((vertical_la == 0) or (vertical_la > 0 and num_look_ahead_bottom > vertical_la)):
                if vertical_la > 0:
                    # Restore original values
                    bottom_left  = original_bottom_left
                    bottom_right = original_bottom_right
                if np.all(left_border == 255) and ((horizontal_la == 0) or (horizontal_la > 0 and num_look_ahead_left > horizontal_la)):
                    if horizontal_la > 0:
                        # Restore original values
                        top_left    = original_top_left
                        bottom_left = original_bottom_left
                    if np.all(right_border == 255) and ((horizontal_la == 0) or (horizontal_la > 0 and num_look_ahead_right > horizontal_la)):
                        if horizontal_la > 0:
                            # Restore original values
                            top_right    = original_top_right
                            bottom_right = original_bottom_right
                        done = True
                    else:
                        # Enlarge rectangle one pixel right
                        if top_right[1] == np.shape(img)[1] - 1:
                            done = True
                        else:
                            expand_right_border(img, seeds, top_right, bottom_right, bottom_border)
                            right_border = img[top_right[0]:bottom_right[0], top_right[1]]
                            if horizontal_la > 0 and num_look_ahead_right <= horizontal_la:
                                num_look_ahead_right += 1
                else:
                    # Enlarge rectangle one pixel left
                    if top_left[1] == 0:
                        done = True
                    else:
                        expand_left_border(img, seeds, top_left, bottom_left, left_border)
                        left_border = img[top_left[0]:bottom_left[0], top_left[1]]
                        if horizontal_la > 0 and num_look_ahead_left <= horizontal_la:
                            num_look_ahead_left += 1
            else:
                # Enlarge rectangle one pixel downwards
                if bottom_left[0] == np.shape(img)[0] - 1:
                    done = True
                else:
                    expand_bottom_border(img, seeds, bottom_left, bottom_right, bottom_border)
                    bottom_border = img[bottom_left[0], bottom_left[1]:bottom_right[1]]
                    if vertical_la > 0 and num_look_ahead_bottom <= vertical_la:
                        num_look_ahead_bottom += 1
        else:
            # Enlarge rectangle one pixel upwards
            if top_left[0] == 0:
                done = True
            else:
                expand_top_border(img, seeds, top_left, top_right, top_border)
                top_border = img[top_left[0], top_left[1]:top_right[1]]
                if vertical_la > 0 and num_look_ahead_top <= vertical_la:
                    num_look_ahead_top += 1

def do_expand_feature_rectangle(orig, img, img_enhanced, seeds, upper_boundary, lower_boundary, numRectangles, color_img_blanks, color_img_limit, color_img_regions, file_name, look_ahead):
    """This function expands feature rectangles from a part of the
    image
    """
    output = []
    iterator = seeds
    if upper_boundary == -1 or lower_boundary == -1:
        # Calculate centroid and iterate pixels over the centroid
        keys          = seeds.keys()
        pixels        = np.array(keys).reshape(len(keys), -1)
        centroid      = (np.mean(pixels[:,0]), np.mean(pixels[:,1]))
        mykdtree      = cKDTree(np.array(pixels).reshape(len(pixels), -1))
        dist, indexes = mykdtree.query(centroid)

        horizontal_radius = np.shape(img)[1] - keys[indexes][1]
        vertical_radius   = np.shape(img)[0] - keys[indexes][0]
        radius            = min(horizontal_radius, vertical_radius)
        iterator          = circle_around(seeds, keys[indexes], radius)
        
    for pixel in iterator:
        visited = seeds[pixel]
        if not visited and valid_pixel(pixel, np.shape(img)[0], np.shape(img)[1], upper_boundary, lower_boundary):
            # Mark as visited
            seeds[pixel] = 1

            # Create an initial 3x3 window
            top_left     = [pixel[0] - 1, pixel[1] - 1]
            top_right    = [pixel[0] - 1, pixel[1] + 1]
            bottom_left  = [pixel[0] + 1, pixel[1] - 1]
            bottom_right = [pixel[0] + 1, pixel[1] + 1]
            
            # Mark all pixels inside this very first window as visited
            if (top_left[0], top_left[1]) in seeds:
                seeds[top_left[0], top_left[1]] = 1;
                if (top_right[0], top_right[1]) in seeds:
                    seeds[top_right[0], top_right[1]] = 1;
                    if (bottom_left[0], bottom_left[1]) in seeds:
                        seeds[bottom_left[0], bottom_left[1]] = 1;
                        if (bottom_right[0], bottom_right[1]) in seeds:
                            seeds[bottom_right[0], bottom_right[1]] = 1;
                            if (pixel[0] - 1, pixel[1]) in seeds:
                                seeds[pixel[0] - 1, pixel[1]] = 1
                                if (pixel[0], pixel[1] - 1) in seeds:
                                    seeds[pixel[0], pixel[1] - 1] = 1
                                    if (pixel[0] + 1, pixel[1]) in seeds:
                                        seeds[pixel[0] + 1, pixel[1]] = 1
                                        if (pixel[0], pixel[1] + 1) in seeds:
                                            seeds[pixel[0], pixel[1] + 1] = 1


            # Expand rectangle without look-ahead to extract the initial region
            expand_rectangle(img, seeds, top_right, top_left, bottom_right, bottom_left, 0, 0)

            # Compute the amount of look-ahead
            roi                   = orig[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]];
            vertical_look_ahead   = compute_vertical_look_ahead(roi)
            horizontal_look_ahead = compute_horizontal_look_ahead(roi)
            
            # Try to expand again the region
            expand_rectangle(img, seeds, top_right, top_left, bottom_right, bottom_left, vertical_look_ahead, horizontal_look_ahead)

            # Mark rectangles in red
            cv2.rectangle(color_img_regions, (top_left[1], top_left[0]), (bottom_right[1], bottom_right[0]), (0,0,255), 2) # BGR format

            # Update ROI and build output list
            roi = orig[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]];
            if not np.all(roi == 0):
                output.append(roi)
                numRectangles+=1
    
    return output

def expand_feature_rectangle (orig, img, img_enhanced, seeds, numRectangles, color_img_blanks, color_img_limit, color_img_regions, file_name, look_ahead):
    """ This function expands a feature rectangle starting from every
    unvisited seed in 'seeds' until it can no longer grow. """
    # Get the upper and lower boundaries 
    upper_boundary, lower_boundary = get_boundaries(img_enhanced, color_img_blanks)
    cv2.line(color_img_limit, (0, upper_boundary), (np.shape(img)[1], upper_boundary), (255,0,0), 2)
    cv2.line(color_img_limit, (0, lower_boundary), (np.shape(img)[1], lower_boundary), (255,0,0), 2)
    # If any of the boundaries delimit a region of area 0, discard boundaring
    if upper_boundary == 0 or lower_boundary == np.shape(img)[1]:
        # Pass -1 as boundaries as a way to say "don't use boundaries at all"
        output = do_expand_feature_rectangle(orig, img, img_enhanced, seeds, -1, -1, numRectangles, color_img_blanks, color_img_limit, color_img_regions, file_name, look_ahead)
        return output
    else:
        # Classify seeds in two groups
        upper_seeds = dict((seed, seeds[seed]) for seed in seeds if seed[0] < upper_boundary)
        lower_seeds = dict((seed, seeds[seed]) for seed in seeds if seed[0] > lower_boundary)
        # Do the actual feature rectangle expansion
        upper_output = do_expand_feature_rectangle(orig, img, img_enhanced, upper_seeds, upper_boundary, lower_boundary, numRectangles, color_img_blanks, color_img_limit, color_img_regions, file_name, look_ahead)
        lower_output = do_expand_feature_rectangle(orig, img, img_enhanced, lower_seeds, upper_boundary, lower_seeds, numRectangles, color_img_blanks, color_img_limit, color_img_regions, file_name, look_ahead)
        return upper_output + lower_output

def enhance_image (img):
    """This function enhances the document by applying a dilation after an erosion."""
    img_filt    = cv2.medianBlur(img, 15)
    eroded_img  = cv2.erode(img_filt, None, 5)
    dilated_img = cv2.dilate(eroded_img, None, 2)
    return dilated_img

def process_directory(args, logo_path, dir_list, denoise, look_ahead):
    """This function processes a given directory and, for each .tif file
    inside, applies the logo recognition algorithm.
    """
    for idx, file_name in enumerate(dir_list):
        if file_name.lower().endswith('.tif'):
            print file_name
            numRectangles = 0
            img = cv2.imread(logo_path + file_name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            # Extract a color image to paint in red the detected regions
            color_img_regions = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            color_img_limit   = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            color_img_blanks  = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # Denoise
            if denoise:
                img_enhanced = enhance_image(img)
                # Generate the seeds from where the logo detection starts
                seeds = calculate_seeds(img_enhanced)
                regions = expand_feature_rectangle(img, img_enhanced, img_enhanced, seeds, numRectangles, color_img_blanks, color_img_limit, color_img_regions, file_name, look_ahead)
            else:
                # Generate the seeds from where the logo detection starts
                seeds = calculate_seeds(img)
                regions = expand_feature_rectangle(img, img, img, seeds, numRectangles, color_img_blanks, color_img_limit, color_img_regions, file_name, look_ahead)

            # Expand feature rectangles around the seeds

            # Write the annotated image to the output folder
            cv2.imwrite(args.output_path + file_name, color_img_regions)
            cv2.imwrite(args.output_path_limit + file_name, color_img_limit)
            cv2.imwrite(args.output_path_blanks + file_name, color_img_blanks)
            # Write each region in its folder
            stripped_name = os.path.splitext(file_name)[0]
            os.mkdir(args.output_path_regions + stripped_name)
            for idx, region in enumerate(regions):
                cv2.imwrite(args.output_path_regions + stripped_name + "/" + str(idx) + "_" + file_name, region)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Implements a logo detection algorithm.')
    parser.add_argument("denoise_path", help="path with the input images we need to denoise")
    parser.add_argument("standard_path", help="path with the input images we DO NOT need to denoise")
    parser.add_argument("output_path", help="path where the output images are stored")
    parser.add_argument("output_path_limit", help="path where the output images with boundary marks are stored")
    parser.add_argument("output_path_blanks", help="path where the output images with blank spaces are stored")
    parser.add_argument("output_path_regions", help="path where the region crops are stored")
    parser.add_argument("--enhance", "-e", help="enhance morphologically the input image", action="store_true")
    args = parser.parse_args()

    dir_list_std = os.listdir(args.standard_path)
    dir_list_denoise = os.listdir(args.denoise_path)

    p1 = Process(target=process_directory, args=(args, args.standard_path, dir_list_std, False, 10))
    p1.start()

    p2 = Process(target=process_directory, args=(args, args.denoise_path, dir_list_denoise, True, 100))
    p2.start()

    p1.join()
    p2.join()
