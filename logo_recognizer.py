#! /usr/bin/env python

""" This is a logo recognizer application for detecting logos in official documentation.

Each candidate to logo is stored in its own folder: <Regions_output_path>/<Image_name>/...
Additional intermediate files are also stored for debugging purposes.

It uses the feature rectangles concept from http://www.cvc.uab.es/icdar2009/papers/3725b335.pdf
with some heuristics (position of the logo in the document and vertical profile of the image)
    
TODO: Refactor this code to be able to use it as an independent module.
"""

import cv2
import numpy as np
import os
import argparse
from xml.dom.minidom import parse

class Logo:
    """ This class represent a detected logo """
    def __init__(self, row, col, width, height):
        self.row = row
        self.col = col
        self.width = width
        self.height = height

def get_ground_truth_logos(file_name):
    """This method returns a list of logos from ground-truth
information for a particular document"""
    dom = parse(file_name)
    logos = []
    for node in dom.getElementsByTagName("DL_ZONE"):
        if node.getAttribute("gedi_type") == "DLLogo":
            row = node.getAttribute("row")
            col = node.getAttribute("col")
            width = node.getAttribute("width")
            height = node.getAttribute("height")
            logo = Logo(row, col, width, height)
            logos.append(logo)
    return logos

def longest_increasing_run(blank_lines):
    """This method computes the longest increasing run in a list. That
run is the largest gap and potential separator between logo and
non-logo elements"""
    start = 0
    end = 0
    maxStart = 0
    maxEnd = 0
    maxLength = 1

    for idx in range(1, len(blank_lines)):
        if blank_lines[idx] - blank_lines[idx-1] == 1:
            end = end + 1
        else:
            # Check if it is the longest so far
            if end - start + 1 > maxLength:
                maxLength = end - start + 1
                maxStart = start
                maxEnd = end
            start = end = idx
        if end - start + 1 > maxLength:
            maxLength = end - start + 1
            maxStart = start
            maxEnd = end
    return blank_lines[maxStart], blank_lines[maxEnd]
        
def get_boundaries(img):
    """This function calculates the upper and lower boundary of text in
the document."""
    img_height = np.shape(img)[0]
    blank_lines = []
    first_non_blank = False
    upper_boundary = 0
    lower_boundary = 0
    for idx in range(img_height):
        line = img[idx,:]
        if len(np.where(line == 0)[0]) >= 50:
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

def calculate_seeds (img):
    """ This function scans the image from top to bottom, left to
    right and returns a dictionary of foreground pixels where the keys
    are the pixel coordinates and the values are 1 or 0 depending whether
    the pixel has been visited or not."""
    foreground = np.where(img == 0)
    foreground_length = len(foreground[0])
    pixels = zip(foreground[0], foreground[1])
    pixels_visited = zip(pixels, np.zeros(foreground_length))
    return dict(pixels_visited)

def valid_pixel (pixel, height, width, upper_boundary = 200, lower_boundary = 200):
    """ This function validates a pixel to be within the image
    boundaries."""
    if pixel[1] <= 0 or pixel[1] >= width - 1:
        return False
    if pixel[0] <= 0 or pixel[0] >= height - 1:
        return False
    return pixel[0] < upper_boundary or pixel[0] > lower_boundary

def expand_feature_rectangle (img, seeds, numRectangles):
    """ This function expands a feature rectangle starting from every
    unvisited seed in 'seeds' until it can no longer grow. """
    # Initialize output list
    output = []
    # Get the upper and lower boundaries 
    upper_boundary, lower_boundary = get_boundaries(img)
    cv2.line(color_img_limit, (0, upper_boundary), (np.shape(img)[1], upper_boundary), (255,0,0), 2)
    cv2.line(color_img_limit, (0, lower_boundary), (np.shape(img)[1], lower_boundary), (255,0,0), 2)
    for pixel in seeds:
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

            # Expand the feature rectangle
            done = False
            while not done:
                top_border = img[top_left[0], top_left[1]:top_right[1]]
                if np.all(top_border == 255):
                    bottom_border = img[bottom_left[0], bottom_left[1]:bottom_right[1]]
                    if np.all(bottom_border == 255):
                        left_border = img[top_left[0]:bottom_left[0], top_left[1]]
                        if np.all(left_border == 255):
                            right_border = img[top_right[0]:bottom_right[0], top_right[1]]
                            if np.all(right_border == 255):
                                done = True
                            else:
                                # Enlarge rectangle one pixel right
                                if top_right[1] == np.shape(img)[1] - 1:
                                    done = True
                                else:
                                    top_right[1]+=1
                                    bottom_right[1]+=1
                                    # Set border pixels as visited
                                    right_border = img[top_right[0]:bottom_right[0], top_right[1]]
                                    for idx, pixel in enumerate(right_border):
                                        if (idx + top_right[0], top_right[1]) in seeds:
                                            seeds[idx + top_right[0], top_right[1]] = 1

                        else:
                            # Enlarge rectangle one pixel left
                            if top_left[1] == 0:
                                done = True
                            else:
                                top_left[1]-=1
                                bottom_left[1]-=1
                                # Set border pixels as visited
                                left_border = img[top_left[0]:bottom_left[0], top_left[1]]
                                for idx, pixel in enumerate(left_border):
                                    if (idx + top_left[0], top_left[1]) in seeds:
                                        seeds[idx + top_left[0], top_left[1]] = 1

                    else:
                        # Enlarge rectangle one pixel downwards
                        if bottom_left[0] == np.shape(img)[0] - 1:
                            done = True
                        else:
                            bottom_left[0]+=1
                            bottom_right[0]+=1
                            # Set border pixels as visited
                            bottom_border = img[bottom_left[0], bottom_left[1]:bottom_right[1]]
                            for idx, pixel in enumerate(bottom_border):
                                if (bottom_left[0], idx + bottom_left[1]) in seeds:
                                    seeds[bottom_left[0], idx + bottom_left[1]] = 1
                else:
                    # Enlarge rectangle one pixel upwards
                    if top_left[0] == 0:
                        done = True
                    else:
                        top_left[0]-=1
                        top_right[0]-=1
                        # Set border pixels as visited
                        top_border = img[top_left[0], top_left[1]:top_right[1]]
                        for idx, pixel in enumerate(top_border):
                            if (top_left[0], idx + top_left[1]) in seeds:
                                seeds[top_left[0], idx + top_left[1]] = 1
    
            cv2.rectangle(color_img_regions, (top_left[1], top_left[0]), (bottom_right[1], bottom_right[0]), (0,0,255), 2) # BGR format
            # Save ROI to output list
            roi = img[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
            output.append(roi)
            numRectangles+=1
    
    # Print relevant information
    print file_name, numRectangles
    return output

def dilate (img):
    """ This function dilates the document using a 1x1 kernel."""
    dilated_img = cv2.dilate(img, None)
    return dilated_img

numRectangles = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Implements a logo detection algorithm.')
    parser.add_argument("input_path", help="path with the input images")
    parser.add_argument("output_path", help="path where the output images are stored")
    parser.add_argument("output_path_limit", help="path where the output images with boundary marks are stored")
    parser.add_argument("output_path_blanks", help="path where the output images with blank spaces are stored")
    parser.add_argument("output_path_regions", help="path where the region crops are stored")
    parser.add_argument("--dilate", "-d", help="dilate the input image first", action="store_true")
    args = parser.parse_args()
    
    dir_list = os.listdir(args.input_path)
    for idx, file_name in enumerate(dir_list):
        if idx != 0: # Ignore DS.Store folder
            img = cv2.imread(args.input_path + file_name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            # Extract a color image to paint in red the detected regions
            color_img_regions = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            color_img_limit = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            color_img_blanks = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if args.dilate:
                img = dilate(img)
            # Generate the seeds from where the logo detection starts
            seeds = calculate_seeds(img)
            # Expand feature rectangles around the seeds
            regions = expand_feature_rectangle(img, seeds, numRectangles)
            # Write the annotated image to the output folder
            cv2.imwrite(args.output_path + file_name, color_img_regions)
            cv2.imwrite(args.output_path_limit + file_name, color_img_limit)
            cv2.imwrite(args.output_path_blanks + file_name, color_img_blanks)
            # Write each region in its folder
            stripped_name = os.path.splitext(file_name)[0]
            os.mkdir(args.output_path_regions + stripped_name)
            for idx, region in enumerate(regions):
                cv2.imwrite(args.output_path_regions + stripped_name + "/" + str(idx) + "_" + file_name, region)
