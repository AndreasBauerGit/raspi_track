#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 17:46:43 2019

@author: andy
"""

import numpy as np
from collections import defaultdict
import copy
import pickle
import os
import datetime
import time
import imageio
from segementation_bugs import *
from stitching import *
from tracking import *
import re


camera_resolution= (1400, 1000)

# settings for detection and tracking
s1 = 12  # large sigma for difference of gaussian filters
s2 = 5  # large sigma for difference of gaussian filters
sd_threshold = 5  # factor for segmentation threshold
min_treshohld = None  # optional minimal threshold
min_size = 30  # minimal size for of objects
max_dist = 150  # maximal distance between two detection in consecutive frames to allow tracking

# settings for stitching and filtering
s_max = 170  # maximal allowed spatial distance for stitching
f_max = 10  # maximal frame difference allowed for stitching
f_min = -2  # minimal allowed frame difference for stitching
end_dist = 30  # minimal distance at start and end of two tracks to allow removal as "parallel" tracks
mean_dist = 30  # minimal average distance between two tracks to allow removal as "parallel" tracks

# change video settings in the corresponding function

# output files and others
folder="/home/andy/Desktop/bug_tracking_images/"
file = "tracks.txt"  # output file for the unfiltered/unstitched tracks
file2 = "tracks2.txt"  # output file for stitched/filtered tracks
file_times = 'times.pickle'  # output file for times: uses pickle format
out_img = "rec%s.jpeg"  # output format for the images
# %s signifies where numbers are inserted; don't remove this
#mask = np.load("/home/pi/Desktop/bugs_scripts/mask.npy")  # mask of dish area; make sure the path is correct
mask = np.load("/home/andy/Git_hub_projects/raspi_track/mask.npy")
n_zfill = 5  # number of zeros padded to the output name of images. Must not be to small.




# interpolating mask to suitable size
mask=interpolation(mask,dims=(camera_resolution[1],camera_resolution[0]))        
mask=mask.astype(bool)

#filling the path for the files
file=os.path.join(folder,file)
file2=os.path.join(folder,file2)
file_times=os.path.join(folder,file_times)
out_img=os.path.join(folder,out_img)


######
if os.path.isfile(file):  # deleting old text file if it exists
        os.remove(file)
# empty variables for tracking
prev_detections = np.array([])  # list of detections from the previous frame
prev_tracks = {}  # dictionary assigning "index in prev_detection list":"track id"
tracks = {}  # dictionary assigning "index in detection list":"track id"
max_track_id = -1  # highest id of tracking
times = {}  # dictionary to note all time points, when images where recorded
n = 0  # frame counter

# finding images
files=[x for x in os.listdir(folder) if re.search("\D*(\d{1,6}).jpeg",x)]

# sorting images
files=sorted(files,key=lambda x: re.search("\D*(\d{1,6}).jpeg",x).group(1))
# listing frames
frames=[ re.search("\D*(\d{1,6}).jpeg",x).group(1) for x in files]


for im_file,frame in tqdm(zip(files,frames),total=len(frames)):
    image=plt.imread(os.path.join(folder,im_file))
    image = np.mean(image, axis=2)
    # tracking and writing to file for each step
    prev_tracks, prev_detections, max_track_id = tracking(image, mask, prev_tracks,
        max_track_id, prev_detections, frame, file, sd_threshold,
         max_dist, min_size, s1, s2, min_treshohld)




# reading  tracks from file
tracks_dict, frame_number = read_tracks(file)  # read tracks.txt
tracks_arr = return_track_array(tracks_dict, frame_number=frame_number)  # convert to nan padded array
tracks_stitched, stitched_id, gaps, old_ids = stitch(tracks_dict, f_min=f_min, f_max=f_max, s_max=s_max)  # stitching
# convert the stitched tracks to nan padded array
tracks_stitched_arr = return_track_array(tracks_stitched, frame_number=frame_number)
# convert the gaps introduced by stitching to nan padded array (needed for video)
gaps_arr = return_track_array(gaps, frame_number=frame_number)
# removing "parallel tracks"
tracks_f = remove_parralle_tracks(tracks_stitched, tracks_stitched_arr, end_dist=end_dist, mean_dist=mean_dist)
tracks_f_arr = return_track_array(tracks_f, frame_number=frame_number)
# saving stitched and filtered tracks
write_tracks2(tracks_f_arr, file2)

# makeing a video
#folder = "/home/pi/Desktop/images_camera/"
tracks_dict, frame_number = read_tracks(file2)  # reading tracks dict
tracks_arr = return_track_array(tracks_dict, frame_number)

# generating a list of frames
frames = list(range(len(files)))

# generating a list of paths to images
root_im = out_img
frame_list = [str(n).zfill(n_zfill) for n in frames]
image_list = [root_im % i for i in frame_list]
# getting the correct image dimensions
dims = plt.imread(image_list[0]).shape

# producing the video
figures = make_tracks_video(tracks_arr, frames, folder, show_trailing=3, fps=3,
                          dims=dims, images=image_list, name="out")
