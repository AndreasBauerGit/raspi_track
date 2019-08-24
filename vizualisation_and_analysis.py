
"""
Created on Sat Jul  6 18:58:19 2019

@author: andy

reading the output textfiles, conversion to arrays, making a movie
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib import cm 
from tqdm import tqdm
from matplotlib.animation import FuncAnimation,ArtistAnimation
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import imageio
import os
import copy
import cv2
# reading tracks to a dictionary:

def read_tracks(file):

    '''
    reading a text file containing tracks. Returns a dictionary with
    track_id:[[x_pos,y_pos,frame_id],...] 
    :param: file: full path to the text file
    :returns tracks_dict, dictionary containing all tracks
            frame, the last frame in this experiment
    '''
    
    tracks_dict=defaultdict(list) # dictionary with track_id:[[x_pos,ypos,frame_id],...]
    with open(file,"r") as f:
        for l in f.readlines():
            l=l[:-2].split("\t") # splitting at each tab stop, also removing the
            # tailing \n (newline)
            frame=l.pop(0) # extracting the frame and removing it from the list
            frame=int(frame)
            # iterating through all columns (one column represents a track) in one line l
            for id,position in enumerate(l):
                if position[:2] != "-1": # don't write to dict if entry is -1
                    # writing x_pos,y_pos,frame as one position to the track
                    tracks_dict[id].append([float(p) for p in position.split(",")]+[int(frame)])

    return tracks_dict,frame # the last frame during in the input textfile

def return_track_array(tracks_dict,frame_number):

    '''
    Converting a dictionary containing the tracks to a numpy array.
    Each column of the array represent one track, each row a frame and each cell
    is filled with the x,y  positions of the detections. Np.nan  is filled in
    if no detections was assigned at a frame for a track.
    :param tracks_dict: dictionary with track_id:[[x_pos,y_pos,frame_id],...]
    :param frame_number: number of the last frame
    :return: tracks_arr, nanpadded 3 dimensional array:  axis 0: frames,
     axis 1: tracks, axis 2: x,y coordinates
    '''

    # initiating empty array
    tracks_arr=np.zeros((frame_number+1,len(tracks_dict.keys()),2))+np.nan
    # filling the array
    for i,(track_id,values) in enumerate(tracks_dict.items()):
        for detections in values:
            tracks_arr[int(detections[2]),i,:]=detections[0:2]
    tracks_arr[tracks_arr==-1]=np.nan # inserting nan where no detections are found
    
    return tracks_arr



def make_tracks_video(tracks_arr,frames,folder,gaps=0,show_trailing=5,dims=(1000,1400), fps=5,
                      images=None,name="out"):

    '''
    Producing a gif/video that shows the tracking. Tracks are plotted as lines
    with markers at each detection and their id written at the detection of the 
    current frame. The number of previous detections displayed can be set in with the
    show_trailing parameter. Recorded images can be displayed with the tracks. Gaps
    introduced by stitching can be plotted as dashed lines. Making long videos on the pi
    may be difficult. Could get problems for >2000 frames (test your self).
    
    :param tracks_arr: Nan-padded array containing the tracks and corresponding
    detections. Should be generated by return_track_array.
    :param frames: list of frame ids to be displayed in the video
    :param folder: path to outputfolder
    :param gaps: nan padded array of the gaps introduced by stitching. Should be
    generated by return_track_array from the gaps_dict returned by the stitching function.
    :param show_tailing: number of detections from past frame displayed in the current
    window
    :param: dims: dimensions of the recorded images. The values will be used
    as x and y limits for the plot. You need to provide them even if no images
    are provided.
    :param fps: frame rate of the video
    :param images list of paths to images to be displayed. Optional. Must be same length
    and order as frames
    :param name: str, output name of the video. .avi will be added
    '''
    plt.ioff()  # avoid opening matplotlib windows while producing single plots
    # checking if image list is provided and constructing a dummy list if not
    if not isinstance(images,list):
        images=[False]*len(frames)
    # making first figure to get correct dimensions
    fig=plt.figure(1)
    plt.xlim((0,dims[1]))
    plt.ylim((0,dims[0]))
    width, height = (fig.get_size_inches() * fig.get_dpi()).astype(int) # dimensions of th created figures
    # initializing video with cv2
    fourcc = cv2.VideoWriter_fourcc(*'H264') # suitable encoding for videos
    video = cv2.VideoWriter(os.path.join(folder, name+".avi"), fourcc,fps, (width,height))
    plt.close(1)  # closes previous figure

    # iterating through all frames
    print("plotting tracks")
    for im_path,i in tqdm(zip(images,range(len(frames))),total=len(frames)):
        # new figure
        fig=plt.figure(1)
        # setting correct x and y limits
        plt.xlim((0,dims[1]))
        plt.ylim((0,dims[0]))
        # plotting number of the current frame
        plt.text(1, 1, str(i), transform=plt.gca().transAxes)
        # setting the range of frames to be displayed
        f_range=[i-show_trailing,i+1]
        if f_range[0]<0: # dealing with beginning of the animation
            f_range[0]=0
        frames_show=frames[f_range[0]:f_range[1]] # selecting the correct frame

        # iterating through all track ids
        for t_id in range(tracks_arr.shape[1]):
            m=~np.isnan(tracks_arr[frames_show,t_id,0]) # all existing positions in the show range
            if not m[-1]: # stop displaying tracks that have stopped...
                continue
            # x and y coordinates of these positions
            pos_x=tracks_arr[frames_show,t_id,0][m]
            pos_y=tracks_arr[frames_show,t_id,1][m]
            # plot point at track start seperately as points
            if len(pos_x)==1:
                plt.plot(pos_y,pos_x,"o",markersize=4)
                plt.text(pos_y[-1]+0.1,pos_x[-1]+0.1,str(t_id)) # adding track id as text
            # otherwise plot lines
            if len(pos_x)>1:
                plt.plot(pos_y,pos_x,linewidth=2,color="green",alpha=0.6)
                plt.text(pos_y[-1]+0.1,pos_x[-1]+0.1,str(t_id)) # adding track id as text

        # displaying gaos filled by stitching with dashed line if desired
        if isinstance(gaps,np.ndarray): # checking if gaps are provided
            for t_id in range(gaps.shape[1]):
                m=~np.isnan(gaps[frames_show,t_id,0]) # all existing positions
                if not m[-1]: # stop displaying tracks that have stopped...
                    continue
                # x and y coordinates of these positions
                pos_x=gaps[frames_show,t_id,0][m]
                pos_y=gaps[frames_show,t_id,1][m]
                # plot these points as dashed lines
                if len(pos_x)>1:
                    plt.plot(pos_y,pos_x,"--",linewidth=5,color="green")
                
        #showing images if provided
        if isinstance(im_path,str):
            im=plt.imread(im_path)
            fig.get_axes()[0].imshow(im)
        # extracting an image from the figure
        canvas = FigureCanvas(fig)
        canvas.draw()  
        im = np.fromstring(canvas.tostring_rgb(), dtype='uint8') # convert ot integers
        im=im.reshape(height, width, 3)
        # writing the image to the video
        video.write(im.astype("uint8"))
        # closing the figure
        plt.close(1)
    # clsoing the video object
    cv2.destroyAllWindows()
    video.release()
    plt.ion() # switching interactive matplotlib plotting back on
    plt.close("all") # closing all remaining figures, some figures seem to be hidden somehow..
    return 



# some example code
if __name__ == "__main__":
    #%matplotlib qt ##
### something spyder specific tp have plot windows appear seperately
      ## makeing a video
    folder=r"/home/pi/Desktop/images_camera4"
    file= r"/home/pi/Desktop/images_camera4/tracks2.txt"  
    tracks_dict,frame_number=read_tracks(file)
    tracks_arr=return_track_array(tracks_dict,frame_number)

    root_im=r"/home/pi/Desktop/images_camera4/rec%s.jpeg"
    frames=list(range(30)) 
    l=[str(n).zfill(5) for n in frames]
    
    
    image_list=[root_im%i for i in l]
    dims = plt.imread(image_list[0]).shape
    
    figures=make_tracks_video(tracks_arr,frames,folder,show_trailing=3,
                              dims=dims,images=image_list,vid_format="gif")
    figures[0]## cna only show like this in notebook???
        
    figures=make_tracks_video(tracks_arr,frames,folder,show_trailing=3,
                              dims=dims,images=image_list,vid_format="avi")
    
# to do: update make_tracks_video by using frames_dict