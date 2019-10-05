'''
performing tracking (connecting tracks from one frame to the next frame) and writing
the results to an output file
'''

import os
#os.chdir("E:\\tracking_scripts\\")
from segementation_bugs import *
from stitching import *
from vizualisation_and_analysis import *
import numpy as np
from collections import defaultdict
import copy
from tqdm import tqdm


def createFolder(directory):
    '''
    function to create directories, if they dont already exist
    '''
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def interpolation(mask, dims):
	# only works when interpolating to lower resolution
	# note: remove_small_objects labels automatically if mask is bool
	coords = np.array(np.where(mask)).astype(float) # coordinates of all points
	interpol_factors = np.array([dims[0] / mask.shape[0], dims[1] / mask.shape[1]])
	coords[0] = coords[0] * interpol_factors[0]  # interpolating x coordinates
	coords[1] = coords[1] * interpol_factors[1]  # interpolating xy coordinates
	coords = np.round(coords).astype(int)
	
	coords[0,coords[0]>=dims[0]]=dims[0]-1 # fixing issue when interpolated object is just at the image border
	coords[1, coords[1] >= dims[1]] = dims[1]-1
	
	mask_int = np.zeros(dims)
	mask_int[coords[0], coords[1]] = 1
	mask_int = mask_int.astype(int)
	#filling gaps if we interpolate upwards
	return mask_int


def write_tracks(tracks,detections,file,frame,max_track_id):

    '''
    writes tracks to text file. First col gives frame Id. For each other column
    the number of the column gives the track id. Detections of the tracks are
    written as x,y positions in each cell. If the entry is -1,-1 then no detection
    was assigned in this frame to this track. Don't use this method of saving the tracks if you expect
    a very large number of tracks (>1000).

    :param: tracks, dictionary with index_of_detection: track_id
    :param: detection, list of x,y positions of the detections in one frame
    :param: file, filename of a text file, either existing or to be generated
    :param:frame, number of the current frame
    :param: max_track_id, number of the highest current track id
    '''
    # write empty line if no tracks have been  found yet
    if max_track_id==-1:
        with open(file, "a+") as f:
            f.write(str(frame)+"\n")
        return

    track_ids=np.array(list(tracks.values()),dtype=int) # list of track ids
    detect_ids=np.array(list(tracks.keys()),dtype=int) # list of corresponding ids in the detection list

    # initializing lines of the text file
    line_x=np.zeros(int(max_track_id+1))-1
    line_y=np.zeros(int(max_track_id+1))-1
    # filling with x and y coordinates of the detections at the correct positions
    if len (detections)>0:
        line_x[track_ids]= detections[detect_ids,0]
        line_y[track_ids] = detections[detect_ids, 1]
    # converting to string
    line_x=line_x.astype(np.unicode_).tolist()
    line_y=line_y.astype(np.unicode_).tolist()
    # joining to one line
    line=[x+","+y for x,y in zip(line_x,line_y)] # writing as x,y values in one line
    # adding the frame id
    line=[str(frame)]+line
    # joing to one string
    line="\t".join(line)+"\n" # tab seperation between x,y pairs and comma separation between x,y
    # writing line to file
    with open(file,"a+") as f:
        f.write(line)
      

def show_single_tracking_step(image,prev_detections,detections,prev_tracks,tracks):
    plt.figure()
    plt.imshow(image)
    for i,pd in enumerate(prev_detections):
        if prev_tracks[i] in tracks.values(): # connected detection
            p1=prev_detections[i]
            p2=detections[[y for y,x in tracks.items() if x==prev_tracks[i]][0]]
            plt.plot([p1[1],p2[1]],[p1[0],p2[0]],color="C3",linewidth=5)
        p1 = prev_detections[i] # plotting all points
        plt.plot(p1[1], p1[0],"o", color="C0")
    for i,d in enumerate(detections):
        p1 = detections[i] # plotting all tracks
        plt.plot(p1[1], p1[0], "o", color="C1")


def tracking(image,mask,prev_tracks,max_track_id,prev_detections,frame,
             file,sd_threshold,max_dist,min_size,s1,s2,min_treshold):
        '''
        perfoms detection on an image, finds the correct connections to a set of detections from the previous frame,
        assignes the correct track id from a list of ids from the previous frame, and writes to a text file.

        :param image: image: 2 dimensional array representing an image
        :param mask: optional area  which to use for thresholding and segmentation
        :param prev_tracks: dictionary assigning "index in prev. detection list":"track id"
        :param max_track_id: the highest track id so far
        :param prev_detections: list of detections from the previous frame
        :param frame: number of the current frame
        :param file: path to the output text file
        :param sd_threshold: factor of how many standard deviations from the mean the threshold will be set
        :param max_dist: maximum distance allowed to connect two detections
        :param min_size: minimal size of allowed objects. Any area below this size will not return a detection
        :param s1: lower standard deviation for difference of gaussian filter
        :param s2: higher standard deviation for difference of gaussian filter
        :param min_treshold: optional minimal value for the threshold
        :return:prev_tracks,new track dictionary with "index in prev. detection list":"track id"
                prev_detections, new detection list
                max_track_id, new highest track id
        '''

        # detection on an image
        detections=segmentation_and_detection(image,mask,sd_threshold,
                min_size=min_size,s1=s1,s2=s2,min_treshold=min_treshold)
        detections=np.round(np.array(detections), 3) # rounding to a reasonable precision
        
        # produce empty lists and dictionaries if no detection is found
        if detections.size==0:
            #print("A")
            tracks = {}
            track_ids = []
            
        # starting new tracks if previous tracks is empty
        if prev_detections.size==0 and not detections.size==0:
            #print("B")
            #  initiate tracks dictionary, starting from max_track_id
            tracks = {i: i+max_track_id+1 for i in range(len(detections))}
            max_track_id = np.max(list(tracks.values()))  # new highest track id
        
        # nearest neighbour tracking if detections are present in both the current and the last frame
        if not prev_detections.size==0 and not detections.size==0:
            #print("C")
            # calculate distance between all points
            # this is given as matrix with: cols: previous detections, rows: new detections
            distances = np.linalg.norm(prev_detections[None, :] - detections[:, None], axis=2)

            # assignnemnt of detections, col_ind is index from
            # previous detections, row ind is from new detections
            row_ind, col_ind = [], []
            min_dist=0
            while np.nanmin(min_dist< max_dist) and not np.isnan(distances).all():# stops when maximum distance is reached
                min_pos = list(np.unravel_index(np.nanargmin(distances), distances.shape)) # position of the smallest
                min_dist=distances[min_pos[0],min_pos[1]] # value of the smalles distance
                # filling the other entries involving these two detections with nan
                distances[min_pos[0], :] = np.nan
                distances[:, min_pos[1]] = np.nan
                # noting the indices of this detection pair
                row_ind.append(int(min_pos[0]))
                col_ind.append(int(min_pos[1]))
            row_ind, col_ind = np.array(row_ind), np.array(col_ind)

            # assigning the track ids for this frame
            tracks={} # new tracks dictionary
            # iterating through all indices of the new detections
            for all_ind in range(len(detections)):
                if all_ind in row_ind: # if the detection was assigned
                        old_ind=col_ind[row_ind==all_ind] # index of previous detection it was assigned to
                        # writing the index of the track of the previous detection to the new detection
                        tracks[all_ind]=prev_tracks[old_ind[0]]
                else: # if the detection wasn't assigned start a new track
                    max_track_id+=1 # update highest track id
                    tracks[all_ind]= max_track_id # new entry in tracks dictionary


        #show_single_tracking_step(image, prev_detections, detections, prev_tracks, tracks)
        prev_tracks=copy.deepcopy(tracks) # saving track assignment
        prev_detections=copy.deepcopy(detections) # saving for the next round of tracking
        track_ids=np.array(list(tracks.keys())) # list of all ids in this assignment



        #writing tracks to file
        write_tracks(tracks,detections,file,frame=frame,max_track_id=max_track_id)
        
        return prev_tracks,prev_detections,max_track_id
        
        

 # some example code when tracking from existing images
if __name__=="__main__":
    #paramteres
    sd_threshold=5 # paramtere for threholding
    #min_treshohld=0.15# optional minimla threshold
    min_treshohld=None
    #m_d_watershed=7 # minimal seperation between two objects during watershedding
    #p_d=0.75 # amximal "prominence" allowed during watershedding
    max_dist=200 # maximal distancse between two detection in consecutive frames to allow
    min_size=30
    s1=12 # sigmas for dog segmentation
    s2=5
    #l=["0009","0010","0011","0012","0013","0014","0015"]
    
    file="/home/andy/Desktop/rasperry_project_bugs/tracks2.txt"# utput file
    mask = np.load("/home/andy/Desktop/rasperry_project_bugs/mask.npy") # mask of dish area
   
    if os.path.isfile(file):  # deleting old text file if it exists
        os.remove(file)
        
    prev_detections=np.array([])
    prev_tracks={} # dictionary that connects the index in the detection 
    #list to the id of a track key is index, value is track id
    tracks={}
    max_track_id=-1 # starts at -1 to handle completely empty
    # lines at the beggining of the textfile to which tracks are weritten
    
    l=[str(n).zfill(4) for n in range(30)] # images to be iterated through
    for frame in tqdm(l,total=len(l)):
        # reading images and convert to grey scale
            image=plt.imread("/home/andy/Desktop/rasperry_project_bugs/rec%s.jpeg"%frame)
            if len(image.shape) == 3:
                image = np.mean(image, axis=2)
            # tracking and writing to file for each step
            prev_tracks, prev_detections,max_track_id=tracking(image,mask,
                    prev_tracks,max_track_id,prev_detections,frame,
                    file,sd_threshold,max_dist,min_size,s1,s2,min_treshohld) 
             
    ### stitching and video
    
    # reading all tracks from file    
    tracks_dict,frame_number=read_tracks(file) # read tracks.txt
    tracks_arr=return_track_array( tracks_dict,frame_number=frame_number)
    # stitching and 
    tracks_stitched,stitched_id,gaps,old_ids=stitch(tracks_dict,seconds_per_frame=0.2,f_max=10,speed=50,s_max=300)
    tracks_stitched_arr=return_track_array(tracks_stitched,frame_number=frame_number)
    gaps_arr=return_track_array(gaps,frame_number=frame_number)
    
    # removing "parallel tracks
    tracks_f=remove_parralle_tracks(tracks_stitched,tracks_stitched_arr,end_dist=30,mean_dist=30)
    tracks_f_arr=return_track_array(tracks_f,frame_number=frame_number)
    
  
    file2="/home/andy/Desktop/rasperry_project_bugs/tracks2_stiched.txt"
    write_tracks2(tracks_stitched_arr,file2)
    
    
    ## makeing a video
    folder= "/home/andy/Desktop/rasperry_project_bugs/"
    file2= "/home/andy/Desktop/rasperry_project_bugs/tracks2_stiched.txt"
    tracks_dict,frame_number=read_tracks(file2)
    tracks_arr=return_track_array(tracks_dict,frame_number)

    root_im= "/home/andy/Desktop/rasperry_project_bugs/rec%s.jpeg"
    frames=list(range(30)) 
    l=[str(n).zfill(4) for n in frames]
    
    
    image_list=[root_im%i for i in l]
    dims = plt.imread(image_list[0]).shape    
   # figures=make_tracks_video(tracks_arr,frames,folder,gaps_arr,show_trailing=3,
    #                          dims=dims.shape,images=image_list,vid_format="avi", name = str(name)+'_vid')
    figures=make_tracks_video(tracks_arr,frames,folder,show_trailing=3, fps=5,
                              dims=dims,images=image_list,vid_format="avi", name ="1_vid")


# to do: get rid of second np.nanmin() in nearest neighbour tracking

    
