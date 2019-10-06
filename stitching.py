#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 21:07:35 2019

@author: andy

stitching (connecting track start and ends over multiple frames) tracks and filtering parallel tracks

"""
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import copy 
from vizualisation_and_analysis import *


def write_tracks2(tracks_arr,file):

    '''
    writes tacks to same text file format as "write_tracks", but does so all at once.
    :param tracks_arr: nan padded array of tracks
    :param file:  path to the output file, existing file will be over written
    :return:
    '''
    
    tracks_arr_w=copy.deepcopy(tracks_arr)
    tracks_arr_w[np.isnan(tracks_arr)]=-1 # replacing nans according to convention
    tracks_arr_w=tracks_arr_w.astype(np.unicode_) # converting to string
    with open(file,"w+") as f:
        for frame,l in enumerate(tracks_arr_w): # iterating through all rows
            line=[",".join(pos) for pos in l] # inserting comma between x and y position
            line=str(frame)+"\t"+"\t".join(line)+"\n" # adding frame  and separating with tab
            f.write(line) # writing a line
    
def write_times(times,file):

    '''
    writes the time points for each frame to a dictionary
    :param times: dictionary with frama_id:time
    :param file:  path to the outputfile, existing file will be overwritten
    :return:
    '''

    with open(file,"w+") as f:
        for frame,t in times.items():
            f.write(str(frame)+"\t"+str(t)+"\n")

#def fill_gaps(tracks,gaps):
#
#    '''
#    fills the tracks dictionary with position in gaps bridge while stitching
#    :param tracks:
#    :param gaps:
#    :return:
#    '''
#
#    n_tracks=copy.deepcopy(tracks)
#    for t_id in tracks.keys():
#        if len(gaps[t_id])>2:
#            n_tracks[t_id].extend(gaps[t_id]) # appending gaps
#           n_tracks[t_id].sort(key=lambda x: x[-1]) # sorting elements by frame

#   return n_tracks


def stitch_order(stitched_id):
    
    '''
    assembles the correct order in which to stitch tracks and assigne the id of all tracks that are stiched
    to the track at the beginnig of these tracks. Later all stitched tracks will be added to this id.
    :param stitched_id: un ordered list of id pairs of tracks that will be stitched. The end of tracks in the
    first column will be stitched to the start of tracks in the second columnm
    :return: stitching_lists, dictionaray with "id of first track": "all other ids"
    '''
    # return empty list if input is empty
    if len(stitched_id)<1:
        return []
    # all tracks of which the end is stitched, but not the start
    start_points=set(stitched_id[:,0])-set(stitched_id[:,1]) 
    # all tracks where only the start is stitched but not the end
    end_points=set(stitched_id[:,1])-set(stitched_id[:,0]) 

    # going through the stitched list form start point, until endpoint is reached
    stitching_lists = defaultdict(list)
    for sp in start_points:
        new_id=int(stitched_id[:,1][np.where(stitched_id[:,0]==sp)]) # finding new point
        stitching_lists[sp].append(new_id)
        while new_id not in end_points: # iterating until endpoint is reached
            new_id=int(stitched_id[:,1][np.where(stitched_id[:,0]==new_id)])
            stitching_lists[sp].append(new_id) # adding al visited track ids to the correct start id
    return stitching_lists
            
def predict_points(pos1,pos2):
    '''
    interpolation of x,y position between two points lying multiple frames appart.
    This function is used to fill the gaps in a track produced by stitching. Points
    are interpolated by equaly distributing them on a straight line connecting the two
    points at the start and the end of a gap.

    :param pos1: list or array with [x,y,frame] of the last point before the gap
    :param pos2: list or array with [x,y,frame] of the first point after the gap
    :return: pos_new: list of interpolated positions for all frames between pos1 and pos2
    '''

    steps=pos2[-1]-pos1[-1] #number points to interpolate betwen pos1 and pos2
    dif_vec=np.array(pos2[:-1])-np.array(pos1[:-1]) #vector connecting pos1 and pos2 in space
    pos_new=[]
    for i in range(1,steps):
        pos_new.append(list(np.array(pos1[:-1])+dif_vec*i/(steps))+[pos1[-1]+i]) # interpolation

    return pos_new

def stitch(tracks_dict,f_min=-2,f_max=10,s_max=300):
    '''
    Stitching
    Stitching is intendend to connect interrupted tracks (no detection in some
    frames in between). Here tracks ends and beginnings are only allowed to  be
    f_max frames apart, and overlap by f_min frames. Then the euclidean distance
    between tracks is calculated and used as a score. Similar to tracking, the closest
    track are stitched with some maximal values s_max.
    (Note: This approach doesn't work for >1000 tracks.)

    :param tracks_dict: dictionary of the tracks as returned by read_tracks from the output text files.
    Format needs to be track_id:[[x,y,frame1],[x,y,frame2],...]
    :param f_min: minimal allowed temporal overlap. Should be negative or 0.
    :param f_max: maximal allowed temporal overlap
    :param s_max: maximla allowed stitch score(e.g. euclidean distance)
    :return: tracks_stitched3, dictionary of all tracks after stitching.Has the same format as tracks_dict.
                Ids are reassigned from 0 to number_of_tracks. Interpolated values at gaps are not included.
             stitched_id, list of all stitched id pairs
             gaps, dictionary of all gaps containing interpolated points. Same format as tracks_dict.
             old_ids, list of all ids that were presetn in the old track
    '''

    n_tracks=len(tracks_dict.keys()) # number of tracks

    # creating a dictionary with track id:[[all x,y coordinates],[first frame],[last frame]]
    # this makes subsequent steps easier
    stitch_dict = {}
    for track_id,positions in tracks_dict.items():
        positions=np.array(positions)
        stitch_dict[track_id] = (positions[:,np.array([0,1])], positions[0,2], positions[-1,2])

    # calculating the temporal and euclidean distances for all tracks
    # they are represented in a matrix where the end of tracks are on the
    # rows and the start of tracks are on the columns

    # setting up distance matrices in space and time
    distance_matrix_space=np.zeros((n_tracks,n_tracks)) + np.nan
    distance_matrix_time=np.zeros((n_tracks,n_tracks)) + np.nan
    # filling the distances matrix by iterating through all pairs
    for i,key1 in enumerate(stitch_dict.keys()):
        for j,key2 in enumerate(stitch_dict.keys()):
            if key1 != key2: # avoid calculation of start end distance within one track
                # end frame of track_id key1 - start frame of track_id key2
                time_dist=-(stitch_dict[key1][2]-stitch_dict[key2][1])
                # euclidean distance of end track_id key1 and start track_id key2
                space_dist=np.sqrt(np.sum((stitch_dict[key1][0][-1]-stitch_dict[key2][0][0])**2))
                distance_matrix_space[i,j] = space_dist # filling the matrix
                distance_matrix_time[i,j] = time_dist

    # excluding track pairs with temporal distance >f_max and <f_min
    stitch_score_matrix=copy.deepcopy(distance_matrix_space)
    stitch_score_matrix[(distance_matrix_time > f_max) + (distance_matrix_time<=f_min)]=np.nan

    # finding track pairs that need to be stitched. The score matrix is iteratively searched for the best
    # matches, until the maximal allowed score s_max is reached
    stitched_id=[]
    while True:
        # finding  indices of minimum
        minimum=np.nanmin(stitch_score_matrix)        
        if minimum > s_max or np.isnan(minimum):
            break
        minimum_pos = np.unravel_index(np.nanargmin(stitch_score_matrix,axis=None), stitch_score_matrix.shape)
        stitch_score_matrix[minimum_pos[0],:].fill(np.nan)  # deleting stitched end and starts entries in the score matrix
        stitch_score_matrix[:,minimum_pos[1]].fill(np.nan)
        id1=list(stitch_dict.keys())[minimum_pos[0]] # writing stichted ids into a list
        id2=list(stitch_dict.keys())[minimum_pos[1]]
        stitched_id.append((id1,id2))  # stitch stitched[0] (end) to stitched[1] (start)
    stitched_id=np.array(stitched_id)

    # assigning tracks that are stitched together to the track at their beginning
    stitching_lists=stitch_order(stitched_id)
    
    # writing new tracks_dicts.
    tracks_stitched=defaultdict(list)# updated dictionary with stiched tracks
    gaps=defaultdict(list) # dictionary with the gaps produced by stitching
    
    if len(stitched_id)==0: # return unchanged tracks_dict and other empty results if nothing is stitched
        old_ids=list(tracks_stitched.keys())
        return tracks_dict,stitched_id,gaps,old_ids

    # copying tracks that have not been stitched
    not_stitched=set(list(tracks_dict.keys()))-set(stitched_id[:,0]).union(set(stitched_id[:,1]))
    for id in not_stitched:
        tracks_stitched[id]=copy.deepcopy(tracks_dict[id])
        gaps[id]=[] # nothing filled up in none stitched tracks # could be left out??

    # merging the stitched tracks
    for id_start,ids in stitching_lists.items():
        # copying the start track and using its id as the new track id for
        # all other tracks in this stitched group
        tracks_stitched[id_start]=copy.deepcopy(tracks_dict[id_start])
        # adding points from all other tracks in this stitched group
        for id in ids:
            tracks_stitched[id_start]+=tracks_dict[id]
        # noting the gaps and interpolating points in the gaps
        # first gap
        gaps[id_start].extend(predict_points(tracks_dict[id_start][-1],tracks_dict[ids[0]][0]))
        # all further gaps
        for i in range(len(ids)-1): 
            gaps[id_start].extend(predict_points(tracks_dict[ids[i]][-1],tracks_dict[ids[i+1]][0]))

    # giving new_ids to tracks (if we don't do this "return_track_array" gets ugly
    old_ids=list(tracks_stitched.keys())
    tracks_stitched={i:values for i,values in zip(range(len(tracks_stitched.keys())),tracks_stitched.values())}
    gaps={i:values for i,values in zip(range(len(gaps.keys())),gaps.values())}

    # replacing overlapping points
    # points that appear in the same frame are replaced py the position between them
    for key,values in tracks_stitched.items():
        values_s=sorted(values,key=lambda x: x[-1]) # sorting the points according to their frame
        points=np.array(values_s)
        # finding the first overlapping by checking where the frame number is
        # constant for two points
        id_rm=np.where((points[1:,2]-points[:-1,2])==0)[0]
        # calculating the a new point between the points in id_rm and the points
        # directly after
        new_points=[np.mean(np.array([points[i],points[i+1]]),axis=0) for i in id_rm]
        # replacing one of them with the new point
        for j,i in enumerate(id_rm):
            points[i]=new_points[j] # replacing with mean
        # deleting the other
        points=np.delete(points,np.array(id_rm)+1,axis=0)
        # updating the dictionary
        tracks_stitched[key]=points
    
    return tracks_stitched, stitched_id, gaps, old_ids

def remove_parralle_tracks(tracks_dict,tracks_arr,end_dist=30,mean_dist=30):

    '''
    removing parallel tracks
    Parrallel tracks could indicate that one obeject has been detected as two.
    Here parallel tracks are identified by comparing the distances at the start and
    end of a track and the mean distance. At start and end detetctions must be present
    in both tracks. This also ensures that only the shorter of two tracks can be classified
    as "parallel". The "parallel" is simply deleted.

    :param tracks_dict: dictionary of the tracks as returned by read_tracks from the output text files.
    Format needs to be track_id:[[x,y,frame1],[x,y,frame2],...]
    :param tracks_arr: nan padded array as generated by return_track_array
    :param end_dist: maximal allowed distances at both start and end to classify as parallel
    :param mean_dist: maximal allowed average distance to classify as parallel
    :return: tracks_dict_filtered, tracks_dict filtered for parallel tracks
    '''

    tracks_dict_filtered=copy.deepcopy(tracks_dict)
    n_tracks=len(tracks_dict_filtered.keys()) # number of tracks
    dists=np.zeros((n_tracks,n_tracks,3))+np.nan # setting up empty distance matrix
    ids=np.array(list(tracks_dict_filtered.keys())) # list of track ids
    
    # pair wise comparison of all tracks. Start and end distance, as well
    # as average distance are calculate
    # track in rows are compared to tracks in columns. Concretely the average distance
    # is calculated only on the range of tracks in rows

    for i,tid1 in enumerate(tracks_dict_filtered.keys()):
        # i is the correct row in the tracks array, tid is the correct
        #  index in the tracks dictionary
        for j,tid2 in enumerate(tracks_dict_filtered.keys()):
            if i!=j: # don't calculate distance with itself
                # rane of frames coverd by tid1 track
                first_frame = int(tracks_dict_filtered[tid1][0][-1])
                last_frame =  int(tracks_dict_filtered[tid1][-1][-1])
                # all points in this range
                ps1=tracks_arr[first_frame:last_frame+1,i,:]
                ps2=tracks_arr[first_frame:last_frame+1,j,:]
                # distance in end and start points
                # returns nan if track ends/starts don't appear in the same frames
                dists[i,j,0]=np.linalg.norm(ps1[0]-ps2[0]) # start points
                dists[i,j,1]=np.linalg.norm(ps1[-1]-ps2[-1]) # endpoints
                # average distance of all points
                dists[i,j,2]=np.nanmean(np.linalg.norm(ps1-ps2,axis=1))

    # removal conditions:
    # logical and on all conditions --> start,end and mean distance must all
    # be smaller then their threshold values
    exclude=(dists[:,:,0]<end_dist)*(dists[:,:,1]<end_dist)*(dists[:,:,2]<mean_dist)
    # all rows in the distance matrix where exclude is true represent "parallel" tracks
    ex_ids=np.unique(ids[np.where(exclude)[0]]) # list of track ids classified as parallel
    # deleting these ids from the dictionary
    for i in ex_ids:
        del(tracks_dict_filtered[i])

    return tracks_dict_filtered
        
        
        
# some example code
if __name__=="__main__":
    file= "/media/user/7419-BE6E/tracking_scripts/tracks2.txt"
    #dims=
    tracks_dict,frame_number=read_tracks(file)
    tracks_arr=return_track_array( tracks_dict,frame_number=frame_number)
    tracks_stitched,stitched_id,gaps=stitch(tracks_dict,seconds_per_frame=30,f_max=10,speed=50,s_max=300)
    tracks_stitched_arr=return_track_array(tracks_stitched,frame_number=frame_number)
    gaps_arr=return_track_array(gaps,frame_number=frame_number)
    
    tracks_f=remove_parralle_tracks(tracks_stitched,tracks_stitched_arr,end_dist=30,mean_dist=30)
    tracks_f_arr=return_track_array(tracks_f,frame_number=frame_number)
    
    folder="/media/user/7419-BE6E/tracking_scripts/"
    root_im="/media/user/7419-BE6E/ants2/img%s.jpg"
    frames=list(range(55))
    
    l=[str(n).zfill(6) for n in frames]
    
    
    images=[plt.imread(root_im%i) for i in l]
    figures=make_tracks_video(tracks_arr,frames,folder,show_trailing=3,
                              dims=(720, 1280),images=images,vid_format="gif",name="Aout")
    figures=make_tracks_video(tracks_stitched_arr,frames,folder,gaps=gaps_arr,show_trailing=3,
                              dims=(720, 1280),images=None,vid_format="gif",name="Aout2")
    figures=make_tracks_video(tracks_f_arr,frames,folder,gaps=gaps_arr,show_trailing=3,
                              dims=(720, 1280),images=images,vid_format="gif",name="Aout3")
    
    file2= "/media/user/7419-BE6E/tracking_scripts/tracks2_stitched.txt"
    write_tracks2(tracks_f_arr,file2)
    
    
    read_tracks(file2)
# Todo:
# make stitching order faster by first building a dictionary from stitching list.
# pack replacing overalpping points into a function and make faster by using a list of instances where overlapping
#actually occured, but maybe we need sorting???
# make simple distance marix calculation faster  with numpy//// maybe just leave like this for readability reasons
# maybe remove region selection in  reomve_paralle_tracks--> but could be faster, so maybe dont
# fix gaps id assignemnet in main programm set