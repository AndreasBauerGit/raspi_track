'''
functions for normalization, thresholding, segmentation and detection
'''


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import warnings
from scipy.ndimage import distance_transform_edt
import copy
from scipy.optimize import fmin
from scipy.ndimage.filters import gaussian_filter as gaussian
from scipy.ndimage import label,center_of_mass,labeled_comprehension


def normalize(image,lb=0.1,ub=99.9):

    '''
    normalizies an image to a range from 0 and 1 and cuts of extreme values
    e.g. lower then the 0.1 percentile and higher then 99.9  percentile

    :param image: 2 dimensional ndarray representing an image
    :param lb: percentile of lower bound for filter
    :param ub: percentile of upper bound for filter
    :return: image, nomralized image; ndarray
    '''

    image = image - np.percentile(image, 0.01)  # 1 Percentile
    image = image / np.percentile(image, 99.9)  # norm to 99 Percentile
    image[image < 0] = 0.0
    image[image > 1] = 1.0
    return image

def segementation_sd(image,f=5,mask_area=None,min_treshold=None):

    '''
    segmentation based on distance from mean in terms of standard deviations
    :param image: 2 dimensional array representing an image
    :param f: factor of how many standard deviations from the mean the threshold will be set
    :param mask_area: optional area  which to use for thresholding and segmentation
    :param min_threshold: optional minimal value for the threshold
    :return: mask, 2 dimensional boolean array, representing  the area of detected objects
             thres, threshold used for segmentation
    '''

    if isinstance(mask_area,np.ndarray): # checking if mask for dish area is provided
        thres = np.mean(image[mask_area]) + f * np.std(image[mask_area])
    else: # else using the complete image for segmentation
        warnings.warn("no valid mask for dish area found, using the whole image")
        thres = np.mean(image) + f * np.std(image)
    if min_treshold: # cheking for minimal threshold
        if min_treshold>thres: # take min_threshold as new threshold
            thres=min_treshold

    mask=image>thres # segmentation
    mask[~mask_area]=0
    #labeled_objects = label(mask)  # labeling   ## maybe introduce filtering for small objects
    #labeled_objects_filtered = remove_small_objects(labeled_objects, min_size=min_cell_size)  # filtering
    return mask,thres





def detection(mask, min_size=0):

    '''
    finds center of objects in a mask and filters for the total size of one object.
    :param mask: boolean mask of object areas
    :param min_size: minimal size of allowed objects. Any area below this size will not return a detection
    :return: detections_f: list of x,y positions of the center of objects
             labels: labeled mask (2 dimensional array where the background is zero and the area of each object
             is filled by unique integers)
    '''
    # labeling: each isolated area is assigned a unique integer number
    labels,n_objects=label(mask)
    # returning if no labels at all are found
    if n_objects==0:
        return np.array([]), labels
    # detection: finding the center of all objects
    id_objects=np.arange(1,n_objects+1,dtype=int)  # list of the ids if labels
    detections=np.array(center_of_mass(mask,labels,index=id_objects)) # center of mass of all objects
    # filtering small objects
    area=labeled_comprehension(mask,labels,index=id_objects,func=len,
                               out_dtype=float,default= 0) # area of all objects
    detections_f=detections[area>min_size] # exclusion of anything with area smaller then min_size

    return detections_f,labels

def custom_threshold(img, size=None, spacing_factor=1,mask_area=None):

    '''
    thresholding method based on finding the the minimum of the second derivative in the histogram of pixel
    intensities. This histogram should have a large gaussian shaped part representing the background pixels.
    The minimum of the second derivative is right at the the end of this curve. Naturally some background pixels
    are included when thresholding with this method.
    :param img: 2 dimensional array representing an image
    :param size: sample size for kde estimation of the distribution of pixel intensities
     not recommended to go above 20000 due to high calculation time
    :param spacing_factor: additional spacing factor. Will add spacing_factor* distance to peak intensity probability
    this supposed to remove some weak signals that are not desired for some reason
    :param mask_area: mask of the part of the image that should be used for segmentation and thresholding
    :return: mask: dimensional boolean array, representing  the area of detected objects
             thresh_new: threshold used for segmentation

    '''

    if size:
        size1=size
    elif isinstance(mask_area, np.ndarray):
        size1=len(img[mask_area].flatten()) # use all pixels if no size argument is given
    else:
        size1 = img.size
    if size1>20000:
        warnings.warn("your sample size is very large (%s). Calculation might be very slow" % size1)
    if isinstance(mask_area, np.ndarray): # using mask for subset of pixels
        mu, std = np.mean(np.ravel(img[mask_area])), np.std(np.ravel(img[mask_area]), ddof=1)
        flat_img1= img[mask_area].flatten()

    else: #suing full image
        mu, std = np.mean(np.ravel(img)), np.std(np.ravel(img), ddof=1)
        flat_img1 = img.flatten()

    # using only small subset of pixel values to reduce calculation cost
    subset = np.random.choice(flat_img1, size1, replace=False)
    min_value = np.min(subset)
    max_value = np.max(subset)
    xs = np.linspace(min_value, mu + 5 * std,
                     size1)  # maximum value is "far beyond" intensities that are expected for the background
    # kerel density estimation to smoot the histogramm
    kde_sample_scott = gaussian_kde(subset, bw_method='scott')
    kd_estimation_values = kde_sample_scott(xs)  # getting values of the kde

    # calculating first and second derivative with rough approximation
    first_derivative = kd_estimation_values[:-1] - kd_estimation_values[1:]
    second_derivative = first_derivative[:-1] - first_derivative[1:]

    ### strategie: finding maximum of second derivative, will always be just at the bottom of a guass distributed curve
    max_pos_1 = np.where(kd_estimation_values == np.max(kd_estimation_values))[0][
        0]  ## maximum of histogramm (not actually needed)
    max_pos = xs[max_pos_1]
    thresh_1 = np.where(second_derivative == np.max(second_derivative))[0]  # maximum of second derivative
    thresh = xs[thresh_1]  # threshold in terms of intensity
    thresh_new = max_pos + np.abs(
        max_pos - thresh) * spacing_factor  # using absolute distance to maximum to effectively use spacing factor
    # spacing_factor: factor to get wider spacing, will generally be set to 1

    mask=img  >thresh_new
    return  mask,thresh_new


def segmentation_and_detection(image,mask,sd_threshold=5,min_size=50,s1=12,s2=5,min_treshold=None):
    '''
    function to perform all necessary steps for segmentation and detection.
    :param image: image: 2 dimensional array representing an image
    :param mask: optional area  which to use for thresholding and segmentation
    :param sd_threshold:  factor of how many standard deviations from the mean the threshold will be set
    :param min_size: minimal size of allowed objects. Any area below this size will not return a detection
    :param s1: lower standard deviation for difference of gaussian filter
    :param s2: higher standard deviation for difference of gaussian filter
    :param min_treshohld: optional minimal value for the threshold
    :return: detections: list of x,y positions of the center of detected objects
    '''
    image1 = normalize(image, lb=0.1, ub=99.99) # normalizing
    img_gauss = gaussian(image1, sigma=s1) - gaussian(image1, sigma=s2)  # broadband filter
    mask_bugs, thres = segementation_sd(img_gauss, f=sd_threshold, mask_area=mask,min_treshold=min_treshold) # segementation
    detections, labels = detection( mask_bugs, min_size=min_size)
    return detections





# some example code with plotting options
if __name__=="__main__":
    image=plt.imread("/home/andy/Desktop/rasperry_project_bugs/rec0014.jpeg")
    #mask=np.load("/home/andy/Desktop/rasperry_project_bugs/mask.npy")
    # grey scale conversion, could use some wights here though

    if len(image.shape)==3:
        image=np.mean(image,axis=2)
    mask=np.load("/home/andy/Desktop/rasperry_project_bugs/mask.npy")



    image1 = normalize(image,lb=0.1,ub=99.9)

    #plt.figure()
    #plt.imshow(image)
    #plt.figure()
    #plt.imshow(image1)

    img_gauss=gaussian(image1,sigma=12)-gaussian(image1,sigma=5)  ## note: this depends on the reesolution!!!
    #plt.figure()
    #plt.imshow(img_gauss)
    #plt.figure();
    #plt.hist(img_gauss[mask],bins=100)
    #plt.figure();
    #plt.imshow(mask)


    # segementation
    #thres=threshold_otsu(img_gauss[mask]) ## otsu not so great: to little signal values
    #mask_bugs=img_gauss>thres
    #plt.figure()
    #plt.imshow(mask_bugs)

    # this works reasonabbly well..
    #thres=np.mean(img_gauss[mask]) +5* np.std(img_gauss[mask])
    #mask_bugs=img_gauss>thres
    #plt.figure()
    #plt.imshow(mask_bugs)

    mask_bugs,thres = segementation_sd(img_gauss,f=5,mask_area=mask)
    detections,labels=detection(mask_bugs,min_size=60)

    

    
    
   # plt.figure()
   # plt.imshow(mask_bugs)
    #mask_bugs,thres =custom_threshold(img_gauss,size=10000, spacing_factor=5,mask_area=mask)
    #plt.figure()

   
    labels_show=np.zeros(labels.shape)+ np.nan
    labels_show[labels>0]=labels[labels>0]
    plt.figure()
    plt.imshow(image)
    plt.imshow(labels_show,alpha=0.5)
    for i,d in enumerate(detections):
        plt.text(d[1],d[0],str(i))

    plt.show()
    #plt.figure();
    #plt.imshow(labels)
    #for px,py in detections:
     #   plt.plot(py,px,"o")



