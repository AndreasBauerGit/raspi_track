'''
interactive plot/ graphical user interface to select an area for segmentation, and appropriate thresholds.
'''


from matplotlib.widgets import PolygonSelector, Button,Slider
from matplotlib import path
from matplotlib.image import AxesImage
from matplotlib.backend_bases import MouseEvent
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import numpy as np
import warnings
import os
from segementation_bugs import *


def assert_mask(mask):
    '''
    hecks if mask exists.Tries to load from current working directory if it doesn't
    and prints warning messages.

    :param mask: np.darray representing the mask of the dish area
    :return: the mask and a boolean if the the mask could be loaded
    '''

    if not isinstance(mask, np.ndarray):
        print("no current mask for dish area found")
        path_mask=os.path.join(os.getcwd(),"mask.npy")
        if os.path.exists(path_mask): # checks if mask can be loaded from current directory
            mask=np.load(path_mask)
            print("load mask from "+ path_mask)
            return mask,True
        else:
            warnings.warn("couldn't find current selection for the dish area"
                          "or load a mask file")
            return 0, False
    else:
        return mask, True
   
    
    
def set_button(pos,disp_name,fun,font_size=10):

    '''
    helper function to set buttons, returns the axes object of the button and the widget
    object

    :param pos: x,y postion of buttom axes left corner and length and hight dx,dy as a list
    :param disp_name: string of the button name
    :param fun: function that is executed on button click
    :return: ax_button: the axes of the button
             button_widget: the button object
    '''

    ax_button = plt.axes(pos)
    ax_button.xaxis.set_ticks_position('none')
    ax_button.yaxis.set_ticks_position('none')
    ax_button.set_xticks([])
    ax_button.set_yticks([])
    button_widget = Button(ax_button ,disp_name )
    button_widget.on_clicked(fun)
    button_widget.label.set_fontsize(font_size)
    return ax_button,button_widget


def ask_file(event):

    '''
    opens a tk inter file browser. The selected file is loaded as an image to the window.
    Additionally the filtered image is calculated immediately.
    :param event:
    :return:
    '''

    # opens dialog box for image selection
    global fig, ax, image, pix, image_f, im, lasso2
    root = tk.Tk()
    root.withdraw() #
    file_path = filedialog.askopenfile()
    # loading the image
    image = plt.imread(file_path.name)
    # grey scale conversion, one could use some wights here
    if len(image.shape) == 3:
        #image=image[:,:,0]
        image = np.mean(image, axis=2)

    # filtering image
    image_f=normalize(image,lb=0.1,ub=99.9) # normalization to a range  from 0 to 1
    image_f=gaussian(image_f,sigma=s1)-gaussian(image_f,sigma=s2) # difference of gaussian filtering

    # displaying unfiltered image first
    im=ax.imshow(image) # you can choose a diffrent color map here
    fig.canvas.draw_idle() #plot update


    # coordinate of the image, later needed for selction
    pixx = np.arange(image.shape[0])
    pixy = np.arange(image.shape[1])
    xv, yv = np.meshgrid(pixy, pixx)
    pix = np.vstack((xv.flatten(), yv.flatten())).T


    # (re)initialize polygon selector
    #lasso2.disconnect_events() seems to have some kind of bug...
    lasso2 = PolygonSelector(ax, select_mask)  # selector on bf image




def select_mask(verts):
    '''
    Constructs and displays the mask of the dish area from the nodes slected by the polygone selector.

    :param verts: vertices from the polygone selector
    :return:
    '''
    global mask,im_mask,mask_show,verts_out
    if len(verts)==0:
        return # workaround when deleting old mask
    if not isinstance(pix,np.ndarray): #error handling if pix is not caclulated (e.g when no image is selected)
        warnings.warn("could not find coordinates of an image\n"
                 "make sure to select and display an image file")
        return


    verts_out = verts # vertices of curve drawn by lasso selector
    p = path.Path(verts)
    #retrieving mask
    ind = p.contains_points(pix, radius=1) # all points inside of the selected area
    mask = np.reshape(np.array(ind), image.shape)
    # displaying the mask overlayed on image
    mask_show=np.zeros(mask.shape)+np.nan
    mask_show[mask]=1
    im_mask=ax.imshow(mask_show,alpha=0.2)
    fig.canvas.draw_idle()



# deleting the selected mask with right click
def delete_mask(event):

    '''
    deletes the mask of the dish area or the mask for bug selection if you click with the right mouse button
    :param event:
    :return:
    '''

    global mask,im_mask,lasso2,mask_show,im_mask_bugs

    # deleting mask of dish area if exists
    if event.button==3 and event.inaxes==ax and isinstance(mask,np.ndarray) and isinstance(im_mask,AxesImage) : # only on right click, only if clicked in main axes
        # chek if mask is already selected
        mask=0 # reset mask
        mask_show=0
        im_mask.remove() # remove displaying mask
        im_mask=0
        lasso2._xs = []
        lasso2._ys = []
        lasso2._draw_polygon() # removes display of old verts
        lasso2 = PolygonSelector(ax, select_mask) # reinitalizing the selector
        fig.canvas.draw_idle()  # plot update
        #print('you pressed', event.button, event.xdata, event.ydata)
    # deleting mask of bug segemetnation if exists
    # only on right click, only if clicked in main axes or in slider axis
    if event.button == 3 and isinstance(mask_bugs,np.ndarray) and isinstance(im_mask_bugs,AxesImage) and (event.inaxes==ax or event.inaxes == ax_slider):
        im_mask_bugs.remove()#
        im_mask_bugs=0
        fig.canvas.draw_idle()  # plotupdate





def save_mask(event):

    '''
     saves the mask to the currentworking directors as mask.npy.

    :param event:
    :return:
    '''
    ## this will autmatically overide
    file=os.path.join(os.getcwd(),"mask.npy")
    np.save(file,mask)
    print("mask saved to " + file)






### button to show filtering (could alos be done automatically
def show_filtered_image(event):

    '''
    shows the filtered image or the original image. If you press the button once, the
    respectively other option appears on the button.
    :param event:
    :return:
    '''


    global sfi_button,im_f,im,im_mask,lasso2

    # disconnects and stops the lassoselecto
    lasso2.disconnect_events()
    # showing the filtered image
    if sfi_button.label.get_text()=="show filtered\nimage": # checks the text displayed on the button
        im.remove()# removing unfiltered image
        im_f=ax.imshow(image_f)# showning filterd image
        lasso2 = PolygonSelector(ax, select_mask)  # reinitalizing the selector
        if isinstance(mask_show,np.ndarray): # showing mask if already selected
            im_mask=ax.imshow(mask_show, alpha=0.2)
        sfi_button.label.set_text("show unfiltered\nimage")
        fig.canvas.draw_idle()  # plot update
        return

    # showing the unfilterd image
    if sfi_button.label.get_text()=="show unfiltered\nimage": # checks the text displayed on the button
        im_f.remove()  # removing unfiltered image
        im = ax.imshow(image)  # showing filterd image
        lasso2 = PolygonSelector(ax, select_mask)  # reinitalizing the selector
        if isinstance(mask_show, np.ndarray):  # showing mask if alaready selected
            im_mask = ax.imshow(mask_show, alpha=0.2)
        sfi_button.label.set_text("show filtered\nimage")
        fig.canvas.draw_idle()  # plot update
        return


def update_segmentation(val): ## note also disables the polygone selctor

    '''
    Function to perform segmentation adn display the mask. This function is called when the threshold
    slider or the show detections button is pressed.
    :param val: value of the segmentation threshold
    :return:
    '''

    global txt,ax,mask_bugs,mask,im_mask_bugs,im_mask,segmentation_factor
    
    segmentation_factor=val
    if not isinstance(pix, np.ndarray):  # check if an image is selected
        warnings.warn("could not find coordinates of an image\n"
                      "make sure to select and display an image file")
        return
    # checking if disk area mask is selected or can be loaded from the current working directory
    mask,mask_bool=assert_mask(mask)
    if not mask_bool:
        return

    # segmentation
    mask_bugs,thresh=segementation_sd(image_f,f=segmentation_factor,mask_area=mask) #segemtnation with new values

    #updating display
    if isinstance(im_mask,AxesImage): # remove other mask from display
        im_mask.remove()
        im_mask=0

    # clearing previous mask of bugs
    if isinstance(im_mask_bugs,AxesImage):
        im_mask_bugs.remove()
        im_mask_bugs=0

    # showing the new mask from segmentation
    mask_bugs_show=np.zeros(mask_bugs.shape)+np.nan
    mask_bugs_show[mask_bugs]=1
    im_mask_bugs=ax.imshow(mask_bugs_show,alpha=0.7,cmap=cmap_mask)

    # updating the text displaying the formula for the threshold
    txt.remove()
    txt=plt.text(0, 1.5,"thresh: %.2f=mean + %.2f * sd"%(np.round(thresh,2),np.round(val,2))
      ,transform = ax_slider_tresh.transAxes) ### get better text position

    # disabeling the lassoselector
    lasso2.set_visible(False)
    lasso2.set_active(False)
    fig.canvas.draw_idle()  # plot update



def show_dections(event):

    '''
    showing the detections. Every bug is displayed as a dot. This function is called when the
    "show detections" button is pressed or when the "min size" slider is used
    :param event:
    :return:
    '''

    global txt,ax,mask_bugs,mask,im_mask_bugs,im_mask,detections,minsize


    #retreiving the value of minsize if the slider is used
    if type(event)!=MouseEvent:
        minsize=event
    event_cp=event
    if not isinstance(pix, np.ndarray):  # check if image is selected
        warnings.warn("could not find coordinates of an image\n"
                      "make sure to select and display an image file")
        return

    # checking if disk area mask is selected or can be loaded from the current working directory
    mask,mask_bool=assert_mask(mask)
    if not mask_bool:
        return

    # checking if segmentation has already been performed.
    # if not segmentation is performed with the default or the latest selected value of the
    # segmentation factor
    if not isinstance(mask_bugs,AxesImage):
        warnings.warn("clouldnt find mask from segmentation, performing segmentation" 
                      "with factor %f"%segmentation_factor)
       # segmentation
        mask_bugs, thresh = segementation_sd(image_f, f=segmentation_factor, mask_area=mask)  # segemtnation with new values
        # updating segmentation mask
        mask_bugs_show=np.zeros(mask_bugs.shape)+np.nan
        mask_bugs_show[mask_bugs]=1
        im_mask_bugs=ax.imshow(mask_bugs_show,alpha=0.7,cmap=cmap_mask)

    # deleting previous detections:
    if isinstance(detections,np.ndarray) or isinstance(detections,list):
        for l in ax.lines:
            l.remove()#removing all plotted points
        for t in ax.texts [1:]:
            t.remove() # removing all texts except an empty initial one
            
        detections=0 #resetting detections list

    # performing detection
    detections,labels=detection(mask_bugs, min_size=minsize)

    # dsiplaying the new detections with a dot and a number
    offset=0.2 # text is moved a bit out of the way
    for i,(x,y) in enumerate(detections):
        ax.plot(y,x,"o",markersize=3)
        ax.text(y,x,str(i))
    # disabling  the lassoselector
    lasso2.set_visible(False)
    lasso2.set_active(False)
    fig.canvas.draw_idle()






# paramteres for filtering
s1=12
s2=5

##some preset or default values variables; don't change these
pix=0
mask=0
mask_show=0
im_mask=0
im_mask_bugs=0
mask_bugs=0
detections=0
segmentation_factor=5 # default value, you can change this
minsize=0 # default value, you can change this


# custom color map for showing the mask of segmentation
cmap_mask = LinearSegmentedColormap.from_list('mycmap', ["red","white"])

# initializing the interactive window
fig, ax = plt.subplots()
txt=plt.text(0.25,0.15,str("")) # text over sliding bar
plt.subplots_adjust(left=0.25, bottom=0.25)

# adding the open image button to the window
ax_file,file_select=set_button([0.02, 0.5, 0.17, 0.05],'open image',ask_file)

# initialyzing the polygone selector
lasso2 = PolygonSelector(ax, select_mask) #

## adds functionality to delete the mask of segemntation or the dish area on rioght mouse click
delete_mask_event = fig.canvas.mpl_connect('button_press_event', delete_mask)

# buttton so save the mask of the dish area
ax_save_changes,save_changes_button=set_button([0.02, 0.4, 0.17, 0.05],'save mask',save_mask)

# buttton to swithc between the filtered and unfiltered images
ax_sfi, sfi_button = set_button([0.02, 0.15, 0.17, 0.1], "show filtered\nimage",show_filtered_image)

# adding the slider for thresholding
ax_slider_tresh = plt.axes([0.25, 0.1, 0.65, 0.03]) # axis for the slider
slider1= Slider(ax_slider_tresh, 'threshold', 0.5, 30, valinit=5, valstep=0.01) # creates a slider object
slider1.on_changed(update_segmentation) # connects the slider to the segmentation function

# adding the "show detections button
ax_detections, detections_button = set_button([0.02, 0.3,0.17, 0.05], "show detections",show_dections)

# adding the slider for min_size
ax_slider_minsize = plt.axes([0.25, 0.05, 0.65, 0.03]) # axis for the slider
slider2= Slider(ax_slider_minsize, 'min size', 0, 200, valinit=50, valstep=1) # creates a slider object
slider2.on_changed(show_dections) # connects the slider to the show detections function

# showing the entire window
plt.show()










# todo:
# text position when segmenting is off


#bug:rectangel selector seems to get increasingly slower when image is switched multiple times
#(switching by loading new image or by displaying the filtered image

## sidenode, ther is also a "draw event, that might be nice....

# could add afunction to reactivate  polygone selector (not really necessary)
