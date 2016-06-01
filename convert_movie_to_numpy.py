import cv2
import os
import numpy as np
import argparse


def convert_avi_to_numpy(avi_file, save_array=False):
    """Read an avi file and convert to a numpy array, optionally save the array to a file
    The array will be of shape nFrames,Rows,Cols
    Note: only greyscale movies are supported, colour formats will be converted to grey using the G channel
    """
    
    RGB=False #indicator if video is in RGB format, in which case only use G channel
    
    cap = cv2.VideoCapture(avi_file)
    if not cap.isOpened():
        raise IOError("Failed to open file:%s", avi_file)

    # cv2 constants are in the cv2.cv module
    # get size information about the movie
    nframes = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    frameheight = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    framewidth = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    
    #preallocate a numpy array
    data = np.empty([nframes, frameheight, framewidth],dtype=np.uint8)
  
    ret, frame = cap.read() #get the first frame
    if len(frame.shape)>2:
        #frames are RGB format, using only G channel
        logger.debug('Video is in RGB format, using only G channel')
        RGB=True
        
    while(ret):
        frame_idx = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
        if RGB:
            # only use one layer of each frame for the output
            data[frame_idx - 1,:,:] = frame[:,:,1]
        else:
            data[frame_idx -1 ,:,:] = frame[:,:]
        ret, frame = cap.read()        
    cap.release()    
    
    if save_array:
        # save the numpy array to a file with the same name as the input file
        filename_output = os.path.splitext(avi_file)[0] + '.npy'
        np.save(filename_output, data)
        
        
# python "magic" to make file run from command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Convert a video file to numpy array')
    parser.add_argument('-f','--filename',help='full path to movie file to process', default='data/slo.avi')
    args = parser.parse_args()
    
    convert_avi_to_numpy(args.filename, True)
    
