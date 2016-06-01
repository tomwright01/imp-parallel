import cv2
import numpy as np
import argparse
import logging


def convert_numpy_to_movie(data,fname):
    """Convert a nFrames x nRows x nCols numpy array into a movie"""
    nframes, height,width = data.shape
    
    fourcc = cv2.cv.CV_FOURCC(*'I420')
    try:
        vid = cv2.VideoWriter(fname,fourcc,10,(width,height))
    except IOError:
        raise AssertionError("Failed to open file %s for writing", fname)
    
    
    for idx in range(nframes):
        frame = np.uint8(data[idx,:,:])
        frame = np.tile(frame,(3,1,1))
        frame = np.transpose(frame, (1,2,0))
        
        vid.write(frame)
    
    vid.release()  
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Convert a numpy array to a video file')
    parser.add_argument('-i', '--inputFile', required=True,
                        help='full path to movie file to process')
    parser.add_argument('-o', '--outputFile', required=True,
                        help='full path to movie file to process')
    args = parser.parse_args()
    
    try:
        data = np.load(args.inputFile)
    except IOError:
        sys.exit("Failed to open file: %s", args.inputFile)
        
    convert_numpy_to_movie(data, args.outputFile)    