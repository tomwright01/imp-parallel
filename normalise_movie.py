import os
import numpy as np
import argparse
import multiprocessing
import matplotlib.pyplot as plt

def normalise_and_plot_frame(frame_idx, frame_data, output_dir):
    """Function to normalise an image
    Details: 
    Scales all values to be in a range 0-255
    
    frame_idx: value is returned unchanged
    frame_data: a numpy NxM array
    output_dir: the directory to write the output frames"""
    
    assert len(frame_data.shape) < 3, "Only grayscale images are supported"
    
    print("Processing frame {}".format(frame_idx))

    #setup the filename to write too
    out_filename = "frame_{0:0>3}.png".format(frame_idx)
    out_filepath = os.path.join(output_dir, out_filename)

    # convert frame to type float
    frame_data = frame_data.astype(np.float)
    try:
        frame_data = frame_data - frame_data.min()
        frame_data = frame_data / frame_data.max()
        frame_data = frame_data * 255
    except (AttributeError, TypeError):
        raise AssertionError("Expected frame_data to be a numeric ndArray")
    
    # use matplot lib to visualise the frame
    plt.imshow(frame_data)
    plt.gray()
    try:
        plt.savefig(out_filepath)
    except IOError:
        raise IOError("Failed to write output file: {}".format(out_filepath))
    plt.clf()

    #time.sleep(1)
    return (frame_idx,frame_data)
    
# python magic so it works when run as an app, but not when loaded in a module
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Convert a video file to numpy array')
    parser.add_argument('-i','--inputFile', required=True,
                        help='full path to numpy file to process')
    parser.add_argument('-o', '--outputDir', required=True,
                        help='The directory to save the processed images, will be created if it doesn\'t exist')
    parser.add_argument('--outputFile', required=False,
                        help='The filename to save the processed numpy array')    
    parser.add_argument('-n', '--numProcessors',type=int, required=False,
                        help='Number of processors to use. ' + \
                        "Default for this machine is %d" % (multiprocessing.cpu_count(),),
                        default=multiprocessing.cpu_count())
    
    args = parser.parse_args()
    
    if args.numProcessors < 1:
        sys.exit('Number of processors to use must be greater than 0')    
       
    # confirm the output directory exists, create it if not
    if not os.path.exists(args.outputDir):
        try:
            os.makedirs(args.outputDir)
        except IOError:
            sys.exit("Failed to create output dir: {}".format(args.outputDir))
    # try to load the data file    
    try:    
        data = np.load(args.inputFile)
    except IOError:
        sys.exit('Failed to load file: {}'.format(args.inputFile))
        
    if len(data.shape) < 3:
        sys.exit('Expected an ndArray of shape nFrames x nRows x nCols')

    # preallocate the output array to the same shape as data
    # with datatype unsigned integer
    output = np.empty_like(data, dtype=np.uint8)    
        
    # start the process pool
    pool = multiprocessing.Pool(args.numProcessors)
    
    # Build task list
    tasks = []
    
    frame_idx = 0
    for frame_idx in range(data.shape[0]):
        tasks.append((frame_idx, data[frame_idx,:,:], args.outputDir))
    
    # Run tasks
    results = [pool.apply_async( normalise_and_plot_frame, t ) for t in tasks]

    # Process results
    for result in results:
        (frame_idx, frame_data) = result.get()
        output[frame_idx,:,:] = frame_data

    pool.close()
    pool.join()    
    
    # save the output file if filename is supplied
    if args.outputFile:
        try:
            np.save(args.outputFile, output)
        except IOError:
            sys.exit('Failed to write output file: {}'.format(args.outputFile))