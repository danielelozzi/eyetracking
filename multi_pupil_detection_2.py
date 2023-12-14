#from parfor import pmap
#import skimage
#import av
#import numpy as np
#import matplotlib.pyplot as plt
#import pandas as pd
#import time

#video_path = 'eyes.mp4'


def load_video(path):
    import av
    v = av.open(path)

    frames_r = []
    frames_l = []
    frames = []
    for frame in v.decode(video=0):
        # Decode video frame, and convert to NumPy array in BGR pixel format (use BGR because it used by OpenCV).
        frame = frame.to_ndarray(format='gray')  # For Grayscale video, use: frame = frame.to_ndarray(format='gray')
        frames.append(frame)
        lenght = int(frame.shape[1]/2)
        #frame = frame[:,:,0]
        frame_r = frame[:,0:lenght]
        frame_l = frame[:,lenght:]
        frames_r.append(frame_r)
        frames_l.append(frame_l)

    return frames

def multi_pupil_detection(frame):
    import numpy as np
    import skimage
    import pandas as pd
    import time
    # provare con immagini a 4 bit
    # escludere la  o fondere le due immagini
    # gaussian blur + apertura chiusura per rumore
    # scalare valori px in base a dimensione immagine
    # rescaling luminosità + multiostu classes sulla base della luminosità
    t1 = time.time_ns()
    lenght = int(frame.shape[1]/2)
    #frame = frame[:,:,0]
    threshold = skimage.filters.threshold_multiotsu(frame, classes=5)
    frame_r = frame[:,lenght:]
    try:
        #threshold = skimage.filters.threshold_multiotsu(frame_r, classes=5)
        image_mask = frame_r < threshold[0]
        image_mask = image_mask * 1
        #plt.imshow(image_mask, cmap='gray')
        labels_r = skimage.measure.label(image_mask)
        props = skimage.measure.regionprops_table(labels_r, properties=('centroid',
                                                                        'orientation','axis_major_length','axis_minor_length', 'bbox', 'image', 'label',
                                                                        'area_bbox', 'area', 'eccentricity'))

        props = pd.DataFrame(props)
        props = props.loc[
            (props['axis_minor_length'] > 0) & (props['axis_major_length'] > 0) & (props['area'] > 50) & (props['area'] < 1000)].sort_values(
            'eccentricity')
        eccentricity_r = props['eccentricity'].iat[0]
        xmin_r = props['bbox-0'].iat[0]
        xmax_r = props['bbox-2'].iat[0]
        ymin_r = props['bbox-1'].iat[0]
        ymax_r = props['bbox-3'].iat[0]
        pupil_r = frame_r[xmin_r:xmax_r, ymin_r:ymax_r]
        canny_r = skimage.feature.canny(pupil_r)
        diameter_r = props['axis_major_length'].iat[0]
    except:
        diameter_r = 0
        eccentricity_r = 1

    frame_l = frame[:,0:lenght]
    try:
        #threshold = skimage.filters.threshold_multiotsu(frame_l, classes=5)
        image_mask = frame_l < threshold[0]
        image_mask = image_mask * 1
        #plt.imshow(image_mask, cmap='gray')
        labels_l = skimage.measure.label(image_mask)
        props = skimage.measure.regionprops_table(labels_l, properties=('centroid',
                                                                        'orientation','axis_major_length','axis_minor_length', 'bbox', 'image', 'label',
                                                                        'area_bbox', 'area', 'eccentricity'))

        props = pd.DataFrame(props)
        props = props.loc[
            (props['axis_minor_length'] > 0) & (props['axis_major_length'] > 0) & (props['area'] > 50) & (props['area'] < 1000)].sort_values(
            'eccentricity')
        eccentricity_l = props['eccentricity'].iat[0]
        xmin_l = props['bbox-0'].iat[0]
        xmax_l = props['bbox-2'].iat[0]
        ymin_l = props['bbox-1'].iat[0]
        ymax_l = props['bbox-3'].iat[0]
        pupil_l = frame_l[xmin_l:xmax_l, ymin_l:ymax_l]
        canny_l = skimage.feature.canny(pupil_l)
        diameter_l = props['axis_major_length'].iat[0]
    except:
        diameter_l = 0
        eccentricity_l = 1

    diameter = max(diameter_l,diameter_r)
    #if (diameter_r>diameter_l) and (eccentricity_r<eccentricity_l):
    if (eccentricity_r<eccentricity_l):
        choose = 'right'
        pupil = pupil_r
    #elif (diameter_r<diameter_l) and (eccentricity_r>eccentricity_l):
    elif (eccentricity_r>eccentricity_l):
        choose = 'left'
        pupil = pupil_l
    elif diameter_r==diameter_l:
        choose = 'equal'
        #pupil = (pupil_r,pupil_r)
    else:
        choose = 'error'

    t2 = time.time_ns()
    duration = np.round((t2-t1)*10**(-9),decimals=3)
    #plt.imshow(frame)
    #plt.imshow(pupil)

    return diameter,duration,diameter_l,diameter_r,choose, eccentricity_r, eccentricity_l
    #return pupil



def fast_multi_pupil_detection(frame):
    import numpy as np
    import skimage
    import pandas as pd
    import time
    # provare con immagini a 4 bit
    # escludere la  o fondere le due immagini
    # gaussian blur + apertura chiusura per rumore
    # scalare valori px in base a dimensione immagine
    # rescaling luminosità + multiostu classes sulla base della luminosità
    t1 = time.time_ns()
    lenght = int(frame.shape[1]/2)
    #frame = frame[:,:,0]
    threshold = skimage.filters.threshold_otsu(frame)
    frame[frame>threshold] = 256
    threshold = skimage.filters.threshold_otsu(frame)
    frame_r = frame[:,lenght:]
    try:
        #threshold = skimage.filters.threshold_multiotsu(frame_r, classes=5)
        image_mask = frame_r < threshold
        image_mask = image_mask * 1
        #plt.imshow(image_mask, cmap='gray')
        labels_r = skimage.measure.label(image_mask)
        props = skimage.measure.regionprops_table(labels_r, properties=('centroid',
                                                                        'orientation','axis_major_length','axis_minor_length', 'bbox', 'image', 'label',
                                                                        'area_bbox', 'area', 'eccentricity'))

        props = pd.DataFrame(props)
        props = props.loc[
            (props['axis_minor_length'] > 0) & (props['axis_major_length'] > 10) & (props['area'] > 100) & (props['area'] < 1000)].sort_values(
            'eccentricity')
        eccentricity_r = props['eccentricity'].iat[0]
        xmin_r = props['bbox-0'].iat[0]
        xmax_r = props['bbox-2'].iat[0]
        ymin_r = props['bbox-1'].iat[0]
        ymax_r = props['bbox-3'].iat[0]
        pupil_r = frame_r[xmin_r:xmax_r, ymin_r:ymax_r]
        canny_r = skimage.feature.canny(pupil_r)
        diameter_r = props['axis_major_length'].iat[0]
    except:
        diameter_r = 0
        eccentricity_r = 1

    frame_l = frame[:,0:lenght]
    try:
        #threshold = skimage.filters.threshold_multiotsu(frame_l, classes=5)
        image_mask = frame_l < threshold
        image_mask = image_mask * 1
        #plt.imshow(image_mask, cmap='gray')
        labels_l = skimage.measure.label(image_mask)
        props = skimage.measure.regionprops_table(labels_l, properties=('centroid',
                                                                        'orientation','axis_major_length','axis_minor_length', 'bbox', 'image', 'label',
                                                                        'area_bbox', 'area', 'eccentricity'))

        props = pd.DataFrame(props)
        props = props.loc[
            (props['axis_minor_length'] > 0) & (props['axis_major_length'] > 10) & (props['area'] > 100) & (props['area'] < 1000)].sort_values(
            'eccentricity')
        eccentricity_l = props['eccentricity'].iat[0]
        xmin_l = props['bbox-0'].iat[0]
        xmax_l = props['bbox-2'].iat[0]
        ymin_l = props['bbox-1'].iat[0]
        ymax_l = props['bbox-3'].iat[0]
        pupil_l = frame_l[xmin_l:xmax_l, ymin_l:ymax_l]
        canny_l = skimage.feature.canny(pupil_l)
        diameter_l = props['axis_major_length'].iat[0]
    except:
        diameter_l = 0
        eccentricity_l = 1
    print(diameter_l,diameter_r)
    diameter = max(diameter_l,diameter_r)
    #if (diameter_r>diameter_l) and (eccentricity_r<eccentricity_l):
    if (eccentricity_r<eccentricity_l):
        choose = 'right'
        pupil = pupil_r
    #elif (diameter_r<diameter_l) and (eccentricity_r>eccentricity_l):
    elif (eccentricity_r>eccentricity_l):
        choose = 'left'
        pupil = pupil_l
    elif diameter_r==diameter_l:
        choose = 'equal'
        #pupil = (pupil_r,pupil_r)
    else:
        choose = 'error'

    t2 = time.time_ns()
    duration = np.round((t2-t1)*10**(-9),decimals=3)
    #plt.imshow(frame)
    #plt.imshow(pupil)
    diameter = np.round(diameter,decimals=1)
    diameter_r = np.round(diameter_r,decimals=1)
    diameter_l = np.round(diameter_l,decimals=1)

    return diameter,duration,diameter_l,diameter_r,choose, eccentricity_r, eccentricity_l
    #return frame

#frames = load_video(video_path)
##diameter_list = pmap(multi_pupil_detection,frames)
#plt.plot(diameter_list)
