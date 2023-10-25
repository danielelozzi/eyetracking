def pupil_detection(frame):
    import time
    import skimage
    import pandas as pd
    
    t1 = time.time_ns()
    try:
        threshold = skimage.filters.threshold_multiotsu(frame, classes=5)
        threshold
        image_mask = frame < threshold[0]
        #plt.imshow(image_mask, cmap='gray')
        labels = skimage.measure.label(image_mask * 1)
        props = skimage.measure.regionprops_table(labels, properties=('centroid',
                                                                      'orientation',
                                                                      'axis_major_length',
                                                                      'axis_minor_length', 'bbox', 'image', 'label',
                                                                      'area_bbox', 'area', 'eccentricity'))

        props = pd.DataFrame(props)
        props
        props = props.loc[
            (props['axis_minor_length'] > 0) & (props['axis_major_length'] > 0) & (props['area'] > 20) & (props['area'] < 800)].sort_values(
            'eccentricity')
        props
        xmin = props['bbox-0'].iat[0]
        xmax = props['bbox-2'].iat[0]
        ymin = props['bbox-1'].iat[0]
        ymax = props['bbox-3'].iat[0]
        pupil = frame[xmin:xmax, ymin:ymax]
        #plt.imshow(pupil, cmap='gray')
        diameter = props['axis_major_length'].iat[0]
    except:
        diameter = 0
    t2 = time.time_ns()
    duration = np.round((t2-t1)*10**(-9),decimals=3)

    return diameter,duration
