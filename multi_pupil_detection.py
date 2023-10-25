def multi_pupil_detection(frame):
    # provare con immagini a 4 bit
    # escludere la  o fondere le due immagini
    # gaussian blur + apertura chiusura per rumore
    # scalare valori px in base a dimensione immagine
    # rescaling luminosità + multiostu classes sulla base della luminosità
    t1 = time.time_ns()
    lenght = int(frame.shape[1]/2)
    #frame = frame[:,:,0]
    frame_r = frame[:,0:lenght]
    try:
        threshold = skimage.filters.threshold_multiotsu(frame_r, classes=5)
        threshold
        image_mask = frame_r < threshold[0]
        #plt.imshow(image_mask, cmap='gray')
        labels = skimage.measure.label(image_mask * 1)
        props = skimage.measure.regionprops_table(labels, properties=('centroid',
                                                                      'orientation','axis_major_length','axis_minor_length', 'bbox', 'image', 'label',
                                                                      'area_bbox', 'area', 'eccentricity'))

        props = pd.DataFrame(props)
        props
        props = props.loc[
            (props['axis_minor_length'] > 0) & (props['axis_major_length'] > 0) & (props['area'] > 150) & (props['area'] < 800)].sort_values(
            'eccentricity')
        props
        xmin = props['bbox-0'].iat[0]
        xmax = props['bbox-2'].iat[0]
        ymin = props['bbox-1'].iat[0]
        ymax = props['bbox-3'].iat[0]
        pupil = frame_r[xmin:xmax, ymin:ymax]
        plt.imshow(pupil, cmap='gray')
        diameter_r = props['axis_major_length'].iat[0]
    except:
        diameter_r = 0

    frame_l = frame[:,lenght:]
    try:
        threshold = skimage.filters.threshold_multiotsu(frame_l, classes=5)
        threshold
        image_mask = frame_l < threshold[0]
        #plt.imshow(image_mask, cmap='gray')
        labels = skimage.measure.label(image_mask * 1)
        props = skimage.measure.regionprops_table(labels, properties=('centroid',
                                                                      'orientation','axis_major_length','axis_minor_length', 'bbox', 'image', 'label',
                                                                      'area_bbox', 'area', 'eccentricity'))

        props = pd.DataFrame(props)
        props
        props = props.loc[
            (props['axis_minor_length'] > 0) & (props['axis_major_length'] > 0) & (props['area'] > 150) & (props['area'] < 800)].sort_values(
            'eccentricity')
        props
        xmin = props['bbox-0'].iat[0]
        xmax = props['bbox-2'].iat[0]
        ymin = props['bbox-1'].iat[0]
        ymax = props['bbox-3'].iat[0]
        pupil = frame_l[xmin:xmax, ymin:ymax]
        plt.imshow(pupil, cmap='gray')
        diameter_l = props['axis_major_length'].iat[0]
    except:
        diameter_l = 0

    diameter = max(diameter_l,diameter_r)
    if diameter_r<diameter_l:
        choose = 'right'
    elif diameter_r>diameter_l:
        choose = 'left'
    elif diameter_r==diameter_l:
        choose = 'equal'

    t2 = time.time_ns()
    duration = np.round((t2-t1)*10**(-9),decimals=3)

    return diameter,duration,diameter_l,diameter_r,choose
