from parfor import pmap
import skimage
import av
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from multi_pupil_detection import load_video, multi_pupil_detection

video_path = 'eyes.mp4'
frames = load_video(video_path)
diameter_list = pmap(multi_pupil_detection,frames)
plt.plot(diameter_list)