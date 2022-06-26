""" 
11-05-2022 Linde S. Hesse

File containing the preprocessing performed 

Based on DR grading winner Ben Graham: https://www.kaggle.com/competitions/diabetic-retinopathy-detection/discussion/15801
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def scaleRadius(img, scale):
    x = img[int(img.shape[0]/2),:,:].sum(1)
    r = (x>x.mean()/10).sum()/2
    s = scale * 1.0 / r
    return cv2.resize(img, (0,0), fx=s, fy=s)

def preprocess_im(im, savepath):
    """ process image

    Args:
        im (Path): image path
        savepath (Path): path where to save processed images
    """
    scale = 300
    try:
        a = cv2.imread(str(im))
        a = scaleRadius(a, scale)
        a = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0,0), scale / 30), -4, 128)
        b = np.zeros(a.shape)
        cv2.circle(b, (int(a.shape[1]/2), int(a.shape[0]/2)),
        int(scale*0.9), (1,1,1), -1, 8, 0)

        a = a*b + 128 * (1-b)

        cv2.imwrite(str(savepath / im.name), a)


        # crop or pad if necessary
        if a.shape[0] > 270*2:
            middle = int(a.shape[0]/2)
            a = a[middle- 270 : middle+270]
        elif a.shape[0] < 270*2:
            add = 270*2 - a.shape[0]
            a = cv2.copyMakeBorder(a, int(np.floor(add/2)), int(np.ceil(add/2)), 0 ,0 , cv2.BORDER_CONSTANT, 128)
        
        if a.shape[1] > 270*2:
            middle = int(a.shape[1]/2)
            a = a[:, middle-270 : middle+270]
        elif a.shape[1] < 270*2:
            add = 270*2 - a.shape[1]
            a = cv2.copyMakeBorder(a, 0,0 , int(np.floor(add/2)), int(np.ceil(add/2)),  cv2.BORDER_CONSTANT, 128)

        cv2.imwrite(str(savepath / im.name), a)
        
    except Exception as e:
        print(e)
        print(im)

    
if __name__ == '__main__':

    # define data and savepaths
    datapath = '' # add path to kaggle dataset here, folder should contain 'train' and 'test' folders with the respective test and training data
    basepath = Path(datapath)
    savepath = basepath.parents[0] / 'preprocessed_Bgraham'

    # Folder of images
    for traintype in ['train', 'test']:
        total_path = basepath / traintype
        total_savepath = savepath / traintype
        total_savepath.mkdir(parents=True, exist_ok=True)

        # process all images
        ims = list(total_path.glob('*.jpeg'))
        for im in tqdm(ims, savepath = total_savepath):
            preprocess_im(im)