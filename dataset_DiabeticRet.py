""" 
11-05-2022 Linde S. Hesse

File containing the dataset for the diabetic retinopathy challenge
"""

from torch.utils.data import Dataset
from pathlib import PosixPath
import cv2
import torch
from tqdm import tqdm

class DiabeticRet(Dataset):
    def __init__(self, datapath, labels, preload=True, datapart=1.0, transform = None):
        """ Load in the diabetic Res dataset
        Args:
            datapath (Posixpath / List(Posixpath)): Path to folders or list of image paths
            csv_path (str):name of csv
            preload (bool, optional): [description]. Defaults to True.
            datapart (float, optional): [description]. Defaults to 1.0.
        """
        self.datapath = datapath
        self.preload = preload

        
        if type(datapath) == PosixPath:
            self.image_paths = list(datapath.glob('**/*.jpeg'))
        else:
            self.image_paths = datapath
       
        if preload:
            self.preload_ims()
            
        self.transform = transform  

        # get the labels from the images given
        if labels is not None:
            self.labels = labels
        else:
            self.labels = len(self.image_paths) * [0]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        im_path = self.image_paths[idx]

        # Based on preloading get image
        if self.preload:
            im = self.ims[idx]
        else:
            im = self.read_im(im_path)[0]

        # Get label
        label = float(self.labels[idx])

        if self.transform is not None:
            im = self.transform(im)

        return im, label, str(im_path.name)

    def preload_ims(self):
        """ Preload the images in memory
        """
        self.ims = torch.zeros(len(self.image_paths), 3, 540, 540)

        for i, pathx in enumerate(tqdm(self.image_paths)):
            im = self.read_im(pathx)
            self.ims[i] = im[0]

    def read_im(self, pathx):
        """ Read a single image from the path

        Args:
            pathx ([type]): [description]

        Returns:
            [type]: [description]
        """
        jpeg_im = cv2.imread(str(pathx))
        norm = jpeg_im/255

        im = torch.zeros(1, 3, 540,540)
        im[0] = torch.from_numpy(norm).permute([2,1,0])
        return im

