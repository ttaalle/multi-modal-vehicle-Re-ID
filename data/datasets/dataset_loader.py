# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import pdb

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            #img1 = img
            img1 = img.crop((0, 0, 256, 128))
            img2 = img.crop((256, 0, 512, 128))
            img3 = img.crop((512, 0, 768, 128))
            #print(img1)
            #print(img2)
            #print(img3)
            #pdb.set_trace()
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img1, img2, img3


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        #pdb.set_trace()
        img1, img2, img3 = read_image(img_path)
        #print(img1)
        #T.functional.crop(img,0,0,640,360)
        #print(img)
        #pdb.set_trace()
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        #print(img1)
        #pdb.set_trace()
        return img1, img2, img3, pid, camid, img_path
