# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re
import pdb
import os.path as osp
import numpy as np
import random

from .bases import BaseImageDataset
from collections import Counter


class Vehicleid(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'vehicleid'

    def __init__(self, root='/home/haoluo/data', verbose=True, **kwargs):
        super(Vehicleid, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.all_dir = osp.join(self.dataset_dir, 'image')
        self.file_dir = osp.join(self.dataset_dir, 'train_test_split')
        
        file = open(osp.join(self.file_dir, 'train_list.txt'))
        data_mat = []
        label_mat = []
        for line in file.readlines():
            cur_line = line.strip().split(" ")
            #float_line = map(float,cur_line)
            data_mat.append(cur_line[0:1])
            label_mat.append(cur_line[-1])
        #print(data_mat)
        #print(label_mat)
        #print(np.shape(data_mat))
        #pdb.set_trace()
        #d=np.reshape(data_mat,(113346,1))
        img_paths = glob.glob(osp.join(self.all_dir, '*.jpg'))
        #pdb.set_trace()
        
        train = []
        pid_container = set()
        for i in range(len(label_mat)):
            
            str1 = '%s/%s.jpg'%(self.all_dir,data_mat[i][0])
           # train_paths.append(img_paths[img_paths.index(str1)]) 
            pid = int(label_mat[i])
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        #pdb.set_trace()
        for i in range(len(label_mat)):
            str1 = '%s/%s.jpg'%(self.all_dir,data_mat[i][0])
            pid = int(label_mat[i])
            pid = pid2label[pid]
            camid  = i     
            train.append((str1,pid,camid))
        #pdb.set_trace()
            #pdb.set_trace()
       # pdb.set_trace()
        
        file = open(osp.join(self.file_dir, 'test_list_800.txt'))
        data_mat = []
        label_mat = []
        for line in file.readlines():
            cur_line = line.strip().split(" ")
            #float_line = map(float,cur_line)
            data_mat.append(cur_line[0:1])
            label_mat.append(cur_line[-1])
        #print(data_mat)
        #print(label_mat)
        #print(np.shape(data_mat))
        #pdb.set_trace()
        #d=np.reshape(data_mat,(113346,1))
        img_paths = glob.glob(osp.join(self.all_dir, '*.jpg'))
        #pdb.set_trace()
        
        query = []
        gallery = []
        pid_container = set()
        pid_list = []
       
        for i in range(len(label_mat)):
            
            str1 = '%s/%s.jpg'%(self.all_dir,data_mat[i][0])
           # train_paths.append(img_paths[img_paths.index(str1)])           
           # img_paths[i] = str1
            pid = int(label_mat[i])
            pid_container.add(pid)
            pid_list.append(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        pid_dict = dict(Counter(pid_list))
        #pdb.set_trace()
        for i in range(len(label_mat)):
            str1 = '%s/%s.jpg'%(self.all_dir,data_mat[i][0])
            pid = int(label_mat[i])
            pid = pid2label[pid]
            camid  = i
            query.append((str1,pid,camid))
        #pdb.set_trace()     
        num = 0
        for i in range(0,800):
            gallery.append(query[num])
            num = pid_dict[pid_list[num]] + num
            #pdb.set_trace()
        ##pdb.set_trace()       

        
        #self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        #self.query_dir = osp.join(self.dataset_dir, 'query')
        #self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

      #  self._check_before_run()

       # train = self._process_dir(self.train_dir, relabel=True)
       # query = self._process_dir(self.query_dir, relabel=False)
       # gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Vehicleid loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            #pdb.set_trace()
            #if pid == -1: continue  # junk images are just ignored
            assert 1 <= pid <= 776  # pid == 0 means background
            assert 1 <= camid <= 20
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))
        return dataset
