# Multi-spectral Vehicle Re-identification: A Challenge
## Dataset
In this work, we address the RGB and IR vehicle Re-ID problem and contribute a multi-spectral vehicle Re-ID benchmark named RGBN300, including RGB and NIR (Near Infrared) vehicle images of 300 identities from 8 camera views, giving in total 50125 RGB images and 50125 NIR images respectively. In addition, we have acquired additional TIR (Thermal Infrared) data for 100 vehicles from RGBN300 to form another dataset for three-spectral vehicle Re-ID. 
![RGB-NIR-TIR](fig/challenge.jpg)

RGBN300
link：https://pan.baidu.com/s/1uiKcqiqdhd13nLSW8TUASg 
Extraction code：11y8 

RGBNT100
link：https://pan.baidu.com/s/1xqqh7N4Lctm3RcUdskG0Ug 
Extraction code：rjin


## HAMNet
### Pipeline
![RGB-NIR-TIR](fig/frame.jpg)

        @InProceedings{Li_2020_AAAI,
        author = {Hongchao Li, Chenglong Li, Xianpeng Zhu, Aihua Zheng and Bin Luo},
        title = {Multi-spectral Vehicle Re-identification: A Challenge},
        booktitle = {AAAI},
        month = {February},
        year = {2020}
        }
### Results(Rank1(mAP))
|Modality|RGBN300|RGBNT100|
|:---|:---|:---|
|RGB_onestream|72.6(49.5)|58.5(41.0)|
|NIR_onestream|61.9(42.1)|52.8(37.1)|
|TIR_onestream|-|61.8(35.7)|
|RGB-NIR_multistream|77.2(56.9)|65.4(43.1)|
|RGB-NIR-TIR_multistream|-|82.6(60.5)|
|**RGB-NIR_HAMNet**|**84.0(61.9)**|-|
|**RGB-NIR-TIR_HAMNet**|-|**84.7(64.1)**|

### Get Started

The designed architecture follows this guide PyTorch-Project-Template, you can check each folder's purpose by yourself. The codes are expanded on a [ReID-baseline](https://github.com/L1aoXingyu/reid_baseline). 

1.`cd` to folder where you want to download this repo

2.Run `git clone https://github.com/ttaalle/multi-modal-vehicle-Re-ID.git`

3.Install dependencies:
* pytorch>=0.4
* torchvision
* ignite=0.1.2
* yacs 


4.Prepare Pretraining model on Imagenet

for example /home/——/.torch/models/resnet50-19c8e357.pth

5.Prepare dataset

Create a directory to store reid datasets under this repo or outside this repo. Remember to set your path to the root of the dataset in config/defaults.py for all training and testing or set in every single config file in configs/ or set in every single command.

You can create a directory to store reid datasets under this repo via

        cd multi-modal-vehicle-Re-ID
        mkdir data
        
(1) RGBN300 dataset

Download dataset and only use `rgbir` to data/ 

The data structure would like:

                data
                    rgbir # this folder contains 3 files.
                        bounding_box_test/
                        bounding_box_train/
                        query

(2) RGBNT100 dataset

Download dataset and only use `rgbir` to data/ (It is worth noting that the two datasets use the same read interface in our algorithm, so in order to prevent data from being polluted, we should only change the data folder to `rgbir` when running the code.)

The data structure would like:

                data
                    rgbir # this folder contains 3 files.
                        bounding_box_test/
                        bounding_box_train/
                        query


### train
        
        python3 train.py --config_file='softmax_triplet.yml' MODEL.DEVICE_ID "('your device id')" DATASETS.NAMES "('rgbir')" OUTPUT_DIR "('your path to save checkpoints and logs')"
        
### test
        python3 test.py --config_file='softmax_tripletr.yml' MODEL.DEVICE_ID "('your device id')" DATASETS.NAMES "('rgbir')" MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('your path to trained checkpoints')"
        
        
        
To propose a stronger baseline, this version has been added bag of tricks(Random erasing augmentation, Label smoothing and BNNeck) as [Strong ReID-baseline](https://github.com/michuanhaohao/reid-strong-baseline).

