# SAM-UNet

Official implementation of *SAM-UNet: Boosting Medical Image Segmentation by Segment Anything Model*  


## Requirements
This repository is based on PyTorch 1.12.1, CUDA 12.2 and Python 3.8; All experiments in our paper were conducted on two NVIDIA GeForce RTX 4090 24GB GPU.

## Data 

Following previous works, we have validated our method on two benchmark datasets, including Synapse multi-organ segmentation dataset and Automated Cardiac Diagnosis Challenge dataset.  
It should be noted that we do not have permissions to redistribute the data. Thus, for those who are interested, please follow the instructions below and process the data, or you will get a mismatching result compared with ours.

### Data Preparation

#### Download

Synapse multi-organ segmentation dataset: https://www.synapse.org/#!Synapse:syn3193805/wiki/  
Automated Cardiac Diagnosis Challenge (ACDC) dataset: https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html

#### Data Split

We split the data following previous works. Detailed split could be found in folder `lists`, which are stored in .list files.

#### Data Preprocessing

Download the data from the url above, then run the script `./preprocess/preprocess_Synapse_data.py` and `./preprocess/acdc_data_processing.py` by passing the arguments of data location.

### Quick Download Data
1. The Synapse datasets we used are provided by [SAMed](https://github.com/hitachinsk/SAMed)'s authors.

Training Dataset: https://drive.google.com/file/d/1zuOQRyfo0QYgjcU_uZs0X3LdCnAC2m3G/view?usp=share_link

Testing Dataset: https://drive.google.com/file/d/1RczbNSB37OzPseKJZ1tDxa5OO1IIICzK/view

2. The ACDC datasets we used are provided by [SSL4MIS](https://github.com/HiLab-git/SSL4MIS)'s authors.

ACDC Dataset: https://github.com/HiLab-git/SSL4MIS/tree/master/data/ACDC

### Prepare Your Own Data

Our DCPA could be extended to other datasets with some modifications.  

## Usage
1. Clone the repo;
```
git clone https://github.com/BinYCn/SAMUNet.git
cd SAMUNet
```

2. Put the data in `./data`;

3. Please download the pretrained [SAM Encoder Model](https://pan.baidu.com/s/1b3TwZSPLRlRNU4oCwrbUFQ?pwd=rydg)(provided by the original repository of [SAM](https://segment-anything.com/)). Put it in the `./checkpoints` folder;

4. Test the model use pretrained_weight;
```
bash test_pretrained.sh
```

5. Train and test the model;
```
bash train.sh
bash test.sh
```


## Acknowledgements:
Our code is adapted from [TransUNet](https://github.com/Beckschen/TransUNet), [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet), [SSL4MIS](https://github.com/HiLab-git/SSL4MIS) and [SAMed](https://github.com/hitachinsk/SAMed). Thanks for these authors for their valuable works and hope our model can promote the relevant research as well.
