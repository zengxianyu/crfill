# crfill

[Usage](#basic-usage) | [Web App](#web-app) | | [Paper](https://arxiv.org/pdf/2011.12836.pdf) | [Supplementary Material](https://maildluteducn-my.sharepoint.com/:b:/g/personal/zengyu_mail_dlut_edu_cn/Eda8Q_v7OSNMj0nr2iG7TmABvxLOtAPwVDdk5mjl7c-IFw?e=Cvki0I) | [More results](viscmp.md) |

code for paper ``CR-Fill: Generative Image Inpainting with Auxiliary Contextual Reconstruction". This repo (including code and models) are for research purposes only. 

<img src="https://s3.ax1x.com/2020/11/27/DrVxIO.png" width="160"> <img src="https://s3.ax1x.com/2020/11/27/DrZ9RH.png" width="160"> 
<img src="https://s3.ax1x.com/2020/11/27/DrZlyn.png" width="160"> <img src="https://s3.ax1x.com/2020/11/27/DrZGwV.png" width="160"> 

<img src="https://s3.ax1x.com/2020/11/27/DrZtFU.png" width="360"> <img src="https://s3.ax1x.com/2020/11/27/DrZdSJ.png" width="360"> 

## Usage

### Dependencies
0. Download code
```
git clone --single-branch https://github.com/zengxianyu/crfill
git submodule init
git submodule update
```

0. Download data and model
```
chmod +x download/*
./download/download_model.sh
./download/download_datal.sh
```

1. Install dependencies:
```
conda env create -f environment.yml
```
or install these packages manually in a Python 3.6 enviroment: 

```pytorch=1.3.1, opencv=3.4.2, tqdm, torchvision, dill, matplotlib, opencv```


### Inference

```
./test.sh
```

These script will run the inpainting model on the samples I provided. Modify the options ```--image_dir, --mask_dir, --output_dir``` in ```test.sh``` to test on custom data. 

### Train
1. Prepare training datasets and put them in ```./datasets/``` following the example ```./datasets/places```

2. run the training script:
```
./train.sh
```

open the html files in ```./output``` to visualize training

After the training is finished, the model files can be found in ```./checkpoints/debugarr0```

you may modify the training script to use different settings, e.g., batch size, hyperparameters

### Finetune
For finetune on custom dataset based on my pretrained models, use the following command:
1. download checkpoints
```
./download/download_pretrain.sh
```
2. run the training script
```
./finetune.sh
```
you may change the options in ```finetune.sh``` to use different hyperparameters or your own dataset


### Web APP
<img src="https://s3.ax1x.com/2020/11/27/DrVLs1.png" width=300>

To use the web app, these additional packages are required: 

```flask```, ```requests```, ```pillow```


```
./demo.sh
```

then open http://localhost:2334 in the browser to use the web app

## Citing
```
@inproceedings{zeng2021generative,
  title={CR-Fill: Generative Image Inpainting with Auxiliary Contextual Reconstruction},
  author={Zeng, Yu and Lin, Zhe and Lu, Huchuan and Patel, Vishal M.},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2021}
}
```

## Acknowledgement

* DeepFill https://github.com/jiahuiyu/generative_inpainting
* Pix2PixHD https://github.com/NVIDIA/pix2pixHD
* SPADE https://github.com/NVlabs/SPADE
