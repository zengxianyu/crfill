# crfill

[Usage](#basic-usage) | [Web App](#web-app) | | [Paper](https://arxiv.org/pdf/2011.12836.pdf) | [Supplementary Material](https://maildluteducn-my.sharepoint.com/:b:/g/personal/zengyu_mail_dlut_edu_cn/Eda8Q_v7OSNMj0nr2iG7TmABvxLOtAPwVDdk5mjl7c-IFw?e=Cvki0I) | [More results](viscmp.md) |

code for paper ``Image Inpainting with Contextual Reconstruction Loss". This repo (including code and models) are for research purposes only. 

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
or manually install these packages in a Python 3.6 enviroment: 

```pytorch=1.3.1, opencv=3.4.2, tqdm, torchvision, dill, matplotlib, opencv```


### Inference

```
./test.sh
```

These script will run the inpainting model on the samples I provided. Modify the options in ```test.sh``` to test on custom data

### Train
1. Prepare training datasets and put them in ```./datasets/``` following the example ```./datasets/places```

2. run the training script:
```
./train.sh
```

you may modify the training script to use different settings, e.g., batch size, hyperparameters


### Web APP
<img src="https://s3.ax1x.com/2020/11/27/DrVLs1.png" width=300>

To use the web app, these additional packages are required: 

```flask```, ```requests```, ```pillow```


```
./demo.sh
```

then open http://localhost:2334 in the browser
