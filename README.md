<br />
<p align="center">
  <a href="https://github.com/daoducanhc/Tumor_Segmentation">
    <img src="demo/logo.jpg" alt="Logo" width="200" height="200">
  </a>

  <h1 align="center">Brain Tumor Segmentation</h1>
  
  <p align="center">
    <br />
    <a href="https://github.com/daoducanhc/Tumor_Segmentation/issues">Report Bug</a>
    ·
    <a href="https://github.com/daoducanhc/Tumor_Segmentation/issues">Request Feature</a>
    .
    <a href="https://github.com/daoducanhc/Tumor_Segmentation#results">Results</a>
    ·
    <a href="https://github.com/daoducanhc/Tumor_Segmentation#demo">Demo</a>
    ·
  </p>
</p>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[![Python](https://img.shields.io/badge/Python-v3.8.3-blue.svg?logo=python)](https://www.python.org/downloads/release/python-383/)
[![PyTorch](https://img.shields.io/badge/PyTorch-v1.7.0-critical.svg?logo=pytorch)](https://pytorch.org/get-started/previous-versions/#v170)
[![cuda](https://img.shields.io/badge/CUDA-v11.0.221-success.svg?logo=nvidia)](https://developer.nvidia.com/cuda-11.0-download-archive)
[![matplotlib](https://img.shields.io/badge/Matplotlib-v3.3.3-9cf.svg?logo=matplotlib)](https://matplotlib.org/3.3.3/contents.html)
[![license](https://img.shields.io/badge/License-MIT-lightgrey.svg?logo=license)](https://github.com/daoducanhc/Tumor_Segmentation/blob/master/LICENSE)


## About The Project
The Brain Tumor Segmentation project utilize Deep Learning to help doctors in the segmentation process. 

System built with PyTorch in order to examine two model architectures: **UNet** and **ResUNet**.

## Why we need to consider these 2 models?

### UNet

"U-Net is a convolutional neural network that was developed for biomedical image segmentation." (Wikipedia)

![image](https://user-images.githubusercontent.com/59494615/111856075-eb240380-895a-11eb-88e7-48cd1a2dd890.png)

Owned a unique U-shaped, U-Net consists of a contracting path (encoder) to capture context and a symmetric expanding path (decoder) that enables exact localization. 

Was created for specific task, U-Net can yield more precise segmentation despite fewer trainer samples.

### ResUNet

Replacing encoder path of original U-Net architecture by state-of-the-art model: ResNet. 

We do not apply ResBlock in both the encoder and decoder part of U-Net because it may create a 'too complex' model that our data will be overfit so quickly.

![image](https://github.com/daoducanhc/Tumor_Segmentation/blob/master/demo/ResUNet.PNG)

Image source: [[paper]](https://github.com/daoducanhc/Tumor_Segmentation/blob/master/demo/reference.pdf)
## Training Process
### UNet
![Loss Graph](demo/loss_UNet.png)

### ResUNet
![Loss Graph](demo/loss_ResUNet.png)

## Results
 .                 |      UNet    |     ResUNet 
:---------------:|:------------:|:----------------:
Mean Dice score  |   0.73       |       **0.76**
Number of epochs  |    100       |       **35**
Histoty           | [Detail here](https://github.com/daoducanhc/Tumor_Segmentation/blob/master/outputs/historyUNet) | [Detail here](https://github.com/daoducanhc/Tumor_Segmentation/blob/master/outputs/historyResUNet)


## Demo
.             |      .
:-------------------------:|:-------------------------:
![](demo/14.jpg)  |  ![](demo/1.jpg)
![](demo/2.jpg)  |  ![](demo/3.jpg)
![](demo/4.jpg)  |  ![](demo/5.jpg)
![](demo/6.jpg)  |  ![](demo/7.jpg)
![](demo/8.jpg)  |  ![](demo/9.jpg)
![](demo/10.jpg)  |  ![](demo/11.jpg)
![](demo/12.jpg)  |  ![](demo/13.jpg)
