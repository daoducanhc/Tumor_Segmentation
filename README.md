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
    <a href="https://github.com/daoducanhc/Tumor_Segmentation#dart-results">Results</a>
    ·
    <a href="https://github.com/daoducanhc/Tumor_Segmentation#clapper-demo">Demo</a>
    ·
  </p>
</p>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[![Python](https://img.shields.io/badge/Python-v3.8.3-blue.svg?logo=python)](https://www.python.org/downloads/release/python-383/)
[![PyTorch](https://img.shields.io/badge/PyTorch-v1.7.0-critical.svg?logo=pytorch)](https://pytorch.org/get-started/previous-versions/#v170)
[![cuda](https://img.shields.io/badge/CUDA-v11.0.221-success.svg?logo=nvidia)](https://developer.nvidia.com/cuda-11.0-download-archive)
[![matplotlib](https://img.shields.io/badge/Matplotlib-v3.3.3-9cf.svg?logo=matplotlib)](https://matplotlib.org/3.3.3/contents.html)
[![license](https://img.shields.io/badge/License-MIT-lightgrey.svg?logo=license)](https://github.com/daoducanhc/Tumor_Segmentation#balance_scale-license)


## :brain: About The Project
The Brain Tumor Segmentation project utilize Deep Learning to help doctors in the segmentation process. 

System built with PyTorch in order to examine two model architectures: **UNet** and **ResUNet**.

## :hammer_and_wrench: Why we need to consider these 2 models?

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

## :books: Data

Dataset is stored in the binary data container format that the MATLAB program uses (.mat file) [[link]](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)

It's contains 3064 brain MRI images and coordinates of tumor for each image (Data is labeled). Each image has dimension ```512 x 512 x 1```.

## :chart_with_upwards_trend: Training Process

In both 2 charts, you can see sometime my valid loss is lower than train loss. Don't jump to the conclusion that it's wrong.
<br />
<br />
There a 3 main reasons for that:

  - Regularization applied during training, but not during validation/testing. [[Source]](https://twitter.com/aureliengeron/status/1110839345609465856?s=20)

  - Training loss is measured _during_ each epoch while validation loss is measured _after_ each epoch. [[Source]](https://twitter.com/aureliengeron/status/1110839480024338432?s=20)

  - The validation set may be easier than the training set. [[Source]](https://twitter.com/aureliengeron/status/1110839534013472769?s=20)

### UNet

Learning rate: 0.001 (reduce 70% after each 30 epochs)

Total time: 2 hours 28 minutes

![Loss Graph](demo/loss_UNet.png)
[Detail here](https://github.com/daoducanhc/Tumor_Segmentation/blob/master/outputs/historyUNet)



### ResUNet

Learning rate: 0.001 (reduce 70% after each 15 epochs)

Total time: 58 minutes

![Loss Graph](demo/loss_ResUNet.png)
[Detail here](https://github.com/daoducanhc/Tumor_Segmentation/blob/master/outputs/historyResUNet)


## :dart: Results
 .                 |      UNet    |     ResUNet 
:---------------:|:------------:|:----------------:
Training loss     |   0.0171 |     **0.0161***
Validation loss   |   0.0174 |     **0.0170***
Mean Dice score  |   0.73       |       **0.76***
Number of epochs  |    100       |       **35***

Hyperparameters tuning are almost the same (difference in learning rate scheduler). Hence, we can see how remarkably effective ResBlock is.

Achieves dice score of **0.76** only in **35** epochs. The training time reduce by more than 2 times ('58 minutes' to '2 hours 28 minutes') while train with origin UNet.



## :clapper: Demo

Here are some top score results that we evaluate ResUNet model with testing dataset.

.             |      .
:-------------------------:|:-------------------------:
![](demo/14.jpg)  |  ![](demo/1.jpg)
![](demo/2.jpg)  |  ![](demo/3.jpg)
![](demo/4.jpg)  |  ![](demo/5.jpg)
![](demo/6.jpg)  |  ![](demo/7.jpg)
![](demo/8.jpg)  |  ![](demo/9.jpg)
![](demo/10.jpg)  |  ![](demo/11.jpg)
![](demo/12.jpg)  |  ![](demo/13.jpg)

## :balance_scale: License
Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
