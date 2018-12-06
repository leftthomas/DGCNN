# RRCapsNet
A PyTorch implementation of Convolutional Capsule Network based on the paper [Convolutional Capsule Network For Single Image Reflection Removal]().

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- PyTorch
```
conda install pytorch torchvision -c pytorch
```
- PyTorchNet
```
pip install git+https://github.com/pytorch/tnt.git@master
```
- CapsuleLayer
```
pip install git+https://github.com/leftthomas/CapsuleLayer.git@master
```

## Datasets

The datasets are collected from [perceptual-reflection-removal](https://github.com/ceciliavision/perceptual-reflection-removal)
and [CEILNet](https://github.com/fqnchina/CEILNet).
Download the datasets from [BaiduYun](https://pan.baidu.com/s/1PJuEvmFdpuJIZwtNU6NgtQ) 
or [GoogleDrive](https://drive.google.com/open?id=1abYah24PZKQS8K9G3Xsd_6a8Raptp30a), and extract them into `data` directory.

## Usage

### Train Model
```
python -m visdom.server -logging_level WARNING & python train.py

optional arguments:
--crop_size                   image crop size [default value is 224]
--batch_size                  train batch size [default value is 8]
--num_epochs                  train epoch number [default value is 100]
--train_path                  train image data path [default value is 'data/train']
--test_path                   test image data path [default value is 'data/test']
```
Visdom now can be accessed by going to `127.0.0.1:8097/` in your browser. If you want to interrupt 
this process, just type `ps aux | grep visdom` to find the `PID`, then `kill PID`.

### Test Single Image
```
python vis.py

optional arguments:
--blended_name                test blended image name
--transmission_name           test transmission image name [default value is '']
--crop_size                   image crop size [default value is 224]
--model_name                  reflection removal model name [default value is 'model.pth']
--test_mode                   using GPU or CPU [default value is 'GPU'](choices:['GPU', 'CPU'])
```
The output reflection removed image is on `results` directory.
