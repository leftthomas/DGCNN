# SRCapsNet
A PyTorch implementation of Convolutional Capsule Network for Super-Resolution based on the paper 
[Convolutional Capsule Network For Single Image Super-resolution](xxx).

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

### Train、Val Datasets
The train and val datasets are from [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/).
Download the original datasets from there, and extract the `DIV2K_train_HR` directory into `data` directory, then 
rename the directory name to `train`, extract the `DIV2K_valid_HR` directory into `data` directory, then 
rename the directory name to `val`. 

### Test Dataset
The test dataset are from 
| **Set 5** |  [Bevilacqua et al. BMVC 2012](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html)
| **Set 14** |  [Zeyde et al. LNCS 2010](https://sites.google.com/site/romanzeyde/research-interests)
| **BSD 100** | [Martin et al. ICCV 2001](https://www.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
| **Sun-Hays 80** | [Sun and Hays ICCP 2012](http://cs.brown.edu/~lbsun/SRproj2012/SR_iccp2012.html)
| **Urban 100** | [Huang et al. CVPR 2015](https://sites.google.com/site/jbhuang0604/publications/struct_sr).
Download the test dataset from [BaiduYun](https://pan.baidu.com/s/1S9w3FAbncE-OTQxnb5MtIg) or 
[GoogleDrive](https://drive.google.com/open?id=1jvls4Z0cj470HMUQcNi5rSC4NdggGqHP), and then extract it into `data` directory.

## Usage

### Train Model
```
python -m visdom.server -logging_level WARNING & python train.py

optional arguments:
--input_size                  training images input size [default value is 32]
--upscale_factor              super resolution upscale factor [default value is 4](choices:[2, 3, 4, 8])
--batch_size                  train batch size [default value is 10]
--num_epochs                  train epoch number [default value is 100]
--train_path                  train image data path [default value is data/train]
--val_path                    val image data path [default value is data/val]
```
Visdom now can be accessed by going to `127.0.0.1:8097/env/$upscale_factor` in your browser, 
`$upscale_factor` means the upscale factor which you are training. If you want to interrupt 
this process, just type `ps aux | grep visdom` to find the `PID`, then `kill PID`.

### Test Benchmark Datasets
```
python benchmark.py

optional arguments:
--upscale_factor              super resolution upscale factor [default value is 4](choices:[2, 3, 4, 8])
--model_name                  super resolution model name [default value is upscale_4.pth]
--test_path                   test image data path [default value is data/test]
```
The output super resolution images are on `results` directory, and statistics on `statistics` directory.

### Test Single Image
```
python single_image.py

optional arguments:
--upscale_factor              super resolution upscale factor [default value is 4](choices:[2, 3, 4, 8])
--image_name                  test low resolution image name
--model_name                  super resolution model name [default value is upscale_4.pth]
```
Put the image on the same directory as `README.md`, the output super resolution image is on the same directory.
