# SRCapsNet
A PyTorch implementation of Super-Resolution Capsule Network based on the paper [xxx](xxx).

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
The train and val datasets are from [ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/).
Train dataset has 1,282,167 images and Val dataset has 50,000 images.
Download the train dataset from [here](http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar) 
and val dataset from [here](http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar), 
then extract them into `data` directory.

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
python train.py

optional arguments:
--crop_size                   training images crop size [default value is 240]
--upscale_factor              super resolution upscale factor [default value is 4](choices:[2, 3, 4, 8])
--batch_size                  train batch size [default value is 32]
--num_epochs                  train epoch number [default value is 100]
--train_path                  train image data path [default value is data/train]
--val_path                    val image data path [default value is data/val]
```

### Test Benchmark Datasets
```
python test_benchmark.py

optional arguments:
--upscale_factor              super resolution upscale factor [default value is 4](choices:[2, 3, 4, 8])
--model_name                  super resolution model name [default value is upscale_4.pth]
--test_path                   test image data path [default value is data/test]
```
The output super resolution images are on `results` directory, and statistics on `statistics` directory.

### Test Single Image
```
python test_image.py

optional arguments:
--upscale_factor              super resolution upscale factor [default value is 4](choices:[2, 3, 4, 8])
--image_name                  test low resolution image name
--model_name                  super resolution model name [default value is upscale_4.pth]
```
Put the image on the same directory as `README.md`, the output super resolution image is on the same directory.
