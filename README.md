# DGCNN
A PyTorch implementation of DGCNN based on AAAI 2018 paper 
[An End-to-End Deep Learning Architecture for Graph Classification](https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf).

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)
```
conda install pytorch torchvision -c pytorch
```
- PyTorchNet
```
pip install git+https://github.com/pytorch/tnt.git@master
```
- [PyTorch Geometric](https://rusty1s.github.io/pytorch_geometric/build/html/index.html)
```
pip install torch-scatter
pip install torch-sparse
pip install torch-cluster
pip install torch-spline-conv (optional)
pip install torch-geometric
```

## Datasets
The datasets are collected from [graph kernel datasets](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets).
The code will download and extract them into `data` directory automatically. The `10fold_idx` files are collected from 
[pytorch_DGCNN](https://github.com/muhanzhang/pytorch_DGCNN).

## Usage
### Train Model
```
python -m visdom.server -logging_level WARNING & python train.py --data_type PTC_MR --num_epochs 200
optional arguments:
--data_type                   dataset type [default value is 'DD'](choices:['DD', 'PTC_MR', 'NCI1', 'PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI', 'MUTAG', 'COLLAB'])
--batch_size                  train batch size [default value is 50]
--num_epochs                  train epochs number [default value is 100]
```
Visdom now can be accessed by going to `127.0.0.1:8097/env/$data_type` in your browser, `$data_type` means the dataset type which you are training.

## Benchmarks
Default PyTorch Adam optimizer hyper-parameters were used without learning rate scheduling. 
The model was trained with 100 epochs and batch size of 50 on a NVIDIA GTX 1070 GPU. 

Here are tiny differences between this code and official paper. Firstly, **X** is defined as a concatenated matrix of vertex labels、
vertex attributes and normalized node degrees. Secondly, **CrossEntropyLoss** is used to compute the loss. 

<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>MUTAG</th>
      <th>PTC</th>
      <th>NCI1</th>
      <th>PROTEINS</th>
      <th>D&D</th>
      <th>COLLAB</th>
      <th>IMDB-B</th>
      <th>IMDB-M</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">Num. of Graphs</td>
      <td align="center">188</td>
      <td align="center">344</td>
      <td align="center">4,110</td>
      <td align="center">1,113</td>
      <td align="center">1,178</td>
      <td align="center">5,000</td>
      <td align="center">1,000</td>
      <td align="center">1,500</td>
    </tr>
    <tr>
      <td align="center">Num. of Classes</td>
      <td align="center">2</td>
      <td align="center">2</td>
      <td align="center">2</td>
      <td align="center">2</td>
      <td align="center">2</td>
      <td align="center">3</td>
      <td align="center">2</td>
      <td align="center">3</td>
    </tr>
    <tr>
      <td align="center">Node Attr. (Dim.)</td>
      <td align="center">8</td>
      <td align="center">19</td>
      <td align="center">38</td>
      <td align="center">5</td>
      <td align="center">90</td>
      <td align="center">1</td>
      <td align="center">1</td>
      <td align="center">1</td>
    </tr>
    <tr>
      <td align="center">Num. of Parameters</td>
      <td align="center">52,035</td>
      <td align="center">52,387</td>
      <td align="center">52,995</td>
      <td align="center">51,939</td>
      <td align="center">54,659</td>
      <td align="center">51,940</td>
      <td align="center">51,811</td>
      <td align="center">51,940</td>
    </tr>
    <tr>
      <td align="center">DGCNN (official)</td>
      <td align="center"><b>85.83±1.66</b></td>
      <td align="center"><b>58.59±2.47</b></td>
      <td align="center"><b>74.44±0.47</b></td>
      <td align="center"><b>75.54±0.94</b></td>
      <td align="center"><b>79.37±0.94</b></td>
      <td align="center"><b>73.76±0.49</b></td>
      <td align="center"><b>70.03±0.86</b></td>
      <td align="center"><b>47.83±0.85</b></td>
    </tr>
    <tr>
      <td align="center">DGCNN (ours)</td>
      <td align="center">82.22±10.18</td>
      <td align="center">62.94±8.24</td>
      <td align="center">74.40±2.07</td>
      <td align="center">72.52±3.72</td>
      <td align="center">69.83±4.08</td>
      <td align="center">68.80±2.33</td>
      <td align="center">69.90±4.83</td>
      <td align="center">46.33±3.71</td>
    </tr>
    <tr>
      <td align="center">Training Time</td>
      <td align="center">4.51s</td>
      <td align="center">6.88s</td>
      <td align="center">61.63s</td>
      <td align="center">21.20s</td>
      <td align="center">64.98s</td>
      <td align="center">203.35s</td>
      <td align="center">15.59s</td>
      <td align="center">22.21s</td>
    </tr> 
  </tbody>
</table>

## Results
The train loss、accuracy, test loss、accuracy are showed on visdom.

### MUTAG
![result](results/mutag.png)
### PTC
![result](results/ptc.png)
### NCI1
![result](results/nci1.png)
### PROTEINS
![result](results/proteins.png)
### D&D
![result](results/dd.png)
### COLLAB
![result](results/collab.png)
### IMDB-B
![result](results/imdbb.png)
### IMDB-M
![result](results/imdbm.png)