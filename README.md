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
The code will download and extract them into `data` directory automatically.

## Usage
### Train Model
```
python -m visdom.server -logging_level WARNING & python train.py --data_type PTC_MR --num_epochs 200
optional arguments:
--data_type                   dataset type [default value is 'DD'](choices:['DD', 'PTC_MR', 'NCI1', 'PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI', 'MUTAG', 'COLLAB'])
--batch_size                  train batch size [default value is 20]
--num_epochs                  train epochs number [default value is 100]
```
Visdom now can be accessed by going to `127.0.0.1:8097/env/$data_type` in your browser, `$data_type` means the dataset type which you are training.

## Benchmarks
Default PyTorch Adam optimizer hyper-parameters were used without learning rate scheduling. 
The model was trained with 100 epochs and batch size of 20 on a NVIDIA GTX 1070 GPU. 

Here is tiny difference between this code and office paper. **X** is defined as a concatenated matrix of vertex labels、
vertex attributes and normalized node degrees.

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
      <td align="center">8</td>
      <td align="center">8</td>
      <td align="center">8</td>
      <td align="center">8</td>
      <td align="center">8</td>
      <td align="center">8</td>
    </tr>
    <tr>
      <td align="center">Num. of Parameters</td>
      <td align="center">6,851</td>
      <td align="center">7,203</td>
      <td align="center">6851</td>
      <td align="center">6851</td>
      <td align="center">6851</td>
      <td align="center">6851</td>
      <td align="center">6851</td>
      <td align="center">6851</td>
    </tr>
    <tr>
      <td align="center">DGCNN (offical)</td>
      <td align="center"><b>85.83±1.66</b></td>
      <td align="center"><b>85.83±1.66</b></td>
      <td align="center"><b>85.83±1.66</b></td>
      <td align="center"><b>85.83±1.66</b></td>
      <td align="center"><b>85.83±1.66</b></td>
      <td align="center"><b>85.83±1.66</b></td>
      <td align="center"><b>85.83±1.66</b></td>
      <td align="center"><b>85.83±1.66</b></td>
    </tr>
    <tr>
      <td align="center">DGCNN (ours)</td>
      <td align="center">83.41±8.47</td>
      <td align="center">83.41±8.47</td>
      <td align="center">83.41±8.47</td>
      <td align="center">83.41±8.47</td>
      <td align="center">83.41±8.47</td>
      <td align="center">83.41±8.47</td>
      <td align="center">83.41±8.47</td>
      <td align="center">83.41±8.47</td>
    </tr> 
  </tbody>
</table>