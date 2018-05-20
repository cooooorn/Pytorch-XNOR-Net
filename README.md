# Pytorch-XNOR-Net


# MNIST

## Usage
### Train:
~~~shell
cd <Repository Root>/MNIST/
python3 main.py --arch Bin_LeNet
python3 main.py --arch LeNet
~~~
### Evaluate:
~~~shell
cd <Repository Root>/MNIST/
python3 main.py --arch Bin_LeNet --evaluate --pretrained ./models/Bin_LeNet.best.pth # --no_cuda (Use CPU)
python3 main.py --arch LeNet --evaluate --pretrained ./models/LeNet.best.pth # --no_cuda (Use CPU)
~~~
## Result
|  Network  | Accuracy |  Size   |
|  -------  | -------- |  ----   |
|   LeNet   |  99.50%  |  1.7 MB |
| Bin_LeNet |  99.45%  |  102 KB |


# Cifar10

## Usage
### Train:
~~~shell
cd <Repository Root>/Cifar10/
python3 main.py --arch Bin_VGG16 #(11, 13, 16, 19)
python3 main.py --arch VGG16 #(11, 13, 16, 19)
~~~
### Evaluate:
~~~shell
cd <Repository Root>/Cifar10/
python3 main.py --arch Bin_VGG16 --evaluate --pretrained ./models/Bin_VGG16.best.pth # --no_cuda (Use CPU)
python3 main.py --arch VGG16 --evaluate --pretrained ./models/VGG16.best.pth # --no_cuda (Use CPU)
~~~
## Result
|  Network  | Accuracy |  Size    |
|  -------  | -------- |  ----    |
|   VGG13   |  92.40%  |  37.7 MB |
| Bin_VGG13 |  88.74%  |  1.3  MB |
|   VGG16   |  92.29%  |  59.0 MB |
| Bin_VGG16 |  87.78%  |  2.0  MB |

# Pre-trained models
[Google Drive](https://drive.google.com/open?id=13KAF89w1-OnGTgHlhblnzBafpz-sTCVT)

# Environment
## Software
* Ubuntu  16.04
* Python  3.5
* Pytorch 0.3.1
* CUDA    8.0
* gcc     5.4

## Hardware

* NVIDIA GTX 1080
* Intel  i5-6500 CPU @ 3.20GHz × 4


# Reference
* [Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](https://arxiv.org/pdf/1602.02830.pdf)
* [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](https://arxiv.org/pdf/1603.05279.pdf)
* https://github.com/jiecaoyu/XNOR-Net-PyTorch
* [cpu-gemm](http://apfel.mathematik.uni-ulm.de/~lehn/sghpc/gemm/page02/index.html)
* [cpu-conv2d](https://github.com/pytorch/pytorch/blob/f23feca681c5066c70f0fe1516fc2e269d615e93/aten/src/THNN/generic/SpatialConvolutionMM.c)
* [gpu-gemm and gpu-conv2d](https://github.com/1adrianb/bnn.torch/blob/master/BinarySpatialConvolution.cu)
* [popcount](https://github.com/kimwalisch/libpopcnt)
