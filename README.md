# Dynamic Computational Time for Recurrent Attention Model (DT-RAM)
Torch implementation of DT-RAM form [https://arxiv.org/pdf/1703.10332.pdf](https://arxiv.org/pdf/1703.10332.pdf)  with training/testing scripts.

# Requirements

* Install [Torch](http://torch.ch/docs/getting-started.html) on a machine with CUDA GPU
* Install [cuDNN v4 or v5](https://developer.nvidia.com/cudnn) and the Torch [cuDNN bindings](https://github.com/soumith/cudnn.torch/tree/R4)
* Download [Trained ResNet](https://github.com/facebook/fb.resnet.torch/tree/master/pretrained) on DT-RAM folder.
* Download the [CUB_bird](http://www.vision.caltech.edu/visipedia/CUB-200.html) dataset and run `process.py` as:
`python process.py [Path to CUB_bird]` 

If you already have Torch installed, make sure you have installed `rnn`, `dpnn`,`optim`,`dp`,`net-toolkit` and `cuimage`.

# Training

The training scripts come with several options, which can be listed with the `--help` flag.

`th main.lua --help`

To run the training, simply run demo.sh. By default, the script runs 3 step DT-RAM based ResNet-50 on CUB with 4 GPU and 4 data-loader threads.

`sh demo.sh train.list val.list 3`

# Testing on MNIST

To view some example results, you can directly do as follow. It will run a 9 step DT-RAM on mnist dataset

`cd mnist`

`th recurrent-visual-attention-dynamic.lua --testOnly --xpPath ../save/model_mnist.t7`

# Performance on Fine-Grained Recognition
We train and test DT-RAM on MNIST, CUB-200-2011 and Stanford Cars dataset. Performance on the three datasets are:

| MNIST         | Error (%)   |
| ------------- | ----------- |
| RAM 4 Steps   | 1.54 |
| RAM 5 Steps   | 1.34 |
| RAM 7 steps   | 1.07 |
| DT-RAM 5.2 Steps | 1.12 |

| CUB-200-2011       | Accuracy (%) |
| ------------- | ----------- |
| ResNet-50 Baseline     | 84.5 |
| RAM 3 Steps     | 86.0 |
| DT-RAM 1.9 Steps | 86.0 |

| Stanford Cars       | Accuracy (%) |
| ------------- | ----------- |
| ResNet-50 Baseline     | 92.3 |
| RAM 3 Steps     | 93.1 |
| DT-RAM 1.9 Steps | 93.1 |

# reference
This implements training of DT-RAM based on RAM([Recurrent Models of Visual Attention](https://arxiv.org/pdf/1406.6247.pdf))
,
and we use the framework [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch)
