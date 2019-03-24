# ChannelNets

Created by [Hongyang Gao](http://people.tamu.edu/~hongyang.gao/),
[Zhengyang Wang](http://people.tamu.edu/~zhengyang.wang/), and
[Shuiwang Ji](http://people.tamu.edu/~sji/) at Texas A&M University.

## Introduction

ChannelNets are compact and efficent CNN via Channel-wise convolutions. It has been accepted in NIPS2018.

Detailed information about ChannelNets is provided in https://papers.nips.cc/paper/7766-channelnets-compact-and-efficient-convolutional-neural-networks-via-channel-wise-convolutions.pdf.

## Citation

```
@inproceedings{gao2018channelnets,
  title={ChannelNets: Compact and Efficient Convolutional Neural Networks via Channel-Wise Convolutions},
  author={Gao, Hongyang and Wang, Zhengyang and Ji, Shuiwang},
  booktitle={Advances in Neural Information Processing Systems},
  pages={5203--5211},
  year={2018}
}
```

## Results

| Models        | Top-1 | Params | FLOPs
|-----------|-------|----------|--------|
| GoogleNet     | 0.698 | 6.8m   | 1550m
| VGG16         | 0.715 | 128m   | 15300m
| AlexNet       | 0.572 | 60m    | 720m
| SqueezeNet    | 0.575 | 1.3m   | 833m
| 1.0 MobileNet | 0.706 | 4.2m   | 569m
| ShuffleNet 2x | 0.709 | 5.3m   | 524m
| ChannelNet-v1 | 0.705 | 3.7m   | 407m

## Configure the network

All network hyperparameters are configured in main.py.
