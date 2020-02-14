# RNN-for-Joint-NLU

Pytorch implementation of "Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling" (https://arxiv.org/pdf/1609.01454.pdf)

![](./images/jointnlu0.png)

Intent prediction and slot filling are performed in two branches based on Encoder-Decoder model.

## dataset (Atis)

You can get data from [JointSLU](https://github.com/yvchen/JointSLU/tree/master/data)


## Requirements

* `Pytorch >= 1.3.1`

## Train

`python train.py --file_path "your data path e.g. ./data/atis-2.train.w-intent.iob"`

## Run
`python run.py --train_path "the train dataset you used to train" --test_path "the test dataset you want to use"`


## Result

![](./images/jointnlu1.png)
![](./images/jointnlu2.png)
![](./images/jointnlu3.png)