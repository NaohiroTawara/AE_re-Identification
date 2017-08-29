# AE_re-Identification

## How to install
pip install chainer
pip install pillow

## How to use

### training
python train_autoencoder.py <path to directory which contains training images> --gpu 0 --batchsize 32 --epoch 100 --unit 100

### evaluating
python calc_score.py <path to directory which contains training images>  result/100.npz --unit 100
