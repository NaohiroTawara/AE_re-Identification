import chainer
import chainer.functions as F
import chainer.links as L
from chainer import serializers
import numpy as np
import argparse

from AE import Autoencoder
from utils import ImageDataset

def main():
    parser = argparse.ArgumentParser(description='Autoencoder for person re-identification')
    parser.add_argument('dataset', type=str,
                        help='Directory that contains training images')
    parser.add_argument('model', type=str,
                        help='AutoEncoder model')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--unit', '-u', type=int, default=100,
                        help='Number of units [This number must be same as the input model!!]')
    args = parser.parse_args()

    dataset = ImageDataset(args.dataset, is_online=False)
    model = L.Classifier(Autoencoder(unit=args.unit, dim=dataset.shape()[0]), lossfun=F.mean_squared_error)
    serializers.load_npz(args.model, model)
    model.compute_accuracy = False
    for i,data in enumerate(dataset):
        print('%s: %f' %(dataset.Filename[i], model(np.asarray(data[np.newaxis]), np.asarray(data[np.newaxis])).data))

if __name__ == '__main__':
    main()
