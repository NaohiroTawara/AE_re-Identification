import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset
from chainer import serializers

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import argparse
import cupy

from AE import Autoencoder

from utils import ImageDataset

'''
def plot_mnist_data(samples, epoch=0):
    for index, (data, label) in enumerate(samples[0:10]):
        plt.subplot(4, 4, index + 1)
        plt.axis('off')
        plt.imshow(data.reshape(28, 28), cmap=cm.gray_r, interpolation='nearest')
        n = int(label)
        plt.title(n, color='red')
    plt.savefig("./pict/epoch_"+str(epoch)+'.png')
    #plt.show()
'''

def main():
    parser = argparse.ArgumentParser(description='Autoencoder for person re-identification')
    parser.add_argument('dataset', type=str,
                        help='Directory that contains training images')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=100,
                        help='Number of units')
    parser.add_argument('--seed', '-s', type=int, default=0,
                        help='seed of random')

    args = parser.parse_args()

    dataset = ImageDataset(args.dataset, is_online=False)
    train, test = chainer.datasets.split_dataset_random(dataset, int(len(dataset)*0.9), seed=args.seed)

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('# train data: {}'.format(len(train)))
    print('# test data: {}'.format(len(test)))
    print('# Imagedimension: {}'.format(dataset.shape()))
    print('')

    train = tuple_dataset.TupleDataset(train, train)
    test = tuple_dataset.TupleDataset(test, test)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,repeat=False, shuffle=False)

    model = L.Classifier(Autoencoder(unit=args.unit, dim=dataset.shape()[0]), lossfun=F.mean_squared_error)
    if args.resume:
        serializers.load_npz(args.resume, model)

    model.compute_accuracy = False
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)

    # apply first epoch
    trainer = training.Trainer(updater, (1, 'epoch'), out=args.out)
    trainer.run()

    # apply remaining epochs
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport( ['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())


    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))

    trainer.run()

    '''
    pred_list = []
    for (data, label) in test:
        if args.gpu>=0:            
            pred_data = chainer.cuda.to_cpu(model.predictor(chainer.cuda.to_gpu(cupy.array([data]),device=args.gpu).astype(cupy.float32)).data)
        else:
            pred_data = model.predictor(np.array([data]).astype(np.float32)).data
        pred_list.append((pred_data, label))
    plot_mnist_data(pred_list, args.epoch)
    '''

    model.to_cpu()
    serializers.save_npz(args.out + "/" + str(args.epoch) + ".npz", model)

if __name__ == '__main__':
    main()
