import chainer
import chainer.functions as F
import chainer.links as L

class Autoencoder(chainer.ChainList):
    def __init__(self, unit=256, dim=900):
        super(Autoencoder, self).__init__(
            L.Linear(dim, int((dim-unit)/2)),
            L.Linear(int((dim-unit)/2), unit),
            L.Linear(unit, int((dim-unit)/2)),
            L.Linear(int((dim-unit)/2), dim)
        )
    def __call__(self, x, output_layer=-1):
        h = x
        if output_layer == -1:
            output_layer=len(self)
        for i in range(output_layer):
            h = F.relu(self[i](h))
        return h
            
