import numpy as np
from PIL import Image
import imghdr, os
import chainer

class ImageDataset(chainer.dataset.DatasetMixin):
    def __init__(self, dirname, is_online=True, is_color=False):
        '''Read all images in the specified directory.
        Args of constractor:
          dirname (str): A directory which contain images to be read
          is_online (bool): If `True` all images are read on memory in advance, otherwise they are read on demand.
          is_color (bool): if `True` images are color
        '''
        self.Image = []
        self.Filename=[]
        self.is_color = is_color
        self.is_online= is_online
        self.image_shape = None
        for file in os.listdir(dirname):
            filename=os.path.join(dirname, file)
            if imghdr.what(filename) != None: # check whether the filename is image
                if is_online:
                    self.Image.append(filename)
                else:
                    self.Image.append(read_image(filename, color=self.is_color).flatten())
                if self.image_shape is None:
                    self.image_shape=read_image(filename, color=self.is_color).flatten().shape
                self.Filename.append(file)
    def shape(self):
        return self.image_shape
    def __len__(self):
        return len(self.Image)

    def get_example(self, i):
        if self.is_online:
            img = read_image(self.Image[i], color=self.is_color).flatten()
        else:
            img = self.Image[i]
        return img

def read_image(path, dtype=np.float32, color=True):
    """Read an image from a file.
    This function reads an image from given file. The image is CHW format and
    the range of its value is :math:`[0, 255]`. If :obj:`color = True`, the
    order of the channels is RGB.
    Args:
        path (str): A path of image file.
        dtype: The type of array. The default value is :obj:`~numpy.float32`.
        color (bool): This option determines the number of channels.
            If :obj:`True`, the number of channels is three. In this case,
            the order of the channels is RGB. This is the default behaviour.
            If :obj:`False`, this function returns a grayscale image.
    Returns:
        ~numpy.ndarray: An image.
    """

    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))
