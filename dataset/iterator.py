import mxnet as mx
import numpy as np
import cv2
import scipy.misc
from tools.rand_sampler import RandSampler

class DetRecordIter(mx.io.DataIter):
    """
    The new detection iterator wrapper for mx.io.ImageDetRecordIter which is
    written in C++, it takes record file as input and runs faster.
    Supports various augment operations for object detection.

    Parameters:
    -----------
    path_imgrec : str
        path to the record file
    path_imglist : str
        path to the list file to replace the labels in record
    batch_size : int
        batch size
    data_shape : tuple
        (3, height, width)
    label_width : int
        specify the label width, use -1 for variable length
    label_pad_width : int
        labels must have same shape in batches, use -1 for automatic estimation
        in each record, otherwise force padding to width in case you want t
        rain/validation to match the same width
    label_pad_value : float
        label padding value
    resize_mode : str
        force - resize to data_shape regardless of aspect ratio
        fit - try fit to data_shape preserving aspect ratio
        shrink - shrink to data_shape only, preserving aspect ratio
    mean_pixels : list or tuple
        mean values for red/green/blue
    kwargs : dict
        see mx.io.ImageDetRecordIter

    Returns:
    ----------

    """
    def __init__(self, path_imgrec, batch_size, data_shape, path_imglist="",
                 label_width=-1, label_pad_width=-1, label_pad_value=-1,
                 resize_mode='force',  mean_pixels=[123.68, 116.779, 103.939],
                 **kwargs):
        super(DetRecordIter, self).__init__()
        self.rec = mx.io.ImageDetRecordIter(
            path_imgrec     = path_imgrec,
            path_imglist    = path_imglist,
            label_width     = label_width,
            label_pad_width = label_pad_width,
            label_pad_value = label_pad_value,
            batch_size      = batch_size,
            data_shape      = data_shape,
            mean_r          = mean_pixels[0],
            mean_g          = mean_pixels[1],
            mean_b          = mean_pixels[2],
            resize_mode     = resize_mode,
            **kwargs)

        self.provide_label = None
        self._get_batch()
        if not self.provide_label:
            raise RuntimeError("Invalid ImageDetRecordIter: " + path_imgrec)
        self.reset()

    @property
    def provide_data(self):
        #return self.rec.provide_data
        _provide_data = self.rec.provide_data
        (b, c, h, w) = _provide_data[0][1]
        _provide_data[0][1] = (b, c*2, h, w)
        _provide_data[0].shape = (b, c*2, h, w)
        return _provide_data

    def reset(self):
        self.rec.reset()

    def iter_next(self):
        return self._get_batch()

    def next(self):
        if self.iter_next():
            return self._batch
        else:
            raise StopIteration

    def _get_batch(self):
        self._batch = self.rec.next()
        if not self._batch:
            return False

        if self.provide_label is None:
            # estimate the label shape for the first batch, always reshape to n*5
            first_label = self._batch.label[0][0].asnumpy()
            self.batch_size = self._batch.label[0].shape[0]
            self.label_header_width = int(first_label[4])
            self.label_object_width = int(first_label[5])
            assert self.label_object_width >= 5, "object width must >=5"
            self.label_start = 4 + self.label_header_width
            self.max_objects = (first_label.size - self.label_start) // self.label_object_width
            self.label_shape = (self.batch_size, self.max_objects, self.label_object_width)
            self.label_end = self.label_start + self.max_objects * self.label_object_width
            #self.provide_label = [('label', self.label_shape)]
            self.provide_label = [('label', self.label_shape), ('label2', self.label_shape)]

        # modify label
        label = self._batch.label[0].asnumpy()
        label = label[:, self.label_start:self.label_end].reshape(
            (self.batch_size, self.max_objects, self.label_object_width))

        # central area label conversion
        label2 = label
        #get central area from label2
        #y_min = (y_min - 0.25) * 2
        #y_max = 1 - (0.75 - y_max) * 2

        # process input data: B x 3 x H x W - B x 6 x H x W
        data = self._batch.data[0].asnumpy()
        (b, c, h, w) = data.shape
        new_data = np.zeros([b, 2*c, h, w])
        # resize for each image
        data_resized = np.zeros([b, c, h, w])
        for i in range(b):
            im = data[i]
            im_cv = np.transpose(im, (1, 2, 0))
            im_cv = im_cv + 128
            #cv2.imwrite('test.jpg', im_cv[...,::-1])


            im_cropped = im_cv[h/4:h*3/4, :, :]
            #cv2.imwrite('cropped.jpg', im_cropped[...,::-1])
            im_resized = cv2.resize(im_cropped, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
            #cv2.imwrite('resized.jpg', im_resized[...,::-1])

            im_minus = im_resized - 128
            im_transback = np.transpose(im_minus, (2, 0, 1))
            data_resized[i] = im_transback

        new_data[:, :c, :, :] = data
        new_data[:, c:, :, :] = data_resized

        #self._batch.label = [mx.nd.array(label)]
        self._batch.label = [mx.nd.array(label), mx.nd.array(label2)]
        self._batch.data = [mx.nd.array(new_data)]
        return True

class DetIter(mx.io.DataIter):
    """
    Detection Iterator, which will feed data and label to network
    Optional data augmentation is performed when providing batch

    Parameters:
    ----------
    imdb : Imdb
        image database
    batch_size : int
        batch size
    data_shape : int or (int, int)
        image shape to be resized
    mean_pixels : float or float list
        [R, G, B], mean pixel values
    rand_samplers : list
        random cropping sampler list, if not specified, will
        use original image only
    rand_mirror : bool
        whether to randomly mirror input images, default False
    shuffle : bool
        whether to shuffle initial image list, default False
    rand_seed : int or None
        whether to use fixed random seed, default None
    max_crop_trial : bool
        if random crop is enabled, defines the maximum trial time
        if trial exceed this number, will give up cropping
    is_train : bool
        whether in training phase, default True, if False, labels might
        be ignored
    """
    def __init__(self, imdb, batch_size, data_shape, \
                 mean_pixels=[128, 128, 128], rand_samplers=[], \
                 rand_mirror=False, shuffle=False, rand_seed=None, \
                 is_train=True, max_crop_trial=50):
        super(DetIter, self).__init__()

        self._imdb = imdb
        self.batch_size = batch_size
        if isinstance(data_shape, int):
            data_shape = (data_shape, data_shape)
        self._data_shape = data_shape
        self._mean_pixels = mx.nd.array(mean_pixels).reshape((3,1,1))
        if not rand_samplers:
            self._rand_samplers = []
        else:
            if not isinstance(rand_samplers, list):
                rand_samplers = [rand_samplers]
            assert isinstance(rand_samplers[0], RandSampler), "Invalid rand sampler"
            self._rand_samplers = rand_samplers
        self.is_train = is_train
        self._rand_mirror = rand_mirror
        self._shuffle = shuffle
        if rand_seed:
            np.random.seed(rand_seed) # fix random seed
        self._max_crop_trial = max_crop_trial

        self._current = 0
        self._size = imdb.num_images
        self._index = np.arange(self._size)

        self._data = None
        self._label = None
        self._get_batch()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in self._data.items()]

    @property
    def provide_label(self):
        if self.is_train:
            return [(k, v.shape) for k, v in self._label.items()]
        else:
            return []

    def reset(self):
        self._current = 0
        if self._shuffle:
            np.random.shuffle(self._index)

    def iter_next(self):
        return self._current < self._size

    def next(self):
        if self.iter_next():
            self._get_batch()
            data_batch = mx.io.DataBatch(data=list(self._data.values()),
                                   label=list(self._label.values()),
                                   pad=self.getpad(), index=self.getindex())
            self._current += self.batch_size
            return data_batch
        else:
            raise StopIteration

    def getindex(self):
        return self._current // self.batch_size

    def getpad(self):
        pad = self._current + self.batch_size - self._size
        return 0 if pad < 0 else pad

    def _get_batch(self):
        """
        Load data/label from dataset
        """
        batch_data = mx.nd.zeros((self.batch_size, 3, self._data_shape[0], self._data_shape[1]))
        batch_label = []
        for i in range(self.batch_size):
            if (self._current + i) >= self._size:
                if not self.is_train:
                    continue
                # use padding from middle in each epoch
                idx = (self._current + i + self._size // 2) % self._size
                index = self._index[idx]
            else:
                index = self._index[self._current + i]
            # index = self.debug_index
            im_path = self._imdb.image_path_from_index(index)
            with open(im_path, 'rb') as fp:
                img_content = fp.read()
            img = mx.img.imdecode(img_content)
            gt = self._imdb.label_from_index(index).copy() if self.is_train else None
            data, label = self._data_augmentation(img, gt)
            batch_data[i] = data
            if self.is_train:
                batch_label.append(label)
        self._data = {'data': batch_data}

        # prepare additional data for concat
        data = self._data['data'].asnumpy()
        (b, c, h, w) = data.shape
        new_data = np.zeros([b, 2*c, h, w])
        # resize for each image
        data_resized = np.zeros([b, c, h, w])
        for i in range(b):
            im = np.copy(data[i])
            im[0] = im[0] + self._mean_pixels.asnumpy()[2][0][0]
            im[1] = im[1] + self._mean_pixels.asnumpy()[1][0][0]
            im[2] = im[2] + self._mean_pixels.asnumpy()[0][0][0]
            im_cv = np.transpose(im, (1, 2, 0))
            im_cropped = im_cv[h/4:h*3/4, :, :]
            im_resized = cv2.resize(im_cropped, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
            #cv2.imwrite('test.jpg', im_cv[...,::-1])
            #cv2.imwrite('cropped.jpg', im_cropped[...,::-1])
            #cv2.imwrite('resized.jpg', im_resized[...,::-1])
            im_resized = np.transpose(im_resized, (2, 0, 1))
            im_resized[0] = im_resized[0] - self._mean_pixels.asnumpy()[2][0][0]
            im_resized[1] = im_resized[1] - self._mean_pixels.asnumpy()[1][0][0]
            im_resized[2] = im_resized[2] - self._mean_pixels.asnumpy()[0][0][0]
            data_resized[i] = im_resized
        new_data[:, :c, :, :] = data
        new_data[:, c:, :, :] = data_resized
        #new_data[:, c:, :, :] = data  # test two same stream

        # central area label conversion
        b = data.shape[0]
        label2 = np.copy(batch_label)
        label1_pad = -1 * np.ones((b, 70, 5))
        label2_pad = -1 * np.ones((b, 70, 5))
        # adapt label coordinates
        for b in range(len(label2)):
            label_im = label2[b]
            index = 0
            for n in range(label_im.shape[0]):
                label_bb = label_im[n]
                [cls, xmin, ymin, xmax, ymax] = [i for i in label_bb]
                if ymin >= 0.25 and ymax <= 0.75:
                    ymin = (ymin - 0.25) * 2
                    ymax = 1 - (0.75 - ymax) * 2
                    label2_pad[b][index][:] = np.array([cls, xmin, ymin, xmax, ymax])
                    index = index + 1

        # TODO: augumentation

        self._data = {'data': mx.io.array(new_data)}

        # pad label with 0
        for i in range(len(batch_label)):
            for j in range(len(batch_label[i])):
                label1_pad[i][j] = batch_label[i][j]

        if self.is_train:
            #self._label = {'label': mx.nd.array(np.array(batch_label))}
            self._label = {'label': mx.nd.array(np.array(label1_pad)),
                           'label2': mx.nd.array(np.array(label2_pad))}
        else:
            #self._label = {'label': None}
            self._label = {'label': None, 'label2': None}

    def _data_augmentation(self, data, label):
        """
        perform data augmentations: crop, mirror, resize, sub mean, swap channels...
        """
        if self.is_train and self._rand_samplers:
            rand_crops = []
            for rs in self._rand_samplers:
                rand_crops += rs.sample(label)
            num_rand_crops = len(rand_crops)
            # randomly pick up one as input data
            if num_rand_crops > 0:
                index = int(np.random.uniform(0, 1) * num_rand_crops)
                width = data.shape[1]
                height = data.shape[0]
                crop = rand_crops[index][0]
                xmin = int(crop[0] * width)
                ymin = int(crop[1] * height)
                xmax = int(crop[2] * width)
                ymax = int(crop[3] * height)
                if xmin >= 0 and ymin >= 0 and xmax <= width and ymax <= height:
                    data = mx.img.fixed_crop(data, xmin, ymin, xmax-xmin, ymax-ymin)
                else:
                    # padding mode
                    new_width = xmax - xmin
                    new_height = ymax - ymin
                    offset_x = 0 - xmin
                    offset_y = 0 - ymin
                    data_bak = data
                    data = mx.nd.full((new_height, new_width, 3), 128, dtype='uint8')
                    data[offset_y:offset_y+height, offset_x:offset_x + width, :] = data_bak
                label = rand_crops[index][1]
        if self.is_train:
            interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, \
                              cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        else:
            interp_methods = [cv2.INTER_LINEAR]
        interp_method = interp_methods[int(np.random.uniform(0, 1) * len(interp_methods))]
        data = mx.img.imresize(data, self._data_shape[1], self._data_shape[0], interp_method)
        if self.is_train and self._rand_mirror:
            if np.random.uniform(0, 1) > 0.5:
                data = mx.nd.flip(data, axis=1)
                valid_mask = np.where(label[:, 0] > -1)[0]
                tmp = 1.0 - label[valid_mask, 1]
                label[valid_mask, 1] = 1.0 - label[valid_mask, 3]
                label[valid_mask, 3] = tmp
        data = mx.nd.transpose(data, (2,0,1))
        data = data.astype('float32')
        data = data - self._mean_pixels
        return data, label
