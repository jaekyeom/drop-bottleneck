from __future__ import print_function

from collections import OrderedDict, defaultdict
import numpy as np
import random
import copy
#from mpi_util import mpi_moments


#def fc(x, scope, nh, *, init_scale=1.0, init_bias=0.0):
#    with tf.variable_scope(scope):
#        nin = x.get_shape()[1].value
#        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
#        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias))
#        return tf.matmul(x, w)+b
#
#def conv(x, scope, *, nf, rf, stride, pad='VALID', init_scale=1.0, data_format='NHWC', one_dim_bias=False, bias_initializer=tf.constant_initializer(0.0)):
#    if data_format == 'NHWC':
#        channel_ax = 3
#        strides = [1, stride, stride, 1]
#        bshape = [1, 1, 1, nf]
#    elif data_format == 'NCHW':
#        channel_ax = 1
#        strides = [1, 1, stride, stride]
#        bshape = [1, nf, 1, 1]
#    else:
#        raise NotImplementedError
#    bias_var_shape = [nf] if one_dim_bias else [1, nf, 1, 1]
#    nin = x.get_shape()[channel_ax].value
#    wshape = [rf, rf, nin, nf]
#    with tf.variable_scope(scope):
#        w = tf.get_variable("w", wshape, initializer=ortho_init(init_scale))
#        b = tf.get_variable("b", bias_var_shape, initializer=bias_initializer)
#        if not one_dim_bias and data_format == 'NHWC':
#            b = tf.reshape(b, bshape)
#        return b + tf.nn.conv2d(x, w, strides=strides, padding=pad, data_format=data_format)
#
#
#def deconv(x, scope, *, nf, rf, stride, init_scale=1.0, data_format='NHWC'):
#    if data_format == 'NHWC':
#        channel_ax = 3
#        strides = (stride, stride)
#        #strides = [1, stride, stride, 1]
#    elif data_format == 'NCHW':
#        channel_ax = 1
#        strides = (stride, stride)
#        #strides = [1, 1, stride, stride]
#    else:
#        raise NotImplementedError
#
#    with  tf.variable_scope(scope):
#        out = tf.contrib.layers.conv2d_transpose(x,
#                                                num_outputs=nf,
#                                                kernel_size=rf,
#                                                stride=strides,
#                                                padding='VALID',
#                                                weights_initializer=ortho_init(init_scale),
#                                                biases_initializer=tf.constant_initializer(0.0),
#                                                activation_fn=None,
#                                                data_format=data_format)
#        return out
#
#
#def ortho_init(scale=1.0):
#    def _ortho_init(shape, dtype, partition_info=None):
#        #lasagne ortho init for tf
#        shape = tuple(shape)
#        if len(shape) == 2:
#            flat_shape = shape
#        elif len(shape) == 4: # assumes NHWC
#            flat_shape = (np.prod(shape[:-1]), shape[-1])
#        else:
#            raise NotImplementedError
#        a = np.random.normal(0.0, 1.0, flat_shape)
#        u, _, v = np.linalg.svd(a, full_matrices=False)
#        q = u if u.shape == flat_shape else v # pick the one with the correct shape
#        q = q.reshape(shape)
#        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
#    return _ortho_init

def tile_images(array, n_cols=None, max_images=None, div=1):
    if max_images is not None:
        array = array[:max_images]
    if len(array.shape) == 4 and array.shape[3] == 1:
        array = array[:, :, :, 0]
    assert len(array.shape) in [3, 4], "wrong number of dimensions - shape {}".format(array.shape)
    if len(array.shape) == 4:
        assert array.shape[3] == 3, "wrong number of channels- shape {}".format(array.shape)
    if n_cols is None:
        n_cols = max(int(np.sqrt(array.shape[0])) // div * div, div)
    n_rows = int(np.ceil(float(array.shape[0]) / n_cols))

    def cell(i, j):
        ind = i * n_cols + j
        return array[ind] if ind < array.shape[0] else np.zeros(array[0].shape)

    def row(i):
        return np.concatenate([cell(i, j) for j in range(n_cols)], axis=1)

    return np.concatenate([row(i) for i in range(n_rows)], axis=0)


def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        #from mpi4py import MPI
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)


#def explained_variance_non_mpi(ypred,y):
#    """
#    Computes fraction of variance that ypred explains about y.
#    Returns 1 - Var[y-ypred] / Var[y]
#
#    interpretation:
#        ev=0  =>  might as well have predicted zero
#        ev=1  =>  perfect prediction
#        ev<0  =>  worse than just predicting zero
#
#    """
#    assert y.ndim == 1 and ypred.ndim == 1
#    vary = np.var(y)
#    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary
#
#def mpi_var(x):
#    return mpi_moments(x)[1]**2
#
#def explained_variance(ypred,y):
#    """
#    Computes fraction of variance that ypred explains about y.
#    Returns 1 - Var[y-ypred] / Var[y]
#
#    interpretation:
#        ev=0  =>  might as well have predicted zero
#        ev=1  =>  perfect prediction
#        ev<0  =>  worse than just predicting zero
#
#    """
#    assert y.ndim == 1 and ypred.ndim == 1
#    vary = mpi_var(y)
#    return np.nan if vary==0 else 1 - mpi_var(y-ypred)/vary


def add_noise(img, noise_p, noise_type):
    noise_mask = np.random.binomial(1, noise_p, size=img.shape[0]).astype(np.bool)
    w = 12
    n = 84//12
    idx_list = np.arange(n*n)
    random.shuffle(idx_list)
    idx_list = idx_list[:np.random.randint(10, 40)]
    for i in range(img.shape[0]):
        if not noise_mask[i]:
            continue
        for idx in idx_list:
            y = (idx // n)*w
            x = (idx % n)*w
            img[i, y:y+w, x:x+w, -1] += np.random.normal(0, 255*0.3, size=(w,w)).astype(np.uint8)

    img = np.clip(img, 0., 255.)
    return img

g_font = [None]
def draw_text_to_image(text, height=None, width=None, channels=None):
    from PIL import Image, ImageDraw, ImageFont
    if g_font[0] is None:
        g_font[0] = ImageFont.load_default()
    font = g_font[0]

    # ImageFont.ImageFont.getsize doesn't work for multi-line strings.
    # https://github.com/python-pillow/Pillow/issues/2966
    #text_size = font.getsize(text)
    dummy_img = Image.fromarray(np.zeros((1, 1), dtype=np.uint8))
    dummy_draw = ImageDraw.Draw(dummy_img)
    text_size = dummy_draw.textsize(text, font=font)

    if channels is None:
        shape = (height or text_size[1], width or text_size[0])
    else:
        shape = (height or text_size[1], width or text_size[0], channels)
    i = np.zeros(shape, dtype=np.uint8)
    img = Image.fromarray(i)
    draw = ImageDraw.Draw(img)
    draw.text((3, 0), text, font=font, fill=(255,)*channels)
    return np.asarray(img)

def get_percentile_indices(data, percentiles=np.arange(0.0, 1.05, 0.1)):
    assert len(data.shape) == 1
    data_asc = np.argsort(data)
    percentile_indices = (percentiles * (len(data_asc) - 1)).astype(int)
    percentile_indices = data_asc[percentile_indices]
    #assert np.all(data[percentile_indices[:-1]] <= data[percentile_indices[1:]])
    return percentile_indices

class CContext():
    def __init__(self, verbose=False, print_func=print):
        self._state_funcs = OrderedDict()
        self._evaluated_states = OrderedDict()
        self._dependencies = defaultdict(set)
        self._eval_context = []
        self._verbose = verbose
        self._print_func = print_func

    def register_state(self, name, create):
        if name in self._state_funcs:
            raise Exception('State already registered: {}'.format(name))
        self._state_funcs[name] = create

    def invalidate_state(self, name):
        if name not in self._evaluated_states:
            return
        del self._evaluated_states[name]
        if self._verbose:
            self._print_func('Invalidated state "{}"'.format(name))
        for n in self._dependencies[name]:
            self.invalidate_state(n)
        del self._dependencies[name]

    def __getattr__(self, attr):
        if attr not in self._state_funcs:
            raise Exception('Unknown state {}'.format(attr))
        if attr in self._eval_context:
            raise Exception('Circular dependency detected: {}, {}'.format(attr, self._eval_context))
        self._dependencies[attr] = self._dependencies[attr].union(set(self._eval_context))
        if attr not in self._evaluated_states:
            self._eval_context.append(attr)
            evaluated_state = self._state_funcs[attr](self)
            if self._verbose:
                self._print_func('Evaluated state "{}"'.format(attr))
            self._eval_context.pop()
            self._evaluated_states[attr] = evaluated_state
        return self._evaluated_states[attr]

class EmptyClass():
    pass


# From https://github.com/openai/large-scale-curiosity/blob/0c3d179fd61ee46233199d0891c40fbe7964d3aa/cppo_agent.py#L226-L236
class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

class SimpleWeightedMovingScalarMeanStd():
    def __init__(self, alpha=0.0001):
        self._alpha = alpha
        self.mean = 0.0
        self.var = 1.0

    def update(self, values):
        self.mean = (1 - self._alpha) * self.mean + self._alpha * np.mean(values)
        self.var = (1 - self._alpha) * self.var + self._alpha * np.mean(np.square(values - self.mean))

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

