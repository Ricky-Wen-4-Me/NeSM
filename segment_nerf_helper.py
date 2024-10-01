
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.constraints import Constraint
from tensorflow.keras import backend as K
import numpy as np
import io
import matplotlib.pyplot as plt

# Misc utils
def show_colorbar(image, cmap='rainbow'):
    figure = plt.figure(figsize=(5, 5))
    plt.imshow(image.numpy(), cmap=cmap)
    plt.colorbar()
    buf = io.BytesIO()
    # Use plt.savefig to save the plot to a PNG in memory.
    plt.savefig(buf, format='png')
    plt.close(figure)
    return buf
def img2mse(x, y): return tf.reduce_mean(tf.square(x - y))


def mse2psnr(x): return -10.*tf.log(x)/tf.log(10.)


def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)

def define_image_grid_3D_np(x_size, y_size):
    y = np.array(range(x_size))
    x = np.array(range(y_size))
    xv, yv = np.meshgrid(x, y, indexing='ij')
    image_grid_xy = np.vstack((xv.ravel(), yv.ravel()))
    z = np.zeros(image_grid_xy.shape[1])
    image_grid = np.vstack((image_grid_xy, z))
    return image_grid

# Positional encoding

class Embedder:

    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):

        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        B = self.kwargs['B']


        if self.kwargs['log_sampling']:
            freq_bands = 2.**tf.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = tf.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                if B is not None:
                    embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq, B=B: p_fn(x @ tf.transpose(B) * freq))
                    out_dim += d
                    out_dim += B.shape[1]
                else:
                    embed_fns.append(lambda x, p_fn=p_fn,
                                            freq=freq,: p_fn(x * freq))
                    out_dim += d


        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return tf.concat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0, b=0):

    if i == -1:
        return tf.identity, 3
    if b != 0:
        #TODO: check seed
        B = tf.random.normal((b, 3), seed=1)
    else:
        B = None

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [tf.math.sin, tf.math.cos],
        'B': B
    }
    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim

def feature_vector_transformation(W, path="segment_data/thyroid/pseudo_color_prediction/"):
    file_list = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    file_list = [f for f in file_list if f.endswith('.png')]
    img_tensor = []

    for file_name in file_list:
        file_path = os.path.join(path, file_name)

        img = load_img(file_path)
        img_array = img_to_array(img)

        img_tensor.append(tf.image.rgb_to_grayscale(tf.image.convert_image_dtype(img_array, dtype=tf.float32)))

    img_tf_array = tf.stack(img_tensor, axis=0)
    img_arr_sqz = np.squeeze(img_tf_array)

    arr_sum_dim0 = np.sum(img_arr_sqz,axis = 0)
    vec_sum_dim0 = np.sum(arr_sum_dim0,axis = 0)

    if path=="segment_data/thyroid/pseudo_color_prediction/":
        compressed_img = tf.convert_to_tensor(vec_sum_dim0, dtype=tf.float32)
        return compressed_img
    elif path=="segment_data/thyroid/added_prediction/":
        def softmax(x):
            exp_x = np.exp(x - np.max(x))
            return exp_x / exp_x.sum(axis=0, keepdims=True)

        vec_sum_dim0_softmax = softmax(vec_sum_dim0)
        compressed_img = tf.convert_to_tensor(vec_sum_dim0_softmax, dtype=tf.float32)
        return compressed_img

# Model architecture

def init_nerf_model(D=8, W=256, input_ch=3, input_ch_views=3, output_ch=6, skips=[4], use_viewdirs=False, mode=0, modulation=False, use_segment=False):

    relu = tf.keras.layers.LeakyReLU(alpha=0.1)
    def dense(W, act=relu, modulation=None):
        hidden = tf.keras.layers.Dense(W, activation=act)
        return hidden

    print('MODEL', input_ch, input_ch_views, type(
        input_ch), type(input_ch_views), use_viewdirs)
    input_ch = int(input_ch)
    input_ch_views = int(input_ch_views)

    inputs = tf.keras.Input(shape=(input_ch + input_ch_views))
    inputs_pts, inputs_views = tf.split(inputs, [input_ch, input_ch_views], -1)
    inputs_pts.set_shape([None, input_ch])
    inputs_views.set_shape([None, input_ch_views])

    print(inputs.shape, inputs_pts.shape, inputs_views.shape)
    outputs = inputs_pts
    print("input {}".format(inputs_pts.shape))
    
    if mode == 0: #pseudo color
        path = 'segment_data/thyroid/pseudo_color_prediction/'
    elif mode == 1: #added
        path = 'segment_data/thyroid/added_prediction/'
    
    modulation_v_set = feature_vector_transformation(W, path)
    print(modulation_v_set)

    for i in range(D):
        outputs = dense(W)(outputs)
        print("{} layer, {} shape".format(i, outputs.shape))
           
        if i in skips:
            outputs = tf.concat([inputs_pts, outputs], -1)
    
    if use_viewdirs:
        bottleneck = dense(256, act=None)(outputs)
        inputs_viewdirs = tf.concat(
            [bottleneck, inputs_views], -1)
        outputs = inputs_viewdirs
        for i in range(4):
            outputs = dense(W//2)(outputs)
        outputs = dense(output_ch-1, act=None)(outputs)
        outputs = tf.concat([outputs, alpha_out], -1)
    elif use_segment:
        realReLU = tf.keras.layers.ReLU(max_value=1.0)
        
        bottleneck = dense(5, act=None)(outputs)

        for i in range(2):
            outputs = dense(W//2)(outputs)
        outputs = dense(1, act=relu)(outputs)

        outputs = tf.concat([bottleneck, outputs], -1)
    else:
        outputs = dense(output_ch, act=None)(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    if modulation:
        dense_3_w = model.get_layer('dense_3').get_weights()
    #print(dense_3_w.shape)
        print("dense_3_w: ", dense_3_w, len(dense_3_w))
        for i in range(len(dense_3_w[0])):
            dense_3_w[0][i] = dense_3_w[0][i] + modulation_v_set[i]
        #print(dense_3_w[0][0])
        model.get_layer('dense_3').set_weights(dense_3_w)
        
    model.summary()
    return model

# Ray helpers
def get_rays_us_linear(H, W, sw, sh, c2w):
    t = c2w[:3, -1]
    R = c2w[:3, :3]
    x = tf.range(-W/2, W/2, dtype=tf.float32) * sw
    y = tf.zeros_like(x)
    z = tf.zeros_like(x)

    origin_base = tf.stack([x, y, z], axis=1)
    origin_base_prim = origin_base[..., None, :]
    origin_rotated = R * origin_base_prim
    ray_o_r = tf.reduce_sum(origin_rotated, axis=-1)
    rays_o = ray_o_r + t

    dirs_base = tf.constant([0., 1., 0.])
    dirs_r = tf.linalg.matvec(R, dirs_base)
    rays_d = tf.broadcast_to(dirs_r, rays_o.shape)

    return rays_o, rays_d


# Hierarchical sampling helper

def sample_pdf(bins, weights, N_samples, det=False):

    # Get pdf
    weights += 1e-5  # prevent nans
    pdf = weights / tf.reduce_sum(weights, -1, keepdims=True)
    cdf = tf.cumsum(pdf, -1)
    cdf = tf.concat([tf.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = tf.linspace(0., 1., N_samples)
        u = tf.broadcast_to(u, list(cdf.shape[:-1]) + [N_samples])
    else:
        u = tf.random.uniform(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    inds = tf.searchsorted(cdf, u, side='right')
    below = tf.maximum(0, inds-1)
    above = tf.minimum(cdf.shape[-1]-1, inds)
    inds_g = tf.stack([below, above], -1)
    cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = tf.where(denom < 1e-5, tf.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples
