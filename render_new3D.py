#!/usr/bin/env python
# coding: utf-8
import os

# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

import numpy as np
import imageio
import pprint
import pathlib

import matplotlib.pyplot as plt

import segment_ultra_nerf_v03 as run_nerf_ultrasound
from load_us import load_us_data

basedir = './logs/for_paper/correct_thyroid/'
expname = 'for_paper_thyroid_v36_correct_noMod'

testposes = np.load('./data/correct_thyroid/test_pose_15.npy')
testposes = testposes.astype(np.float32)
#testposes[:, :3, :3] *= 1000
testposes[:, :3, 3] *= 0.001

config = os.path.join(basedir, expname, 'config.txt')
print('Args:')
print(open(config, 'r').read())
parser = run_nerf_ultrasound.config_parser()
model_no = 'model_480000'

args = parser.parse_args('--config {} --ft_path {}'.format(config, os.path.join(basedir, expname, model_no + ".npy")))
print('loaded args')
model_name = args.datadir.split("/")[-1]
images, poses, i_test = load_us_data(args.datadir)
H, W = images[0].shape

H = int(H)
W = int(W)

images = images.astype(np.float32)
poses = poses.astype(np.float32)

near = 0.
far = args.probe_depth * 0.001

# In[3]:

print('testposes[0][2]', testposes[0][2])

# Create nerf model
_, render_kwargs_test, start, grad_vars, models = run_nerf_ultrasound.create_nerf(args)
render_kwargs_test["args"] = args
bds_dict = {
    'near': tf.cast(near, tf.float32),
    'far': tf.cast(far, tf.float32),
}
render_kwargs_test.update(bds_dict)

print('Render kwargs:')
pprint.pprint(render_kwargs_test)
sw = args.probe_width * 0.001 / float(W)
sh = args.probe_depth * 0.001 / float(H)

down = 4
render_kwargs_fast = {k: render_kwargs_test[k] for k in render_kwargs_test}

frames = []
impedance_map = []
map_number = 0
output_dir = "{}/render_test/{}_rt16/render_result_{}_{}_{}/".format(basedir, expname, model_name, model_no, map_number)
output_dir_params = "{}/params/".format(output_dir)
output_dir_output = "{}/output/".format(output_dir, expname, model_name, model_no)
os.makedirs(output_dir)
os.mkdir(output_dir_params)
os.mkdir(output_dir_output)


def show_colorbar(image, name=None, cmap='rainbow', np_a=False):
    figure = plt.figure()
    if np_a:
        image_out = plt.imshow(image, cmap=cmap)
    else:
        image_out = plt.imshow(image.numpy(), cmap=cmap, vmin=0, vmax=1)
    plt.tick_params(top=False, bottom=False, left=False, right=False,
                    labelleft=False, labelbottom=False)
    m = plt.cm.ScalarMappable(cmap=cmap)
    m.set_clim(0., 1.)
    plt.colorbar(m)
    figure.savefig(name)
    plt.close(figure)
    return image_out
save_it = 300

rendering_params_save = None
for i, c2w in enumerate(testposes):
    print(i)

    # run_out = output_dir + str(i) + "/"
    # os.mkdir(run_out)
    rendering_params = run_nerf_ultrasound.render_us(H, W, sw, sh, c2w=c2w[:3, :4], **render_kwargs_fast)
    imageio.imwrite(output_dir_output + "Generated_" + str(2000 + i) + "_test.png",
                    tf.image.convert_image_dtype(tf.transpose(rendering_params['final_intensity_map']), tf.uint8))
    imageio.imwrite(output_dir_output + "Generated_segmentation" + str(2000 + i) + "_test.png",
                    tf.image.convert_image_dtype(tf.transpose(tf.round(rendering_params['positive_segmentation'])), tf.uint8))
    
    if rendering_params_save is None:
        rendering_params_save = dict()
        for key, value in rendering_params.items():
            rendering_params_save[key] = list()
    for key, value in rendering_params.items():
        rendering_params_save[key].append(tf.transpose(value).numpy())
        if np.all(rendering_params_save[key][0] == value):
            raise Exception

    if i == save_it:
        for key, value in rendering_params_save.items():
            np_to_save = np.array(value)
            np.save(f"{output_dir_params}/{key}.npy", np_to_save)
        rendering_params_save = None
    if i != save_it and i % save_it == 0 and i != 0:

        for key, value in rendering_params_save.items():
            f_name = f"{output_dir_params}/{key}.npy"
            np_to_save = np.array(value)
            np_existing = np.load(f_name)
            new_to_save = np.concatenate((np_existing, np_to_save), axis=0)
            np.save(f_name, new_to_save)
        rendering_params_save = None

for key, value in rendering_params_save.items():
    f_name = f"{output_dir_params}/{key}.npy"
    path_to_save = pathlib.Path(f_name)
    np_to_save = np.array(value)
    if path_to_save.exists():
        np_existing = np.load(f_name)
        np_to_save = np.concatenate((np_existing, np_to_save), axis=0)
    np.save(f_name, np_to_save)
