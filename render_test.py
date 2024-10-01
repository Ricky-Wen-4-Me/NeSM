#This is the python code using for testing the NeSM
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
from load_us import load_segment
from criteria import *

# The path of pre-trained model & render result location.
# Change it if u trained another NeSM.
basedir = './logs/for_paper/ablation_study/'
expname = 'for_paper_USNeRF_v1_noMod'

testimages_dir = './data/synthetic_testing/l2_test'
testsegments_dir = './segment_data/result_US_test_l2'

testimages, testposes, test_test = load_us_data(testimages_dir) 

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

print("training number, testing_number: ", images.shape, testimages.shape)
GT_seg_test = load_segment(testsegments_dir)
print("The number and size of segmentation map: ", GT_seg_test.shape)

# Create nerf model
_, render_kwargs_test, start, grad_vars, models = run_nerf_ultrasound.create_nerf(args)
render_kwargs_test["args"] = args
bds_dict = {
    'near': tf.cast(near, tf.float32),
    'far': tf.cast(far, tf.float32),
}
render_kwargs_test.update(bds_dict)

sw = args.probe_width * 0.001 / float(W)
sh = args.probe_depth * 0.001 / float(H)

down = 4
render_kwargs_fast = {k: render_kwargs_test[k] for k in render_kwargs_test}

frames = []
impedance_map = []
map_number = 0
output_dir = "{}/true_render_test/USNeRF/{}/{}_{}_{}_v1/".format(basedir, expname, model_name, model_no, map_number)
output_dir_params = "{}/params/".format(output_dir)
output_dir_output = "{}/output/".format(output_dir, expname, model_name, model_no)
#os.makedirs(output_dir)
#os.mkdir(output_dir_params)
#os.mkdir(output_dir_output)

#save_it = 300

#rendering_params_save = None
for i, c2w in enumerate(testposes):
    print(i)
    target_segment = tf.transpose(GT_seg_test[i])
    target_segment = tf.round(target_segment)

    rendering_params = run_nerf_ultrasound.render_us(H, W, sw, sh, c2w=c2w[:3, :4], **render_kwargs_fast)
#    imageio.imwrite(output_dir_output + "Generated_" + str(2000 + i) + "_test.png",
#                    tf.image.convert_image_dtype(tf.transpose(rendering_params['final_intensity_map']), tf.uint8))
#    imageio.imwrite(output_dir_output + "Generated_segmentation" + str(2000 + i) + "_test.png",
#                    tf.image.convert_image_dtype(tf.transpose(tf.round(rendering_params['positive_segmentation'])), tf.uint8))
    output_image = rendering_params['final_intensity_map']
    output_segment = rendering_params['positive_occupied']
    
    o_seg = tf.expand_dims(tf.expand_dims(output_segment, 0), -1)
    t_seg = tf.expand_dims(tf.expand_dims(target_segment, 0), -1)

    dice, dice_loss = dice_coefficient(o_seg, t_seg)
    m_iou = mean_iou(o_seg, t_seg)
    pa = pixel_accuary(o_seg, t_seg)
    print("dice: ", dice)
    print("mean IoU: ", m_iou)
    print("accuary: ", pa)
