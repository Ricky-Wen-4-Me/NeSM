import numpy as np
import tensorflow as tf

def confusion_matrix_old(pred_label, ground_truth):
    #print(ground_truth[0].shape, pred_label[0].shape) # origin: (1, 512, 256, 1)
    ground_truth = np.squeeze(ground_truth[0])
    pred_label = np.squeeze(pred_label[0])
    #print(np.max(pred_label))   # 1
    gt_equal_pred = np.where(ground_truth==pred_label, 1, 0)
    gt_is_false = np.where(np.logical_not(ground_truth), 1, 0)

    tp = np.sum(gt_equal_pred * ground_truth)
    fn = np.sum(ground_truth) - tp
    fp = np.sum(pred_label) - tp
    tn = np.sum(gt_equal_pred * gt_is_false)
    #print("tp, fn, fp, tn: ", tp, fn, fp, tn)
    return tp, fn, fp, tn

def make_confusion_matrix(pred_label, ground_truth):
    conf_mat = skl.metrics.confusion_matrix(tf.round(ground_truth), tf.round(pred_label))

    tn, fp, fn, tp = conf_mat.ravel()

    return tp, fn, fp, tn

def dice_coefficient(pred_label, ground_truth):
    tp, fn, fp, tn = confusion_matrix_old(pred_label, ground_truth)

    all_positive = fp + fn + (2 * tp)
    true_positive = 2 * tp

    dice = tf.math.divide_no_nan(true_positive, all_positive)
    dice_loss = 1. - dice

    return dice, dice_loss

def mean_iou(pred_label, ground_truth):
    total_tp = 0
    total_fn = 0
    total_fp = 0
    #print(pred_label.shape, ground_truth.shape) # origin: (1, 512, 256, 1)
    for pred_label, ground_truth in zip(pred_label, ground_truth):
        pred_label = np.expand_dims(pred_label, axis=0)
        ground_truth = np.expand_dims(ground_truth, axis=0)
        #print(pred_label.shape, ground_truth.shape) # element in zip(pred_label, ground_truth): (512, 256, 1)
        tp, fn, fp, _ = confusion_matrix_old(pred_label, ground_truth)

        total_tp += tp
        total_fn += fn
        total_fp += fp

    m_iou = tf.math.divide_no_nan(total_tp, (total_tp + total_fp + total_fn))

    return m_iou

def pixel_accuary(pred_label, ground_truth):
    tp, fn, fp, tn = confusion_matrix_old(pred_label, ground_truth)

    all_pred = tp + tn + fp + fn
    true_pred = tp + tn

    pixel_accuary = tf.math.divide_no_nan(true_pred, all_pred)

    return pixel_accuary
