''' Evaluating Frustum PointNets.
Write evaluation results to KITTI format labels.
and [optionally] write results to pickle files.

Author: Charles R. Qi
Date: September 2017
'''
from __future__ import print_function

import os
import sys
import argparse
import importlib
import numpy as np
import tensorflow as tf
import pickle

from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER
import provider
from train_util import get_batch


def get_session_and_ops(batch_size, num_point, model, model_path):
  ''' Define model graph, load model parameters,
  create session and return session handle and tensors
  '''
  with tf.Graph().as_default():
    with tf.device('/cpu:0'):
      (pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl,
      heading_class_label_pl, heading_residual_label_pl,
      size_class_label_pl, size_residual_label_pl) = model.placeholder_inputs(batch_size, num_point)
      is_training_pl = tf.placeholder(tf.bool, shape=())
      end_points = model.get_model(pointclouds_pl, one_hot_vec_pl, is_training_pl)
      loss = model.get_loss(labels_pl, centers_pl,
                            heading_class_label_pl, heading_residual_label_pl,
                            size_class_label_pl, size_residual_label_pl, end_points)
      saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, model_path)
    ops = {'pointclouds_pl': pointclouds_pl,
           'one_hot_vec_pl': one_hot_vec_pl,
           'labels_pl': labels_pl,
           'centers_pl': centers_pl,
           'heading_class_label_pl': heading_class_label_pl,
           'heading_residual_label_pl': heading_residual_label_pl,
           'size_class_label_pl': size_class_label_pl,
           'size_residual_label_pl': size_residual_label_pl,
           'is_training_pl': is_training_pl,
           'logits': end_points['mask_logits'],
           'center': end_points['center'],
           'end_points': end_points,
           'loss': loss}
    return sess, ops


def softmax(x):
  ''' Numpy function for softmax'''
  shape = x.shape
  probs = np.exp(x - np.max(x, axis=len(shape) - 1, keepdims=True))
  probs /= np.sum(probs, axis=len(shape) - 1, keepdims=True)
  return probs


def inference(sess, ops, pc, one_hot_vec, batch_size, num_classes):
  ''' Run inference for frustum pointnets in batch mode '''
  assert pc.shape[0] % batch_size == 0
  num_batches = pc.shape[0] // batch_size
  logits = np.zeros((pc.shape[0], pc.shape[1], num_classes))
  centers = np.zeros((pc.shape[0], 3))
  heading_logits = np.zeros((pc.shape[0], NUM_HEADING_BIN))
  heading_residuals = np.zeros((pc.shape[0], NUM_HEADING_BIN))
  size_logits = np.zeros((pc.shape[0], NUM_SIZE_CLUSTER))
  size_residuals = np.zeros((pc.shape[0], NUM_SIZE_CLUSTER, 3))
  scores = np.zeros((pc.shape[0],))  # 3D box score

  ep = ops['end_points']
  for i in range(num_batches):
    feed_dict = {
      ops['pointclouds_pl']: pc[i * batch_size:(i + 1) * batch_size, ...],
      ops['one_hot_vec_pl']: one_hot_vec[i * batch_size:(i + 1) * batch_size, :],
      ops['is_training_pl']: False}

    batch_logits, batch_centers, \
    batch_heading_scores, batch_heading_residuals, \
    batch_size_scores, batch_size_residuals = \
      sess.run([ops['logits'], ops['center'],
                ep['heading_scores'], ep['heading_residuals'],
                ep['size_scores'], ep['size_residuals']],
               feed_dict=feed_dict)

    logits[i * batch_size:(i + 1) * batch_size, ...] = batch_logits
    centers[i * batch_size:(i + 1) * batch_size, ...] = batch_centers
    heading_logits[i * batch_size:(i + 1) * batch_size, ...] = batch_heading_scores
    heading_residuals[i * batch_size:(i + 1) * batch_size, ...] = batch_heading_residuals
    size_logits[i * batch_size:(i + 1) * batch_size, ...] = batch_size_scores
    size_residuals[i * batch_size:(i + 1) * batch_size, ...] = batch_size_residuals

    # Compute scores
    batch_seg_prob = softmax(batch_logits)[:, :, 1]  # BxN
    batch_seg_mask = np.argmax(batch_logits, 2)  # BxN
    mask_mean_prob = np.sum(batch_seg_prob * batch_seg_mask, 1)  # B,
    mask_mean_prob = mask_mean_prob / np.sum(batch_seg_mask, 1)  # B,
    heading_prob = np.max(softmax(batch_heading_scores), 1)  # B
    size_prob = np.max(softmax(batch_size_scores), 1)  # B,
    batch_scores = np.log(mask_mean_prob) + np.log(heading_prob) + np.log(size_prob)
    scores[i * batch_size:(i + 1) * batch_size] = batch_scores
    # Finished computing scores

  heading_cls = np.argmax(heading_logits, 1)  # B
  size_cls = np.argmax(size_logits, 1)  # B
  heading_res = np.array([heading_residuals[i, heading_cls[i]] \
                          for i in range(pc.shape[0])])
  size_res = np.vstack([size_residuals[i, size_cls[i], :] \
                        for i in range(pc.shape[0])])

  return np.argmax(logits, 2), centers, heading_cls, heading_res, \
         size_cls, size_res, scores
