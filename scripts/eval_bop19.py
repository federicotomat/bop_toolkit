# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Evaluation script for the BOP Challenge 2019."""

import os
import time
import argparse
import subprocess
import numpy as np

from auto_pose.bop_toolkit.bop_toolkit_lib import config
from auto_pose.bop_toolkit.bop_toolkit_lib import inout
from auto_pose.bop_toolkit.bop_toolkit_lib import misc


# PARAMETERS (some can be overwritten by the command line arguments below).
################################################################################
p = {
  # Errors to calculate.
  #'errors': [
  #  {
  #    'n_top': 1,
  #    'type': 'vsd',
  #    'vsd_deltas': {
  #      'hb': 15,
  #      'icbin': 15,
  #      'icmi': 15,
  #      'itodd': 5,
  #      'lm': 15,
  #      'lmo': 15,
  #      'ruapc': 15,
  #      'tless': 15,
  #      'tudl': 15,
  #      'tyol': 15,
  #      'ycbv': 15,
  #    },
  #    'vsd_taus': list(np.arange(20.0)),
  #    'vsd_normalized_by_diameter': True,
  #    'correct_th': [[th] for th in np.arange(0.30, 0.30)]
  #  },
  #  {
  #    'n_top': -1,
  #    'type': 'mssd',
  #    'correct_th': [[th] for th in np.arange(0.30, 0.30)]
  #  },
  #  {
  #    'n_top': -1,
  #    'type': 'mspd',
  #    'correct_th': [[th] for th in np.arange(5, 51, 5)]
  # },
  #],

  # Minimum visible surface fraction of a valid GT pose.
  # -1 == k most visible GT poses will be considered, where k is given by
  # the "inst_count" item loaded from "targets_filename".
  #'visib_gt_min': -1,

  # See misc.get_symmetry_transformations().
  'max_sym_disc_step': 0.01,

  # Type of the renderer (used for the VSD pose error function).
  #'renderer_type': 'cpp',  # Options: 'cpp', 'python'.

  # Names of files with results for which to calculate the errors (assumed to be
  # stored in folder p['results_path']). See docs/bop_challenge_2019.md for a
  # description of the format. Example results can be found at:
  # http://ptak.felk.cvut.cz/6DB/public/bop_sample_results/bop_challenge_2019/
  #'result_filenames': [
  #  '/relative/path/to/csv/with/results',
  #],

  # Folder with results to be evaluated.
  #'results_path': config.results_path,

  # Folder for the calculated pose errors and performance scores.
  #'eval_path': config.eval_path,

  # File with a list of estimation targets to consider. The file is assumed to
  # be stored in the dataset folder.
  #'targets_filename': 'test_targets_bop19.json',
}
################################################################################

def eval_bop(eval_args, eval_dir, filename, errors):

  # Command line arguments.
  # ------------------------------------------------------------------------------
 
  p['results_path'] = os.path.join(eval_dir, 'bop')
  p['eval_path'] = p['results_path']
  p['result_filenames'] = [filename]
  p['renderer_type'] = eval_args.get('DATA', 'RENDERER_TYPE')
  p['targets_filename'] = eval_args.get('DATA', 'TARGETS_FILENAME')
  p['visib_gt_min'] = eval_args.getfloat('METRIC', 'VISIB_GT_MIN')
  
  p['errors'] = errors

  # Evaluation.
  # ------------------------------------------------------------------------------
  for result_filename in p['result_filenames']:

    misc.log('===========')
    misc.log('EVALUATING: {}'.format(result_filename))
    misc.log('===========')

    time_start = time.time()

    # Volume under recall surface (VSD) / area under recall curve (MSSD, MSPD).
    average_recalls = {}

    # Name of the result and the dataset.
    result_name = os.path.splitext(os.path.basename(result_filename))[0]
    dataset = str(result_name.split('_')[1].split('-')[0])

    # Calculate the average estimation time per image.
    ests = inout.load_bop_results(
      os.path.join(p['results_path'], result_filename), version='bop19')
    times = {}
    times_available = True
    for est in ests:
      result_key = '{:06d}_{:06d}'.format(est['scene_id'], est['im_id'])
      if est['time'] < 0:
        # All estimation times must be provided.
        times_available = False
        break
      elif result_key in times:
        if abs(times[result_key] - est['time']) > 0.001:
          raise ValueError(
            'The running time for scene {} and image {} is not the same for '
            'all estimates.'.format(est['scene_id'], est['im_id']))
      else:
        times[result_key] = est['time']

    if times_available:
      average_time_per_image = np.mean(list(times.values()))
    else:
      average_time_per_image = -1.0

    # Evaluate the pose estimates.
    for error in p['errors']:
     
      # Paths (rel. to p['eval_path']) to folders with calculated pose errors.
      # For VSD, there is one path for each setting of tau. For the other pose
      # error functions, there is only one path.
      error_dir_paths = {}
      if error['type'] == 'vsd':
        for vsd_tau in error['vsd_taus']:
          error_sign = misc.get_error_signature(
            error['type'], error['n_top'], vsd_delta=error['vsd_deltas'][dataset],
            vsd_tau=vsd_tau)
          error_dir_paths[error_sign] = os.path.join(result_name, error_sign)
      else:
        error_sign = misc.get_error_signature(error['type'], error['n_top'])
        error_dir_paths[error_sign] = os.path.join(result_name, error_sign)

      # Recall scores for all settings of the threshold of correctness (and also
      # of the misalignment tolerance tau in the case of VSD).
      recalls = []

      # Calculate performance scores.
      for error_sign, error_dir_path in error_dir_paths.items():
        for correct_th in error['correct_th']:
      
          # Path to file with calculated scores.
          score_sign = misc.get_score_signature([correct_th], p['visib_gt_min'])

          scores_filename = 'scores_{}.json'.format(score_sign)
          scores_path = os.path.join(
            p['eval_path'], result_name, error_sign, scores_filename)
          
          # Load the scores.
          misc.log('Loading calculated scores from: {}'.format(scores_path))
          scores = inout.load_json(scores_path)
          recalls.append(scores['recall'])

      average_recalls[error['type']] = np.mean(recalls)

      misc.log('Recall scores: {}'.format(' '.join(map(str, recalls))))
      misc.log('Average recall: {}'.format(average_recalls[error['type']]))

    time_total = time.time() - time_start
    misc.log('Evaluation of {} took {}s.'.format(result_filename, time_total))

    # Calculate the final scores.
    final_scores = {}
    for error in p['errors']:
      final_scores['bop19_average_recall_{}'.format(error['type'])] =\
        average_recalls[error['type']]

    # Final score for the given dataset.
    #final_scores['bop19_average_recall'] = np.mean([
    #  average_recalls['vsd'], average_recalls['mssd'], average_recalls['mspd']])
    final_scores['bop19_average_recall'] = np.mean([average_recalls['vsd']])

    # Average estimation time per image.
    final_scores['bop19_average_time_per_image'] = average_time_per_image

    # Save the final scores.
    final_scores_path = os.path.join(
      p['eval_path'], result_name, 'scores_bop19.json')
    inout.save_json(final_scores_path, final_scores)

    # Print the final scores.
    misc.log('FINAL SCORES:')
    for score_name, score_value in final_scores.items():
      misc.log('- {}: {}'.format(score_name, score_value))

  misc.log('Done.')
