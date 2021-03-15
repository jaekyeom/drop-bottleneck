# coding=utf-8
# Copyright 2019 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script that launches policy training with the right hyperparameters.

All specified runs are launched in parallel as subprocesses.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import stat
import re
import socket
import subprocess
import time

import argparse

import sys
sys.path.append(os.getcwd())

from episodic_curiosity import constants
import six
import tensorflow as tf


DMLAB_SCENARIOS = ['noreward', 'norewardnofire', 'sparse', 'verysparse',
                   'sparseplusdoors', 'dense1', 'dense2']
MUJOCO_ANT_SCENARIOS = ['ant_no_reward']


def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser = arg_parser()

parser.add_argument('--exp_parent_dir', type=str, default='exp')
parser.add_argument('--auto_exp_symlink_dir', type=str, default=None)

parser.add_argument('--base_seed', type=int, default=123)
parser.add_argument('--global_seed', type=int, default=None)
parser.add_argument('--patch_tf_determinism', type=int, default=0, choices=[0, 1])

parser.add_argument('--scenario', type=str, default='verysparse',
                    choices=DMLAB_SCENARIOS + MUJOCO_ANT_SCENARIOS,
                    help='Scenario to launch.')
parser.add_argument('--num_timesteps', type=int, default=20000000,
                    help='Number of training timesteps to run.')
parser.add_argument('--num_env', type=int, default=12,
                    help='Number of envs to run in parallel for training the '
                         'policy.')
parser.add_argument('--r_networks_path', type=str, default=None,
                    help='Only meaningful for the "ppo_plus_ec" method. Path to the '
                         'root dir for pre-trained r networks. If specified, '
                         'we train the policy using those pre-trained r networks. '
                         'If not specified, we first generate the R network '
                         'training data, train the R network and then train the '
                         'policy.')
parser.add_argument('--noise_type', type=str, default='',
                    choices=['', 'image_action', 'image', 'noise_action', 'noise'])
parser.add_argument('--noise_tv_num_images', type=int, default=30)

parser.add_argument('--optimize_env_reset', type=int, default=0, choices=[0, 1])

parser.add_argument('--lr', type=float, default=2.5e-4)
parser.add_argument('--ent_coef', type=float, default=None)

parser.add_argument('--rlb_ot', type=int, default=1, choices=[1])
parser.add_argument('--rlb_ot_lr', type=float, default=1e-4)
parser.add_argument('--rlb_ot_batch_size', type=int, default=512)
parser.add_argument('--rlb_ot_history_size', type=int, default=1800)
parser.add_argument('--rlb_ot_train_interval', type=int, default=-1)
parser.add_argument('--rlb_ot_exploration_min_step', type=int, default=12600)
parser.add_argument('--rlb_ot_num_epochs', type=int, default=2)
parser.add_argument('--rlb_ot_memory_capacity', type=int, default=2000)
parser.add_argument('--rlb_ot_memory_algo', type=str, default='fifo',
                    choices=['fifo', 'random'])
parser.add_argument('--rlb_ot_deterministic_z_for_ir', type=int, default=1, choices=[0, 1])
parser.add_argument('--rlb_ot_ir_clip_low', type=float, default=None)

parser.add_argument('--rlb_ir_weight', type=float, default=0.005)
parser.add_argument('--rlb_loss_weight', type=float, default=1.0)
parser.add_argument('--rlb_normalize_ir', type=int, default=2, choices=[0, 1, 2, 3])

parser.add_argument('--rlb_beta', type=float, default=0.001)
parser.add_argument('--rlb_prediction_type', type=str, default='deepinfomax',
                    choices=['deepinfomax'])
parser.add_argument('--rlb_z_dim', type=int, default=128)
parser.add_argument('--rlb_prediction_term_num_samples', type=int, default=50)
parser.add_argument('--rlb_int_rew_type', type=str, default='epimem_dim_latent_sp',
                    choices=[
                        'epimem_dim_latent', 'epimem_dim_latent_sp',
                    ])
parser.add_argument('--rlb_no_entropy_term', type=int, default=0, choices=[0, 1])
parser.add_argument('--rlb_featdrop_temperature', type=float, default=0.1)
parser.add_argument('--rlb_featdrop_drop_prob_p_init', type=float, nargs='+', default=[-2.0, 1.0])
parser.add_argument('--rlb_featdrop_model_hid_sizes', type=int, nargs='*', default=[])
parser.add_argument('--rlb_featdrop_entropy_num_bins', type=int, default=32)
def _parse_rlb_var_approx_model_sigma(s):
    if s in ['implicit', 'learned_scalar', 'learned_output']:
        return s
    return float(s)
parser.add_argument('--rlb_dim_discriminator_hid_sizes', type=int, nargs='*', default=[64, 32, 16])
parser.add_argument('--rlb_dim_train_ahead', type=int, default=8)
parser.add_argument('--rlb_dim_train_ahead_aggregate_all', type=int, default=1, choices=[0, 1])
parser.add_argument('--rlb_dim_train_ahead_no_noise', type=int, default=0, choices=[0, 1])
parser.add_argument('--rlb_dim_marginal_strategy', type=str, default='reshuffle_combinationwise',
                    choices=['half_batch', 'shuffle_combinationwise', 'reshuffle_combinationwise', 'shuffle_full', 'reshuffle_full'])
parser.add_argument('--rlb_dim_no_joint_training', type=int, default=0, choices=[0, 1])
parser.add_argument('--rlb_target_dynamics', type=str, default='latent_aggmean',
                    choices=['latent_aggmean'])
parser.add_argument('--rlb_num_z_variables', type=int, default=1)

parser.add_argument('--debug_tf_timeline', type=int, default=0, choices=[0, 1])

parser.add_argument('--checkpoint_path_for_debugging', type=str, default=None)

parser.add_argument('--slack_notify_target', type=str, default=None)

FLAGS = parser.parse_args()

if FLAGS.global_seed is not None:
  os.environ['PYTHONHASHSEED'] = str(FLAGS.global_seed)

PYTHON_BINARY = 'python'


def logged_check_call(command):
  """Logs the command and calls it."""
  print('=' * 70 + '\nLaunching:\n', ' '.join(command))
  subprocess.check_call(command)


def flatten_list(to_flatten):
  # pylint: disable=g-complex-comprehension
  return [item for sublist in to_flatten for item in sublist]


def quote_gin_value(v):
  if isinstance(v, six.string_types):
    return '"{}"'.format(v)
  return v


def assemble_command(base_command, params):
  """Builds a command line to launch training.

  Args:
    base_command: list(str), command prefix.
    params: dict of param -> value. Parameters prefixed by '_gin.' are
      considered gin parameters.

  Returns:
    List of strings, the components of the command line to run.
  """
  gin_params = {param_name: param_value
                for param_name, param_value in params.items()
                if param_name.startswith('_gin.')}
  params = {param_name: param_value
            for param_name, param_value in params.items()
            if not param_name.startswith('_gin.')}
  return (base_command +
          ['--{}={}'.format(param, v)
           for param, v in params.items()] +
          flatten_list([['--gin_bindings',
                         '{}={}'.format(gin_param[len('_gin.'):],
                                        quote_gin_value(v))]
                        for gin_param, v in gin_params.items()]))


def get_ppo_params(scenario):
  """Returns the param for the 'ppo' method."""
  if scenario == 'ant_no_reward':
    raise NotImplementedError
    return {
        'policy_architecture': 'mlp',
        '_gin.CuriosityEnvWrapper.scale_task_reward': 0.0,
        '_gin.RLBEnvWrapper.scale_task_reward': 0.0,
        '_gin.create_single_parkour_env.run_oracle_before_monitor': True,
        '_gin.OracleExplorationReward.reward_grid_size': 5,
        '_gin.OracleExplorationReward.cell_reward_normalizer': 25,
        '_gin.CuriosityEnvWrapper.exploration_reward': 'none',
        '_gin.RLBEnvWrapper.exploration_reward': 'none',
        '_gin.train.ent_coef': FLAGS.ent_coef or 8e-6,
        '_gin.train.learning_rate': 3e-4,
        '_gin.train.nsteps': 256,
        '_gin.train.nminibatches': 4,
        '_gin.train.noptepochs': 10,
        '_gin.AntWrapper.texture_mode': 'random_tiled',
    }

  if scenario == 'noreward' or scenario == 'norewardnofire':
    out_dict = {
        'action_set': '' if scenario == 'noreward' else 'nofire',
        '_gin.create_single_env.run_oracle_before_monitor': True,
        '_gin.CuriosityEnvWrapper.scale_task_reward': 0.0,
        '_gin.RLBEnvWrapper.scale_task_reward': 0.0,
        '_gin.CuriosityEnvWrapper.scale_surrogate_reward': 0,
        #'_gin.RLBEnvWrapper.scale_surrogate_reward': 0,
        '_gin.create_environments.scale_task_reward_for_eval': 0,
        '_gin.create_environments_with_rlb.scale_task_reward_for_eval': 0,
        '_gin.create_environments.scale_surrogate_reward_for_eval': 1,
        '_gin.create_environments_with_rlb.scale_surrogate_reward_for_eval': 1,
        '_gin.OracleExplorationReward.reward_grid_size': 30,
        '_gin.CuriosityEnvWrapper.exploration_reward': 'oracle',
        #'_gin.RLBEnvWrapper.exploration_reward': 'oracle',
        '_gin.train.ent_coef': 0.0010941138105771857,
        '_gin.train.learning_rate': 0.00019306977288832496,
    }
  else:
    out_dict = {
        '_gin.CuriosityEnvWrapper.exploration_reward': 'oracle',
        #'_gin.RLBEnvWrapper.exploration_reward': 'oracle',
        '_gin.CuriosityEnvWrapper.scale_task_reward': 1.0,
        '_gin.RLBEnvWrapper.scale_task_reward': 1.0,
        '_gin.CuriosityEnvWrapper.scale_surrogate_reward': 0.0,
        #'_gin.RLBEnvWrapper.scale_surrogate_reward': 0.0,
        '_gin.train.ent_coef': 0.0010941138105771857,
        '_gin.train.learning_rate': 0.00019306977288832496,
    }

  out_dict.update({
    '_gin.train.learning_rate': FLAGS.lr,
    '_gin.train.patch_tf_determinism': FLAGS.patch_tf_determinism,
    '_gin.train.global_seed': FLAGS.global_seed,
    '_gin.DMLabWrapper.noise_type': FLAGS.noise_type,
    '_gin.DMLabWrapper.tv_num_images': FLAGS.noise_tv_num_images,
    '_gin.DMLabWrapper.optimize_env_reset': FLAGS.optimize_env_reset,
    '_gin.create_environments.base_seed': FLAGS.base_seed,
    '_gin.create_environments_with_rlb.base_seed': FLAGS.base_seed,
  })
  if FLAGS.checkpoint_path_for_debugging is not None:
    out_dict.update({
      '_gin.train.checkpoint_path_for_debugging': FLAGS.checkpoint_path_for_debugging,
    })
  if FLAGS.ent_coef is not None:
    out_dict.update({
      '_gin.train.ent_coef': FLAGS.ent_coef,
    })
  return out_dict

def get_ppo_rlb_params(scenario):
  params = get_ppo_params(scenario)
  params.update({
    '_gin.RLBEnvWrapper.exploration_reward': 'rlb',
  })
  params.update({
    '_gin.get_rlb_args.' + n: getattr(FLAGS, n) for n in FLAGS.__dict__.keys() if n.startswith('rlb_') or n.startswith('debug_')
  })
  return params

def get_exp_name(scenario):
  exp_name = {
    'noreward': 'Dnr',
    'norewardnofire': 'Dnrnf',
    'sparse': 'Ds',
    'verysparse': 'Dvs',
    'sparseplusdoors': 'Dspd',
    'dense1': 'Dd1',
    'dense2': 'Dd2',

    'ant_no_reward': 'Anr',
  }[scenario]
  exp_name += {
    '': '',
    'image_action': 'IA{}'.format(FLAGS.noise_tv_num_images),
    'image': 'I{}'.format(FLAGS.noise_tv_num_images),
    'noise_action': 'NA',
    'noise': 'N',
  }[FLAGS.noise_type]
  if 'SLURM_JOB_ID' in os.environ:
    exp_name += '.s_{}'.format(os.environ['SLURM_JOB_ID'])
  if 'SLURM_PROCID' in os.environ:
    exp_name += '.{}'.format(os.environ['SLURM_PROCID'])
  exp_name_prefix = exp_name + '.'
  if 'SLURM_RESTART_COUNT' in os.environ:
    exp_name += '.restarted_{}'.format(os.environ['SLURM_RESTART_COUNT'])
  if 'GCP_INSTANCE_NAME' in os.environ:
    exp_name += '.{}'.format(os.environ['GCP_INSTANCE_NAME'])
    if 'GCP_EXP_ID' in os.environ:
      exp_name += '.{}'.format(os.environ['GCP_EXP_ID'])
  else:
    #exp_name += datetime.datetime.now().strftime(".%Y-%m-%d-%H-%M-%S")
    exp_name += '.{}'.format(int(time.time()))
    exp_name += '.{}'.format(socket.gethostname())

  if FLAGS.rlb_ot:
    exp_name += '__O_rlr{}_bs{}_his{}_exms{}_ne{}_mem{}{}_dz{}'.format(
        FLAGS.rlb_ot_lr,
        FLAGS.rlb_ot_batch_size,
        FLAGS.rlb_ot_history_size,
        FLAGS.rlb_ot_exploration_min_step,
        FLAGS.rlb_ot_num_epochs,
        {'fifo': 'FF', 'random': 'RN'}[FLAGS.rlb_ot_memory_algo],
        FLAGS.rlb_ot_memory_capacity,
        FLAGS.rlb_ot_deterministic_z_for_ir,
    )

  assert FLAGS.rlb_int_rew_type == 'epimem_dim_latent_sp'
  assert FLAGS.rlb_target_dynamics == 'latent_aggmean'

  exp_name += '_lr{}_iw{}_in{}_d{}_b{}_pr{}-{}'.format(
      FLAGS.lr,
      FLAGS.rlb_ir_weight,
      FLAGS.rlb_normalize_ir,
      FLAGS.rlb_z_dim,
      FLAGS.rlb_beta,
      {'deepinfomax': 'DIM'}[FLAGS.rlb_prediction_type],
      FLAGS.rlb_prediction_term_num_samples,
  )
  exp_name += '__FDO_tem{}_dpi{}_hid{}__enb{}'.format(
      FLAGS.rlb_featdrop_temperature,
      ':'.join([str(s) for s in FLAGS.rlb_featdrop_drop_prob_p_init]),
      '-'.join([str(s) for s in FLAGS.rlb_featdrop_model_hid_sizes]),
      FLAGS.rlb_featdrop_entropy_num_bins,
  )
  if FLAGS.rlb_prediction_type == 'deepinfomax':
    assert FLAGS.rlb_dim_train_ahead_aggregate_all
    assert not FLAGS.rlb_dim_train_ahead_no_noise
    assert not FLAGS.rlb_dim_no_joint_training
    exp_name += '__DIM_h{}_tra{}_m{}'.format(
        '-'.join([str(s) for s in FLAGS.rlb_dim_discriminator_hid_sizes]),
        FLAGS.rlb_dim_train_ahead,
        {'half_batch': 'HB', 'shuffle_combinationwise': 'SCW', 'reshuffle_combinationwise': 'RSCW', 'shuffle_full': 'SF', 'reshuffle_full': 'RSF'}[FLAGS.rlb_dim_marginal_strategy],
    )

  return exp_name


def get_ppo_plus_eco_params(scenario):
  assert False
  """Returns the param for the 'ppo_plus_eco' method."""
  assert scenario in DMLAB_SCENARIOS, (
      'Non-DMLab scenarios not supported as of today by PPO+ECO method')

  if scenario == 'noreward' or scenario == 'norewardnofire':
    return {
        'action_set': '' if scenario == 'noreward' else 'nofire',
        '_gin.create_single_env.run_oracle_before_monitor': True,
        '_gin.CuriosityEnvWrapper.scale_task_reward': 0.0,
        '_gin.RLBEnvWrapper.scale_task_reward': 0.0,
        '_gin.create_environments.scale_task_reward_for_eval': 0,
        '_gin.create_environments_with_rlb.scale_task_reward_for_eval': 0,
        '_gin.create_environments.scale_surrogate_reward_for_eval': 1,
        '_gin.create_environments_with_rlb.scale_surrogate_reward_for_eval': 1,
        '_gin.OracleExplorationReward.reward_grid_size': 30,
        'r_checkpoint': '',
        '_gin.CuriosityEnvWrapper.scale_surrogate_reward': 0.03017241379310345,
        '_gin.RLBEnvWrapper.scale_surrogate_reward': 0.03017241379310345,
        '_gin.train.ent_coef': 0.002053525026457146,
        '_gin.create_environments.online_r_training': True,
        '_gin.create_environments_with_rlb.online_r_training': True,
        '_gin.RNetworkTrainer.observation_history_size': 60000,
        '_gin.RNetworkTrainer.training_interval': -1,
        '_gin.CuriosityEnvWrapper.exploration_reward_min_step': 60000,
        '_gin.RLBEnvWrapper.exploration_reward_min_step': 60000,
        '_gin.RNetworkTrainer.num_epochs': 10,
    }
  else:
    return {
        'action_set': '',
        'r_checkpoint': '',
        '_gin.EpisodicMemory.capacity': 200,
        '_gin.similarity_to_memory.similarity_aggregation': 'percentile',
        '_gin.EpisodicMemory.replacement': 'random',
        '_gin.CuriosityEnvWrapper.scale_task_reward': 1.0,
        '_gin.RLBEnvWrapper.scale_task_reward': 1.0,
        '_gin.CuriosityEnvWrapper.scale_surrogate_reward': 0.03017241379310345,
        '_gin.RLBEnvWrapper.scale_surrogate_reward': 0.03017241379310345,
        '_gin.train.ent_coef': 0.002053525026457146,
        '_gin.create_environments.online_r_training': True,
        '_gin.create_environments_with_rlb.online_r_training': True,
        '_gin.RNetworkTrainer.observation_history_size': 60000,
        '_gin.RNetworkTrainer.training_interval': -1,
        '_gin.CuriosityEnvWrapper.exploration_reward_min_step': 60000,
        '_gin.RLBEnvWrapper.exploration_reward_min_step': 60000,
        '_gin.RNetworkTrainer.num_epochs': 10,
    }


def get_ppo_plus_grid_oracle_params(scenario):
  assert False
  """Returns the param for the 'ppo_plus_grid_oracle' method."""
  assert scenario in DMLAB_SCENARIOS, (
      'Non-DMLab scenarios not supported as of today by PPO+grid oracle method')
  if scenario == 'noreward' or scenario == 'norewardnofire':
    return {
        'action_set': '' if scenario == 'noreward' else 'nofire',
        '_gin.create_single_env.run_oracle_before_monitor': True,
        '_gin.OracleExplorationReward.reward_grid_size': 30,
        '_gin.CuriosityEnvWrapper.scale_task_reward': 0.0,
        '_gin.RLBEnvWrapper.scale_task_reward': 0.0,
        '_gin.CuriosityEnvWrapper.scale_surrogate_reward': 0.05246913580246913,
        '_gin.RLBEnvWrapper.scale_surrogate_reward': 0.05246913580246913,
        '_gin.create_environments.scale_task_reward_for_eval': 0,
        '_gin.create_environments_with_rlb.scale_task_reward_for_eval': 0,
        '_gin.create_environments.scale_surrogate_reward_for_eval': 1,
        '_gin.create_environments_with_rlb.scale_surrogate_reward_for_eval': 1,
        '_gin.CuriosityEnvWrapper.exploration_reward': 'oracle',
        '_gin.RLBEnvWrapper.exploration_reward': 'oracle',
        '_gin.train.ent_coef': 0.0066116902624148155,
    }
  else:
    return {
        'action_set': '',
        '_gin.CuriosityEnvWrapper.exploration_reward': 'oracle',
        '_gin.RLBEnvWrapper.exploration_reward': 'oracle',
        '_gin.CuriosityEnvWrapper.scale_task_reward': 1.0,
        '_gin.RLBEnvWrapper.scale_task_reward': 1.0,
        '_gin.CuriosityEnvWrapper.scale_surrogate_reward': 0.05246913580246913,
        '_gin.RLBEnvWrapper.scale_surrogate_reward': 0.05246913580246913,
        '_gin.train.ent_coef': 0.0066116902624148155,
        '_gin.OracleExplorationReward.reward_grid_size': 30,
    }


def get_ppo_plus_ec_params(scenario, r_network_path):
  assert False
  """Returns the param for the 'ppo_plus_ec' method."""
  if scenario == 'ant_no_reward':
    return {
        'policy_architecture': 'mlp',
        '_gin.CuriosityEnvWrapper.scale_task_reward': 0.0,
        '_gin.RLBEnvWrapper.scale_task_reward': 0.0,
        '_gin.create_single_parkour_env.run_oracle_before_monitor': True,
        '_gin.OracleExplorationReward.reward_grid_size': 5,
        '_gin.OracleExplorationReward.cell_reward_normalizer': 25,
        '_gin.CuriosityEnvWrapper.exploration_reward': 'episodic_curiosity',
        '_gin.RLBEnvWrapper.exploration_reward': 'episodic_curiosity',
        '_gin.EpisodicMemory.capacity': 1000,
        '_gin.EpisodicMemory.replacement': 'random',
        '_gin.similarity_to_memory.similarity_aggregation': 'nth_largest',
        '_gin.CuriosityEnvWrapper.similarity_threshold': 1.0,
        '_gin.RLBEnvWrapper.similarity_threshold': 1.0,
        '_gin.train.nsteps': 256,
        '_gin.train.nminibatches': 4,
        '_gin.train.noptepochs': 10,
        '_gin.CuriosityEnvWrapper.bonus_reward_additive_term': 0.5,
        '_gin.RLBEnvWrapper.bonus_reward_additive_term': 0.5,
        'r_checkpoint': r_network_path,
        '_gin.AntWrapper.texture_mode': 'random_tiled',
        '_gin.CuriosityEnvWrapper.scale_surrogate_reward': 1.0,
        '_gin.RLBEnvWrapper.scale_surrogate_reward': 1.0,
        '_gin.train.ent_coef': 2.23872113857e-05,
        '_gin.train.learning_rate': 7.49894209332e-05,
    }

  if scenario == 'noreward' or scenario == 'norewardnofire':
    return {
        'r_checkpoint': r_network_path,
        'action_set': '' if scenario == 'noreward' else 'nofire',
        '_gin.create_single_env.run_oracle_before_monitor': True,
        '_gin.CuriosityEnvWrapper.scale_task_reward': 0.0,
        '_gin.RLBEnvWrapper.scale_task_reward': 0.0,
        '_gin.create_environments.scale_task_reward_for_eval': 0,
        '_gin.create_environments_with_rlb.scale_task_reward_for_eval': 0,
        '_gin.create_environments.scale_surrogate_reward_for_eval': 1,
        '_gin.create_environments_with_rlb.scale_surrogate_reward_for_eval': 1,
        '_gin.OracleExplorationReward.reward_grid_size': 30,
        '_gin.CuriosityEnvWrapper.scale_surrogate_reward': 0.03017241379310345,
        '_gin.RLBEnvWrapper.scale_surrogate_reward': 0.03017241379310345,
        '_gin.train.ent_coef': 0.002053525026457146,
    }
  else:
    return {
        'r_checkpoint': r_network_path,
        'action_set': '',
        '_gin.EpisodicMemory.capacity': 200,
        '_gin.similarity_to_memory.similarity_aggregation': 'percentile',
        '_gin.EpisodicMemory.replacement': 'random',
        '_gin.CuriosityEnvWrapper.scale_task_reward': 1.0,
        '_gin.RLBEnvWrapper.scale_task_reward': 1.0,
        '_gin.CuriosityEnvWrapper.scale_surrogate_reward': 0.03017241379310345,
        '_gin.RLBEnvWrapper.scale_surrogate_reward': 0.03017241379310345,
        '_gin.train.ent_coef': 0.002053525026457146,
    }


def run_training():
  """Runs training accordding to flags."""

  policy_training_params = get_ppo_rlb_params(FLAGS.scenario)


  #assert bool(FLAGS.rlb_ot) == (FLAGS.rlb_int_rew_type in ['epimem_dim_latent', 'epimem_dim_latent_sp'])
  assert bool(FLAGS.rlb_ot) == FLAGS.rlb_int_rew_type.startswith('epimem_dim_latent_sp')

  if 'image' in FLAGS.noise_type:
    for image in range(1, FLAGS.noise_tv_num_images + 1):
      image_path = 'tv_images/%d.jpeg' % image
      if not os.path.exists(image_path):
        print('''ERROR: Image {} is missing. To run with the same "Image Action" noise setting, the original images used for "Episodic Curiosity Through Reachability" (https://github.com/google-research/episodic-curiosity) are needed.'''.format(image_path))
        return

  exp_name = get_exp_name(FLAGS.scenario)
  workdir = os.path.join('exp', exp_name)

  if FLAGS.scenario in DMLAB_SCENARIOS:
    env_name = ('dmlab:' + constants.Const.find_level_by_scenario(
        FLAGS.scenario).fully_qualified_name)
  else:
    assert FLAGS.scenario in MUJOCO_ANT_SCENARIOS, FLAGS.scenario
    env_name = 'parkour:'

  slurm_identifier = ''
  if 'SLURM_JOB_ID' in os.environ:
    slurm_identifier += os.environ['SLURM_JOB_ID']
  if 'SLURM_PROCID' in os.environ:
    slurm_identifier += '.' + os.environ['SLURM_PROCID']

  policy_training_params.update({
      'workdir': workdir,
      'num_env': str(FLAGS.num_env),
      'env_name': env_name,
      'num_timesteps': str(FLAGS.num_timesteps)})
  print('Params for scenario', FLAGS.scenario, ':\n', policy_training_params)
  if os.path.exists(workdir):
    print('ERROR: workdir already exists. {}'.format(workdir))
    return
  tf.gfile.MakeDirs(workdir)
  base_command = [PYTHON_BINARY, 'rlb/train_policy_rlb.py']
  if FLAGS.slack_notify_target is not None:
    import requests
    requests.post(
        'https://hooks.slack.com/services/{}'.format(FLAGS.slack_notify_target),
        json={'text': 'STARTING {}\n```{}```\n```{}```\noriginal command: ```{}```'.format(slurm_identifier, exp_name, os.path.abspath(workdir), sys.argv)})
  try:
    logged_check_call(assemble_command(
        base_command, policy_training_params))
  except subprocess.CalledProcessError as e:
    if FLAGS.slack_notify_target is not None:
      import requests
      requests.post(
          'https://hooks.slack.com/services/{}'.format(FLAGS.slack_notify_target),
          json={'text': 'FAILED {} retcode {}\n```{}```\n```{}```\noriginal command: ```{}```'.format(slurm_identifier, e.returncode, exp_name, os.path.abspath(workdir), sys.argv)})
  else:
    if FLAGS.slack_notify_target is not None:
      import requests
      requests.post(
          'https://hooks.slack.com/services/{}'.format(FLAGS.slack_notify_target),
          json={'text': 'FINISHED {}\n```{}```\n```{}```\n'.format(slurm_identifier, exp_name, os.path.abspath(workdir))})


def main():
  run_training()
  return 0

if __name__ == '__main__':
  import sys
  sys.exit(main())

