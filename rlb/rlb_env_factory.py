# coding=utf-8

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function
import functools
import os
from absl import flags
from episodic_curiosity import constants
from episodic_curiosity import curiosity_env_wrapper
from episodic_curiosity import episodic_memory
from episodic_curiosity import r_network
from episodic_curiosity import r_network_training
from episodic_curiosity.constants import Const
from episodic_curiosity.environments import dmlab_utils
from rlb.rlb_env_wrapper import RLBEnvWrapper
from rlb.rlb_episodic_memory import RLBEpisodicMemory
from rlb.rlb_training import RLBTrainer
from rlb.rlb_model_wrapper import RLBModelWrapper
from third_party.baselines import logger
from third_party.baselines.bench import Monitor
from third_party.baselines.common import atari_wrappers
from third_party.baselines.common.vec_env import subproc_vec_env
from third_party.baselines.common.vec_env import threaded_vec_env
from third_party.keras_resnet import models
import gin


from episodic_curiosity.env_factory import *

del create_environments

@gin.configurable
def create_environments_with_rlb(env_name,
                                 num_envs,
                                 dmlab_homepath = '',
                                 action_set = '',
                                 base_seed = 123,
                                 scale_task_reward_for_eval = 1.0,
                                 scale_surrogate_reward_for_eval = 0.0,
                                 online_r_training = False,
                                 environment_engine = 'dmlab',
                                 r_network_weights_store_path = '',
                                 level_cache_mode=False,
                                 rlb_image_size=(84, 84)):
  """Creates a environments with R-network-based curiosity reward.

  Args:
    env_name: Name of the DMLab environment.
    num_envs: Number of parallel environment to spawn.
    r_network_weights_path: Path to the weights of the R-network.
    dmlab_homepath: Path to the DMLab MPM. Required when running on borg.
    action_set: One of {'small', 'nofire', ''}. Which action set to use.
    base_seed: Each environment will use base_seed+env_index as seed.
    scale_task_reward_for_eval: scale of the task reward to be used for
      valid/test environments.
    scale_surrogate_reward_for_eval: scale of the surrogate reward to be used
      for valid/test environments.
    online_r_training: Whether to enable online training of the R-network.
    environment_engine: either 'dmlab', 'atari', 'parkour'.
    r_network_weights_store_path: Directory where to store R checkpoints
      generated during online training of the R network.

  Returns:
    Wrapped environment with curiosity.
  """
  # Environments without intrinsic exploration rewards.
  # pylint: disable=g-long-lambda
  create_dmlab_single_env = functools.partial(create_single_env,
                                              dmlab_homepath=dmlab_homepath,
                                              action_set=action_set,
                                              level_cache_mode=level_cache_mode)

  if environment_engine == 'dmlab':
    create_env_fn = create_dmlab_single_env
    is_atari_environment = False
  elif environment_engine == 'atari':
    create_env_fn = create_single_atari_env
    is_atari_environment = True
  elif environment_engine == 'parkour':
    mujoco_key_path = ''
    create_env_fn = functools.partial(
        create_single_parkour_env, mujoco_key_path=mujoco_key_path)
    is_atari_environment = False
  else:
    raise ValueError('Unknown env engine {}'.format(environment_engine))

  VecEnvClass = (subproc_vec_env.SubprocVecEnv
                 if FLAGS.vec_env_class == 'SubprocVecEnv'
                 else threaded_vec_env.ThreadedVecEnv)

  with logger.ProfileKV('create_envs'):
    vec_env = VecEnvClass([
        (lambda _i=i: create_env_fn(env_name, base_seed + _i, use_monitor=True,
                                    split='train'))
        for i in range(num_envs)
    ], level_cache_mode=level_cache_mode)
    valid_env = VecEnvClass([
        (lambda _i=i: create_env_fn(env_name, base_seed + _i, use_monitor=False,
                                    split='valid'))
        for i in range(num_envs)
    ], level_cache_mode=level_cache_mode)
    test_env = VecEnvClass([
        (lambda _i=i: create_env_fn(env_name, base_seed + _i, use_monitor=False,
                                    split='test'))
        for i in range(num_envs)
    ], level_cache_mode=level_cache_mode)
  if level_cache_mode:
    logger.info('Starting the infinite map generation sequence...')
    import time
    while True:
      time.sleep(10)

  # pylint: enable=g-long-lambda

  rlb_image_shape = (84, 84, (4 if is_atari_environment else 3))

  rlb_model_wrapper = RLBModelWrapper(
      input_shape=rlb_image_shape,
      action_space=vec_env.action_space,
      max_grad_norm=0.5)

  rlb_model_trainer = RLBTrainer(
      rlb_model_wrapper,
      ensure_train_between_episodes=True)
      
  embedding_size = rlb_model_wrapper.rlb_all_z_dim
  vec_episodic_memory = [
      RLBEpisodicMemory(
          observation_shape=[embedding_size],
          replacement=rlb_model_wrapper.all_rlb_args.outer_args['rlb_ot_memory_algo'],
          capacity=rlb_model_wrapper.all_rlb_args.outer_args['rlb_ot_memory_capacity'])
      for _ in range(num_envs)
  ]

  exploration_reward_min_step = rlb_model_wrapper.all_rlb_args.outer_args['rlb_ot_exploration_min_step']
  if exploration_reward_min_step < 0:
    exploration_reward_min_step = rlb_model_trainer.training_interval

  env_wrapper = RLBEnvWrapper(
      vec_env=vec_env,
      vec_episodic_memory=vec_episodic_memory,
      observation_embedding_fn=rlb_model_wrapper.embed_observation,
      intrinsic_reward_fn=rlb_model_wrapper.compute_intrinsic_rewards,
      rlb_image_shape=rlb_image_shape,
      #target_image_shape=None,
      target_image_shape=[84, 84, 4 if is_atari_environment else 3],
      exploration_reward='rlb',
      scale_surrogate_reward=rlb_model_wrapper.all_rlb_args.outer_args['rlb_ir_weight'],
      ir_normalize_type=rlb_model_wrapper.all_rlb_args.outer_args['rlb_normalize_ir'],
      ir_clip_low=rlb_model_wrapper.all_rlb_args.outer_args['rlb_ot_ir_clip_low'],
      exploration_reward_min_step=exploration_reward_min_step,
      name='train')
  if rlb_model_trainer is not None:
    env_wrapper.add_observer(rlb_model_trainer)


  valid_env_wrapper, test_env_wrapper = (
      RLBEnvWrapper(
          vec_env=env,
          vec_episodic_memory=None,
          observation_embedding_fn=None,
          intrinsic_reward_fn=None,
          rlb_image_shape=None,
          target_image_shape=[84, 84, 4 if is_atari_environment else 3],
          exploration_reward=('none' if (is_atari_environment or
                                         environment_engine == 'parkour')
                              else 'oracle'),
          scale_task_reward=scale_task_reward_for_eval,
          scale_surrogate_reward=scale_surrogate_reward_for_eval,
          name=name)
      for env, name in [(valid_env, 'valid'), (test_env, 'test')])

  return env_wrapper, valid_env_wrapper, test_env_wrapper
