# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import deque
import os.path as osp
import time
import dill
from third_party.baselines import logger
from third_party.baselines.common import explained_variance
from third_party.baselines.common import tf_util
from third_party.baselines.common.input import observation_input
from third_party.baselines.common.runners import AbstractEnvRunner
from third_party.baselines.ppo2 import pathak_utils
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.python.profiler import option_builder
from rlb.rlb_model import define_rlb_model, construct_ph_set, get_rlb_args
from rlb.utils import RewardForwardFilter, RunningMeanStd


class Model(object):

  def __init__(self, policy, ob_space, ac_space, nbatch_act, nbatch_train,
               nsteps, ent_coef, vf_coef, max_grad_norm, use_curiosity,
               curiosity_strength, forward_inverse_ratio,
               curiosity_loss_strength, random_state_predictor, use_rlb):
    sess = tf.get_default_session()

    nenvs = nbatch_act
    act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, reuse=False)
    train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps,
                         reuse=True)

    assert not (use_curiosity and use_rlb)

    if use_curiosity:
      hidden_layer_size = 256
      self.state_encoder_net = tf.make_template(
          'state_encoder_net', pathak_utils.universeHead,
          create_scope_now_=True,
          trainable=(not random_state_predictor))
      self.icm_forward_net = tf.make_template(
          'icm_forward', pathak_utils.icm_forward_model,
          create_scope_now_=True, num_actions=ac_space.n,
          hidden_layer_size=hidden_layer_size)
      self.icm_inverse_net = tf.make_template(
          'icm_inverse', pathak_utils.icm_inverse_model,
          create_scope_now_=True, num_actions=ac_space.n,
          hidden_layer_size=hidden_layer_size)
    else:
      self.state_encoder_net = None
      self.icm_forward_net = None
      self.icm_inverse_net = None

    A = train_model.pdtype.sample_placeholder([None])
    ADV = tf.placeholder(tf.float32, [None])
    R = tf.placeholder(tf.float32, [None])
    OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
    OLDVPRED = tf.placeholder(tf.float32, [None])
    LR = tf.placeholder(tf.float32, [])
    CLIPRANGE = tf.placeholder(tf.float32, [])
    # When computing intrinsic reward a different batch size is used (number
    # of parallel environments), thus we need to define separate
    # placeholders for them.
    X_NEXT, _ = observation_input(ob_space, nbatch_train)
    X_INTRINSIC_NEXT, _ = observation_input(ob_space, nbatch_act)
    X_INTRINSIC_CURRENT, _ = observation_input(ob_space, nbatch_act)

    trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)

    self.all_rlb_args = get_rlb_args()
    if use_rlb:
      rlb_scope = 'rlb_model'
      #rlb_ir_weight = self.all_rlb_args.outer_args['rlb_ir_weight']
      rlb_loss_weight = self.all_rlb_args.outer_args['rlb_loss_weight']
      self.rlb_model = tf.make_template(
          rlb_scope, define_rlb_model,
          create_scope_now_=True,
          pdtype=train_model.pdtype,
          ac_space=ac_space,
          #nenvs=nenvs,
          optimizer=trainer,
          outer_scope=rlb_scope,
          **self.all_rlb_args.inner_args)
    else:
      self.rlb_model = None

    neglogpac = train_model.pd.neglogp(A)
    entropy = tf.reduce_mean(train_model.pd.entropy())

    vpred = train_model.vf
    vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED,
                                               - CLIPRANGE, CLIPRANGE)
    vf_losses1 = tf.square(vpred - R)
    vf_losses2 = tf.square(vpredclipped - R)
    vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
    ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
    pg_losses = -ADV * ratio
    pg_losses2 = -ADV * tf.clip_by_value(ratio,
                                         1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
    pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
    approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
    clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0),
                                                     CLIPRANGE)))
    curiosity_loss = self.compute_curiosity_loss(
        use_curiosity, train_model.X, A, X_NEXT,
        forward_inverse_ratio=forward_inverse_ratio,
        curiosity_loss_strength=curiosity_loss_strength)
    loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef + curiosity_loss

    if use_curiosity:
      encoded_time_step = self.state_encoder_net(X_INTRINSIC_CURRENT)
      encoded_next_time_step = self.state_encoder_net(X_INTRINSIC_NEXT)
      intrinsic_reward = self.curiosity_forward_model_loss(
          encoded_time_step, A, encoded_next_time_step)
      intrinsic_reward = intrinsic_reward * curiosity_strength

    if self.rlb_model:
      assert 'intrinsic_reward' not in locals()
      intrinsic_reward = self.rlb_model(ph_set=construct_ph_set(
          x=X_INTRINSIC_CURRENT,
          x_next=X_INTRINSIC_NEXT,
          a=A)).int_rew
      #intrinsic_reward = intrinsic_reward * rlb_ir_weight

      rlb_out = self.rlb_model(ph_set=construct_ph_set(
          x=train_model.X,
          x_next=X_NEXT,
          a=A))
      loss = loss + rlb_loss_weight * rlb_out.aux_loss

    #with tf.variable_scope('model'):
    params = tf.trainable_variables()
    logger.info('{} trainable parameters: {}'.format(len(params), [p.name for p in params]))
    # For whatever reason Pathak multiplies the loss by 20.
    pathak_multiplier = 20 if use_curiosity else 1
    grads = tf.gradients(loss * pathak_multiplier, params)
    if max_grad_norm is not None:
      grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
    grads = list(zip(grads, params))
    #trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
    _train = trainer.apply_gradients(grads)

    if self.all_rlb_args.debug_args['debug_tf_timeline']:
      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      builder = option_builder.ProfileOptionBuilder
      profiler_opts = builder(builder.time_and_memory()).order_by('micros').build()
    else:
      run_options = None


    def getIntrinsicReward(curr, next_obs, actions):
      with logger.ProfileKV('get_intrinsic_reward'):
        return sess.run(intrinsic_reward, {X_INTRINSIC_CURRENT: curr,
                                           X_INTRINSIC_NEXT: next_obs,
                                           A: actions})
    def train(lr, cliprange, obs, next_obs, returns, masks, actions, values,
              neglogpacs, states=None, gather_histo=False, gather_sc=False, debug_timeliner=None):
      advs = returns - values
      advs = (advs - advs.mean()) / (advs.std() + 1e-8)
      td_map = {train_model.X: obs, A: actions, ADV: advs, R: returns, LR: lr,
                CLIPRANGE: cliprange, OLDNEGLOGPAC: neglogpacs,
                OLDVPRED: values, X_NEXT: next_obs}
      if states is not None:
        td_map[train_model.S] = states
        td_map[train_model.M] = masks
      fetches = {
        'train': _train,
        'losses': [pg_loss, vf_loss, entropy, approxkl, clipfrac, curiosity_loss],
      }
      if self.rlb_model:
        fetches['losses'].append(rlb_out.aux_loss)
      if gather_histo:
        fetches.update({ 'stats_histo': {} })
        if self.rlb_model:
          fetches['stats_histo'].update({ n: getattr(rlb_out.stats_histo, n) for n in self.stats_histo_names })
      if gather_sc:
        fetches.update({ 'stats_sc': {} })
        if self.rlb_model:
          fetches['stats_sc'].update({ n: getattr(rlb_out.stats_sc, n) for n in self.stats_sc_names })
      if debug_timeliner is not None and self.all_rlb_args.debug_args['debug_tf_timeline']:
        run_metadata = tf.RunMetadata()
        final_run_options = run_options
      else:
        run_metadata = None
        final_run_options = None
      with logger.ProfileKV('train_sess_run'):
        result = sess.run(
            fetches,
            td_map,
            options=final_run_options,
            run_metadata=run_metadata,
        )
      if debug_timeliner is not None and self.all_rlb_args.debug_args['debug_tf_timeline']:
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format(show_memory=True)
        debug_timeliner.update_timeline(chrome_trace)
        tf.profiler.profile(tf.get_default_graph(), run_meta=run_metadata, cmd='scope', options=profiler_opts)
      return result
    self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy',
                       'approxkl', 'clipfrac', 'curiosity_loss']
    if self.rlb_model:
      self.loss_names.append('rlb_loss')
      self.stats_histo_names = sorted(list(rlb_out.stats_histo.__dict__.keys()))
      self.stats_sc_names = sorted(list(rlb_out.stats_sc.__dict__.keys()))
    else:
      self.stats_histo_names = []
      self.stats_sc_names = []

    def save(save_path):
      ps = sess.run(params)
      with tf.gfile.Open(save_path, 'wb') as fh:
        fh.write(dill.dumps(ps))

    def load(load_path):
      with tf.gfile.Open(load_path, 'rb') as fh:
        val = fh.read()
        loaded_params = dill.loads(val)
      restores = []
      for p, loaded_p in zip(params, loaded_params):
        restores.append(p.assign(loaded_p))
      sess.run(restores)
      # If you want to load weights, also save/load observation scaling inside
      # VecNormalize

    self.getIntrinsicReward = getIntrinsicReward
    self.train = train
    self.train_model = train_model
    self.act_model = act_model
    self.step = act_model.step
    self.value = act_model.value
    self.initial_state = act_model.initial_state
    self.save = save
    self.load = load
    tf.global_variables_initializer().run(session=sess)  # pylint: disable=E1101

  def curiosity_forward_model_loss(self, encoded_state, action,
                                   encoded_next_state):
    pred_next_state = self.icm_forward_net(encoded_state, action)
    forward_loss = 0.5 * tf.reduce_mean(
        tf.squared_difference(pred_next_state, encoded_next_state), axis=1)
    forward_loss = forward_loss * 288.0
    return forward_loss

  def curiosity_inverse_model_loss(self, encoded_states, actions,
                                   encoded_next_states):
    pred_action_logits = self.icm_inverse_net(encoded_states,
                                              encoded_next_states)
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=pred_action_logits, labels=actions), name='invloss')

  def compute_curiosity_loss(self, use_curiosity, time_steps, actions,
                             next_time_steps, forward_inverse_ratio,
                             curiosity_loss_strength):
    if use_curiosity:
      with tf.name_scope('curiosity_loss'):
        encoded_time_steps = self.state_encoder_net(time_steps)
        encoded_next_time_steps = self.state_encoder_net(next_time_steps)

        inverse_loss = self.curiosity_inverse_model_loss(
            encoded_time_steps, actions, encoded_next_time_steps)
        forward_loss = self.curiosity_forward_model_loss(
            encoded_time_steps, actions, encoded_next_time_steps)
        forward_loss = tf.reduce_mean(forward_loss)

        total_curiosity_loss = curiosity_loss_strength * (
            forward_inverse_ratio * forward_loss +
            (1 - forward_inverse_ratio) * inverse_loss)
    else:
      total_curiosity_loss = tf.constant(0.0, dtype=tf.float32,
                                         name='curiosity_loss')

    return total_curiosity_loss


class Runner(AbstractEnvRunner):

  def __init__(self, env, model, nsteps, gamma, lam, eval_callback=None):
    super(Runner, self).__init__(env=env, model=model, nsteps=nsteps, obs_dtype=np.uint8)
    self.lam = lam
    self.gamma = gamma

    self._eval_callback = eval_callback
    self._collection_iteration = 0

    if self.model.rlb_model and self.model.all_rlb_args.outer_args['rlb_normalize_ir']:
      self.irff = RewardForwardFilter(self.gamma)
      self.irff_rms = RunningMeanStd()

  def run(self):
    if self._eval_callback:
      global_step = (self._collection_iteration *
                     self.env.num_envs *
                     self.nsteps)
      with logger.ProfileKV('eval_callback'):
        self._eval_callback(self.model.step, global_step)

    self._collection_iteration += 1

    mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
    mb_neglogpacs, mb_next_obs = [], []
    mb_rewards_int, mb_rewards_int_raw = [], []
    mb_rewards_ext = []
    mb_states = self.states
    info_whitelist = ['maze_layout', 'position', 'initial_position', 'position_history', 'reward_history', 'last_ep_maze_layout', 'bonus_reward_raw', 'bonus_reward', 'bonus_reward_raw_history', 'bonus_reward_history']
    mb_selected_infos = []
    epinfos = []
    for _ in range(self.nsteps):
      actions, values, self.states, neglogpacs = self.model.step(self.obs,
                                                                 self.states,
                                                                 self.dones)
      mb_obs.append(self.obs.copy())
      mb_actions.append(actions)
      mb_values.append(values)
      mb_neglogpacs.append(neglogpacs)
      mb_dones.append(self.dones)
      with logger.ProfileKV('train_env_step'):
        self.obs[:], rewards, self.dones, infos = self.env.step(actions)
      assert self.obs.dtype == np.uint8
      mb_next_obs.append(self.obs.copy())

      if self.model.state_encoder_net or self.model.rlb_model:
        mb_rewards_ext.append(rewards)
      else:
        mb_rewards_ext.append([info['task_reward'] for info in infos])

      if self.model.state_encoder_net:
        # {{{
        intrinsic_reward_raw = self.model.getIntrinsicReward(
            mb_obs[-1], mb_next_obs[-1], actions)
        # Clip to [-1, 1] range intrinsic reward.
        intrinsic_reward = [
            max(min(x, 1.0), -1.0) for x in intrinsic_reward_raw]
        #rewards += intrinsic_reward
        rewards = rewards + intrinsic_reward
        # }}}

      elif self.model.rlb_model:
        intrinsic_reward_raw = self.model.getIntrinsicReward(
            mb_obs[-1], mb_next_obs[-1], actions)
        ## Clip to [-1, 1] range intrinsic reward.
        #intrinsic_reward = [
        #    max(min(x, 1.0), -1.0) for x in intrinsic_reward]

        if self.model.all_rlb_args.outer_args['rlb_normalize_ir']:
          irffs = self.irff.update(intrinsic_reward_raw)
          self.irff_rms.update(irffs.ravel())
          intrinsic_reward = intrinsic_reward_raw / np.sqrt(self.irff_rms.var)
        else:
          intrinsic_reward = intrinsic_reward_raw

        #rewards += intrinsic_reward
        rewards = rewards + intrinsic_reward * self.model.all_rlb_args.outer_args['rlb_ir_weight']

      else:
        try:
          intrinsic_reward_raw = [info['bonus_reward_raw'] for info in infos]
          intrinsic_reward = [info['bonus_reward'] for info in infos]
        except KeyError:
          intrinsic_reward_raw = np.zeros_like(rewards, dtype=np.float32)
          intrinsic_reward = intrinsic_reward_raw

      mb_rewards_int_raw.append(intrinsic_reward_raw)
      mb_rewards_int.append(intrinsic_reward)

      mb_selected_infos.append([])
      for info in infos:
        maybeepinfo = info.get('episode')
        if maybeepinfo: epinfos.append(maybeepinfo)
        mb_selected_infos[-1].append({})
        for k in info_whitelist:
          if k in info:
            mb_selected_infos[-1][-1][k] = info[k]
      mb_rewards.append(rewards)
    # batch of steps to batch of rollouts
    mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
    mb_next_obs = np.asarray(mb_next_obs, dtype=self.obs.dtype)
    mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
    mb_rewards_ext = np.asarray(mb_rewards_ext, dtype=np.float32)
    mb_rewards_int = np.asarray(mb_rewards_int, dtype=np.float32)
    mb_rewards_int_raw = np.asarray(mb_rewards_int_raw, dtype=np.float32)
    mb_actions = np.asarray(mb_actions)
    mb_values = np.asarray(mb_values, dtype=np.float32)
    mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
    mb_dones = np.asarray(mb_dones, dtype=np.bool)
    last_values = self.model.value(self.obs, self.states, self.dones)
    # discount/bootstrap off value fn
    mb_returns = np.zeros_like(mb_rewards)
    mb_advs = np.zeros_like(mb_rewards)
    lastgaelam = 0
    for t in reversed(range(self.nsteps)):
      if t == self.nsteps - 1:
        nextnonterminal = 1.0 - self.dones
        nextvalues = last_values
      else:
        nextnonterminal = 1.0 - mb_dones[t+1]
        nextvalues = mb_values[t+1]
      delta = (mb_rewards[t] + self.gamma * nextvalues * nextnonterminal -
               mb_values[t])
      mb_advs[t] = lastgaelam = (delta + self.gamma * self.lam *
                                 nextnonterminal * lastgaelam)
    mb_returns = mb_advs + mb_values
    mb_selected_infos = np.asarray(mb_selected_infos, dtype=np.object)
    assert mb_selected_infos.shape[0] == mb_obs.shape[0]
    assert mb_selected_infos.shape[1] == mb_obs.shape[1]
    return (map(sf01, (mb_obs, mb_next_obs, mb_returns, mb_dones,
                       mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos, (map(sf01, (mb_rewards, mb_rewards_ext, mb_rewards_int, mb_rewards_int_raw, mb_selected_infos, mb_dones))))


def sf01(arr):
  """Swap and then flatten axes 0 and 1.
  """
  s = arr.shape
  return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def constfn(val):
  def f(_):
    return val
  return f


def learn(policy, env, nsteps, total_timesteps, ent_coef, lr,
          vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95,
          log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
          save_interval=0, load_path=None, train_callback=None,
          eval_callback=None, cloud_sync_callback=None, cloud_sync_interval=1000,
          workdir='', use_curiosity=False, curiosity_strength=0.01,
          forward_inverse_ratio=0.2, curiosity_loss_strength=10,
          random_state_predictor=False, use_rlb=False,
          checkpoint_path_for_debugging=None):
  if isinstance(lr, float):
    lr = constfn(lr)
  else:
    assert callable(lr)
  if isinstance(cliprange, float):
    cliprange = constfn(cliprange)
  else:
    assert callable(cliprange)
  total_timesteps = int(total_timesteps)

  nenvs = env.num_envs
  ob_space = env.observation_space
  ac_space = env.action_space
  nbatch = nenvs * nsteps
  nbatch_train = nbatch // nminibatches

  # pylint: disable=g-long-lambda
  make_model = lambda: Model(policy=policy, ob_space=ob_space,
                             ac_space=ac_space, nbatch_act=nenvs,
                             nbatch_train=nbatch_train, nsteps=nsteps,
                             ent_coef=ent_coef, vf_coef=vf_coef,
                             max_grad_norm=max_grad_norm,
                             use_curiosity=use_curiosity,
                             curiosity_strength=curiosity_strength,
                             forward_inverse_ratio=forward_inverse_ratio,
                             curiosity_loss_strength=curiosity_loss_strength,
                             random_state_predictor=random_state_predictor,
                             use_rlb=use_rlb)
  # pylint: enable=g-long-lambda
  if save_interval and workdir:
    with tf.gfile.Open(osp.join(workdir, 'make_model.pkl'), 'wb') as fh:
      fh.write(dill.dumps(make_model))
    saver = tf.train.Saver(max_to_keep=10000000)
    def save_state(fname):
      if not osp.exists(osp.dirname(fname)):
        os.makedirs(osp.dirname(fname))
      saver.save(tf.get_default_session(), fname)
  with tf.device('/gpu:0'):
    model = make_model()
  if load_path is not None:
    model.load(load_path)
  runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam,
                  eval_callback=eval_callback)

  if checkpoint_path_for_debugging is not None:
    tf_util.load_state(checkpoint_path_for_debugging, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='rlb_model'))

  epinfobuf = deque(maxlen=100)
  tfirststart = time.time()

  nupdates = total_timesteps//nbatch
  for update in range(1, nupdates+1):
    assert nbatch % nminibatches == 0
    nbatch_train = nbatch // nminibatches
    tstart = time.time()
    frac = 1.0 - (update - 1.0) / nupdates
    lrnow = lr(frac)
    cliprangenow = cliprange(frac)
    (obs, next_obs, returns, masks, actions, values,
     neglogpacs), states, epinfos, (rewards, rewards_ext, rewards_int, rewards_int_raw, selected_infos, dones) = runner.run()
    epinfobuf.extend(epinfos)
    mblossvals = []
    mbhistos = []
    mbscs = []

    #if model.all_rlb_args.debug_args['debug_tf_timeline'] and update % 5 == 0:
    if model.all_rlb_args.debug_args['debug_tf_timeline'] and update % 1 == 0:
      debug_timeliner = logger.TimeLiner()
    else:
      debug_timeliner = None

    if states is None:  # nonrecurrent version
      inds = np.arange(nbatch)
      for oe in range(noptepochs):
        gather_histo = (oe == noptepochs - 1)
        np.random.shuffle(inds)
        for start in range(0, nbatch, nbatch_train):
          gather_sc = ((oe == noptepochs - 1) and (start + nbatch_train >= nbatch))
          end = start + nbatch_train
          mbinds = inds[start:end]
          slices = [arr[mbinds] for arr in (obs, returns, masks, actions,
                                            values, neglogpacs, next_obs)]
          with logger.ProfileKV('train'):
            fetches = model.train(lrnow, cliprangenow, slices[0],
                                  slices[6], slices[1], slices[2],
                                  slices[3], slices[4], slices[5],
                                  gather_histo=gather_histo, gather_sc=gather_sc,
                                  debug_timeliner=debug_timeliner)
          mblossvals.append(fetches['losses'])
          if gather_histo:
            mbhistos.append(fetches['stats_histo'])
          if gather_sc:
            mbscs.append(fetches['stats_sc'])
    else:  # recurrent version
      assert nenvs % nminibatches == 0
      envsperbatch = nenvs // nminibatches
      envinds = np.arange(nenvs)
      flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
      envsperbatch = nbatch_train // nsteps
      for oe in range(noptepochs):
        gather_histo = (oe == noptepochs - 1)
        np.random.shuffle(envinds)
        for start in range(0, nenvs, envsperbatch):
          gather_sc = ((oe == noptepochs - 1) and (start + nbatch_train >= nbatch))
          end = start + envsperbatch
          mbenvinds = envinds[start:end]
          mbflatinds = flatinds[mbenvinds].ravel()
          slices = [arr[mbflatinds] for arr in (obs, returns, masks, actions,
                                                values, neglogpacs, next_obs)]
          mbstates = states[mbenvinds]
          fetches = model.train(lrnow, cliprangenow, slices[0],
                                slices[6], slices[1], slices[2],
                                slices[3], slices[4], slices[5],
                                mbstates,
                                gather_histo=gather_histo, gather_sc=gather_sc,
                                debug_timeliner=debug_timeliner)
          mblossvals.append(fetches['losses'])
          if gather_histo:
            mbhistos.append(fetches['stats_histo'])
          if gather_sc:
            mbscs.append(fetches['stats_sc'])

    if debug_timeliner is not None:
      with logger.ProfileKV("save_timeline_json"):
        debug_timeliner.save(osp.join(workdir, 'timeline_{}.json'.format(update)))

    lossvals = np.mean(mblossvals, axis=0)
    assert len(mbscs) == 1
    scalars = mbscs[0]
    histograms = { n: np.concatenate([f[n] for f in mbhistos], axis=0) for n in model.stats_histo_names }
    logger.info('Histograms: {}'.format([(n, histograms[n].shape) for n in histograms.keys()]))
    #for v in histograms.values():
    #  assert len(v) == nbatch
    tnow = time.time()
    fps = int(nbatch / (tnow - tstart))
    if update % log_interval == 0 or update == 1:
      fps_total = int((update*nbatch) / (tnow - tfirststart))

      #tf_op_names = [i.name for i in tf.get_default_graph().get_operations()]
      #logger.info('#################### tf_op_names: {}'.format(tf_op_names))
      tf_num_ops = len(tf.get_default_graph().get_operations())
      logger.info('#################### tf_num_ops: {}'.format(tf_num_ops))
      logger.logkv('tf_num_ops', tf_num_ops)
      ev = explained_variance(values, returns)
      logger.logkv('serial_timesteps', update*nsteps)
      logger.logkv('nupdates', update)
      logger.logkv('total_timesteps', update*nbatch)
      logger.logkv('fps', fps)
      logger.logkv('fps_total', fps_total)
      logger.logkv('remaining_time', float(tnow - tfirststart) / float(update) * float(nupdates - update))
      logger.logkv('explained_variance', float(ev))
      logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
      logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
      if train_callback:
        train_callback(safemean([epinfo['l'] for epinfo in epinfobuf]),
                       safemean([epinfo['r'] for epinfo in epinfobuf]),
                       update * nbatch)
      logger.logkv('time_elapsed', tnow - tfirststart)
      for (lossval, lossname) in zip(lossvals, model.loss_names):
        logger.logkv(lossname, lossval)

      for n, v in scalars.items():
        logger.logkv(n, v)

      for n, v in histograms.items():
        logger.logkv(n, v)
        logger.logkv('mean_' + n, np.mean(v))
        logger.logkv('std_' + n, np.std(v))
        logger.logkv('max_' + n, np.max(v))
        logger.logkv('min_' + n, np.min(v))

      for n, v in locals().items():
        if n in ['rewards_int', 'rewards_int_raw']:
          logger.logkv(n, v)
        if n in ['rewards', 'rewards_ext', 'rewards_int', 'rewards_int_raw']:
          logger.logkv('mean_' + n, np.mean(v))
          logger.logkv('std_' + n, np.std(v))
          logger.logkv('max_' + n, np.max(v))
          logger.logkv('min_' + n, np.min(v))

      if model.rlb_model:
        if model.all_rlb_args.outer_args['rlb_normalize_ir']:
          logger.logkv('rlb_ir_running_mean', runner.irff_rms.mean)
          logger.logkv('rlb_ir_running_std', np.sqrt(runner.irff_rms.var))

      logger.dumpkvs()
    if (save_interval and (update % save_interval == 0 or update == 1) and
        workdir):
      checkdir = osp.join(workdir, 'checkpoints')
      if not tf.gfile.Exists(checkdir):
        tf.gfile.MakeDirs(checkdir)
      savepath = osp.join(checkdir, '%.5i'%update)
      print('Saving to', savepath)
      model.save(savepath)

      checkdir = osp.join(workdir, 'full_checkpoints')
      if not tf.gfile.Exists(checkdir):
        tf.gfile.MakeDirs(checkdir)
      savepath = osp.join(checkdir, '%.5i'%update)
      print('Saving to', savepath)
      save_state(savepath)
    if (cloud_sync_interval and update % cloud_sync_interval == 0 and
        cloud_sync_callback):
      cloud_sync_callback()
  env.close()
  return model


def safemean(xs):
  return np.mean(xs) if xs else np.nan

