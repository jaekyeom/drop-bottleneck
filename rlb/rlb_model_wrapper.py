# coding=utf-8

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import tempfile

from absl import logging
from third_party.keras_resnet import models
import numpy as np
import tensorflow as tf

from rlb.rlb_model import define_rlb_model, construct_ph_set, construct_ph_set_for_embedding_net, construct_ph_set_for_epimem_ir, get_rlb_args
from third_party.baselines import logger
from third_party.baselines.common.distributions import make_pdtype


class RLBModelWrapper(object):
  """Encapsulates a trained R network, with lazy loading of weights."""

  def __init__(self,
               input_shape,
               action_space,
               max_grad_norm=0.5,
               ):
    """Inits the RNetwork.

    Args:
      input_shape: (height, width, channel)
      weight_path: Path to the weights of the r_network.
    """

    self.input_shape = input_shape

    self.all_rlb_args = get_rlb_args()

    trainer = tf.train.AdamOptimizer(learning_rate=self.all_rlb_args.outer_args['rlb_ot_lr'])

    policy_pdtype = make_pdtype(action_space)
    self.policy_pdtype = policy_pdtype

    train_batch_size = self.all_rlb_args.outer_args['rlb_ot_batch_size']
    ph_obs = tf.placeholder(shape=(train_batch_size,) + input_shape, dtype=tf.uint8, name='obs')
    ph_obs_next = tf.placeholder(shape=(train_batch_size,) + input_shape, dtype=tf.uint8, name='obs_next')
    ph_acs = policy_pdtype.sample_placeholder([train_batch_size])

    ph_emb_net_obs = tf.placeholder(shape=(None,) + input_shape, dtype=tf.uint8, name='emb_net_obs')

    self.rlb_all_z_dim = self.all_rlb_args.inner_args['rlb_z_dim'] * self.all_rlb_args.inner_args['rlb_num_z_variables']
    ph_epimem_ir_emb_memory = tf.placeholder(shape=(None, None, self.rlb_all_z_dim), dtype=tf.float32, name='epimem_ir_emb_memory')
    ph_epimem_ir_emb_target = tf.placeholder(shape=(None, None, self.rlb_all_z_dim), dtype=tf.float32, name='epimem_ir_emb_target')

    rlb_scope = 'rlb_model'
    self._rlb_model = tf.make_template(
        rlb_scope, define_rlb_model,
        create_scope_now_=True,
        pdtype=policy_pdtype,
        ac_space=action_space,
        optimizer=trainer,
        outer_scope=rlb_scope,
        **self.all_rlb_args.inner_args)

    rlb_train_extra_kwargs = dict()
    rlb_train_out = self._rlb_model(
        ph_set=construct_ph_set(
            x=ph_obs,
            x_next=ph_obs_next,
            a=ph_acs),
        ph_set_for_embedding_net=None,
        ph_set_for_epimem_ir=None,
        **rlb_train_extra_kwargs
        )
    loss = rlb_train_out.aux_loss

    self._loss_names = ['rlb_loss']
    self._stats_histo_names = sorted(list(rlb_train_out.stats_histo.__dict__.keys()))
    self._stats_sc_names = sorted(list(rlb_train_out.stats_sc.__dict__.keys()))

    params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=rlb_scope)
    logger.info('RLBModelWrapper, {} trainable parameters: {}'.format(len(params), [p.name for p in params]))
    grads = tf.gradients(loss, params)
    grads_raw_global_norm = tf.global_norm(grads)
    if max_grad_norm is not None:
      grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
      grads_clipped_global_norm = tf.global_norm(grads)
    grads = list(zip(grads, params))
    train_op = trainer.apply_gradients(grads)
    def _train(obs, obs_next, acs, gather_histo=False, gather_sc=False):
      fetches = {
        'train': train_op,
        'losses': [loss],
      }
      if gather_histo:
        fetches['stats_histo'] = { n: getattr(rlb_train_out.stats_histo, n) for n in self._stats_histo_names }
      if gather_sc:
        fetches['stats_sc'] = { n: getattr(rlb_train_out.stats_sc, n) for n in self._stats_sc_names }
        fetches['additional_sc'] = {
          'rlb_grads_raw_global_norm': grads_raw_global_norm,
        }
        if max_grad_norm is not None:
          fetches['additional_sc'].update({
            'rlb_grads_clipped_global_norm': grads_clipped_global_norm,
          })
      sess = tf.get_default_session()
      result = sess.run(fetches, {ph_obs: obs, ph_obs_next: obs_next, ph_acs: acs})
      return result

    self._train = _train

    rlb_eval_extra_kwargs = dict()
    embedding_output = self._rlb_model(
        ph_set=None,
        ph_set_for_embedding_net=construct_ph_set_for_embedding_net(
            ph_emb_net_obs),
        ph_set_for_epimem_ir=None,
        **rlb_eval_extra_kwargs
        ).z
    def _embedding_network(obs):
      sess = tf.get_default_session()
      return sess.run(embedding_output, {ph_emb_net_obs: obs})
    self._embedding_network = _embedding_network

    epimem_ir_output = self._rlb_model(
        ph_set=None,
        ph_set_for_embedding_net=None,
        ph_set_for_epimem_ir=construct_ph_set_for_epimem_ir(ph_epimem_ir_emb_memory, ph_epimem_ir_emb_target),
        **rlb_eval_extra_kwargs
        ).epimem_ir
    def _ir_network(memory, x):
      sess = tf.get_default_session()
      ir = sess.run(epimem_ir_output, {ph_epimem_ir_emb_memory: memory, ph_epimem_ir_emb_target: x})
      # Don't multiply the IR weight here since it will be normalized in RLBEnvWrapper.
      #ir = ir * self.all_rlb_args.outer_args['rlb_ir_weight']
      return ir
    self._ir_network = _ir_network

  def _maybe_load_weights(self):
    """Loads R-network weights if needed.

    The RNetwork is used together with an environment used by ppo2.learn.
    Unfortunately, ppo2.learn initializes all global TF variables at the
    beginning of the training, which in particular, random-initializes the
    weights of the R Network. We therefore load the weights lazily, to make sure
    they are loaded after the global initialization happens in ppo2.learn.
    """
    if self._weights_loaded:
      return
    if self._weight_path is None:
      # Typically the case when doing online training of the R-network.
      return
    # Keras does not support reading weights from CNS, so we have to copy the
    # weights to a temporary local file.
    with tempfile.NamedTemporaryFile(prefix='r_net', suffix='.h5',
                                     delete=False) as tmp_file:
      tmp_path = tmp_file.name
    tf.gfile.Copy(self._weight_path, tmp_path, overwrite=True)
    logging.info('Loading weights from %s...', tmp_path)
    print('Loading into R network:')
    self._r_network.summary()
    self._r_network.load_weights(tmp_path)
    tf.gfile.Remove(tmp_path)
    self._weights_loaded = True

  def embed_observation(self, x):
    """Embeds an observation.

    Args:
      x: batched input observations. Expected to have the shape specified when
         the RNetwork was contructed (plus the batch dimension as first dim).

    Returns:
      embedding, shape [batch, models.EMBEDDING_DIM]
    """
    return self._embedding_network(x)

  def compute_intrinsic_rewards(self, memory_set, x_set):
    num_envs = len(memory_set)
    memory_len_0th = len(memory_set[0])
    if all(len(memory) == memory_len_0th for memory in memory_set):
      ir = self._ir_network(memory_set, x_set[:, None, :])
    else:
      ir = np.concatenate(
          [self._ir_network(memory_set[i:i+1], x_set[i:i+1, None, :]) for i in range(num_envs)],
          axis=0)
    ir = np.squeeze(ir, axis=-1)
    return ir

  def train(self, batch_gen, steps_per_epoch, num_epochs):
    mblossvals = []
    mbhistos = []
    mbscs = []
    mbascs = []
    for epoch in range(num_epochs):
      gather_histo = (epoch == num_epochs - 1)
      for step in range(steps_per_epoch):
        gather_sc = ((epoch == num_epochs - 1) and (step == steps_per_epoch - 1))
        obs, obs_next, acs = next(batch_gen)
        with logger.ProfileKV('train_ot_inner'):
          fetches = self._train(
              obs, obs_next, acs,
              gather_histo=gather_histo, gather_sc=gather_sc)
        mblossvals.append(fetches['losses'])
        if gather_histo:
          mbhistos.append(fetches['stats_histo'])
        if gather_sc:
          mbscs.append(fetches['stats_sc'])
          mbascs.append(fetches['additional_sc'])

    lossvals = np.mean(mblossvals, axis=0)
    assert len(mbscs) == 1
    assert len(mbascs) == 1
    scalars = mbscs[0]
    additional_scalars = mbascs[0]
    histograms = { n: np.concatenate([f[n] for f in mbhistos], axis=0) for n in self._stats_histo_names }
    logger.info('RLBModelWrapper.train histograms: {}'.format([(n, histograms[n].shape) for n in histograms.keys()]))

    for (lossval, lossname) in zip(lossvals, self._loss_names):
      logger.logkv(lossname, lossval)

    for n, v in scalars.items():
      logger.logkv(n, v)

    for n, v in additional_scalars.items():
      logger.logkv(n, v)

    for n, v in histograms.items():
      logger.logkv(n, v)
      logger.logkv('mean_' + n, np.mean(v))
      logger.logkv('std_' + n, np.std(v))
      logger.logkv('max_' + n, np.max(v))
      logger.logkv('min_' + n, np.min(v))


