import math
import functools
import numpy as np
import tensorflow as tf
from third_party.baselines import logger
from third_party.baselines.a2c.utils import fc, conv
from rlb.utils import CContext, EmptyClass
import gin

@gin.configurable
def get_rlb_args(
        rlb_ir_weight=0.005,
        rlb_loss_weight=1.0,
        rlb_normalize_ir=2,

        rlb_ot=1,
        rlb_ot_lr=1e-4,
        rlb_ot_batch_size=512,
        rlb_ot_history_size=1800,
        rlb_ot_train_interval=-1,
        rlb_ot_exploration_min_step=12600,
        rlb_ot_num_epochs=2,
        rlb_ot_memory_capacity=2000,
        rlb_ot_memory_algo='fifo',
        rlb_ot_deterministic_z_for_ir=1,
        rlb_ot_ir_clip_low=None,

        rlb_beta=0.001,
        rlb_dim_discriminator_hid_sizes=[64, 32, 16],
        rlb_dim_marginal_strategy='reshuffle_combinationwise',
        rlb_dim_no_joint_training=0,
        rlb_dim_train_ahead=8,
        rlb_dim_train_ahead_aggregate_all=1,
        rlb_dim_train_ahead_no_noise=0,
        rlb_featdrop_drop_prob_p_init=[-2.0, 1.0],
        rlb_featdrop_model_hid_sizes=[],
        rlb_featdrop_entropy_num_bins=32,
        rlb_featdrop_temperature=0.1,
        rlb_int_rew_type='epimem_dim_latent_sp',
        rlb_no_entropy_term=0,
        rlb_num_z_variables=1,
        rlb_prediction_term_num_samples=50,
        rlb_prediction_type='deepinfomax',
        rlb_target_dynamics='latent_aggmean',
        rlb_z_dim=128,

        debug_tf_timeline=0,
        ):

    assert rlb_prediction_type in ['deepinfomax']
    assert rlb_int_rew_type in ['epimem_dim_latent', 'epimem_dim_latent_sp', ]
    assert rlb_no_entropy_term in [0, 1]
    assert rlb_dim_train_ahead_aggregate_all in [0, 1]
    assert rlb_dim_train_ahead_no_noise in [0, 1]
    assert len(rlb_featdrop_drop_prob_p_init) in [1, 2]
    assert rlb_dim_marginal_strategy in ['half_batch', 'shuffle_combinationwise', 'reshuffle_combinationwise', 'shuffle_full', 'reshuffle_full']
    assert rlb_dim_no_joint_training in [0, 1]
    assert rlb_target_dynamics in ['latent_aggmean']
    assert rlb_ot_deterministic_z_for_ir in [0, 1]

    outer_arg_names = ['rlb_ir_weight', 'rlb_loss_weight', 'rlb_dim_no_joint_training', 'rlb_normalize_ir'] +  ['rlb_ot', 'rlb_ot_lr', 'rlb_ot_batch_size', 'rlb_ot_history_size', 'rlb_ot_train_interval', 'rlb_ot_exploration_min_step', 'rlb_ot_num_epochs', 'rlb_ot_memory_capacity', 'rlb_ot_memory_algo', 'rlb_ot_ir_clip_low']
    inner_args = {}
    outer_args = {}
    debug_args = {}

    l = locals()
    for k, v in l.items():
        if k.startswith('debug_'):
            debug_args[k] = v
        elif k.startswith('rlb_'):
            if k in outer_arg_names:
                outer_args[k] = v
            else:
                inner_args[k] = v

    result = EmptyClass()
    result.inner_args = inner_args
    result.outer_args = outer_args
    result.debug_args = debug_args
    return result

def define_rlb_model(ph_set, pdtype, ac_space,
                     optimizer,
                     outer_scope,

                     rlb_ot_deterministic_z_for_ir,

                     rlb_beta,
                     rlb_dim_discriminator_hid_sizes,
                     rlb_dim_marginal_strategy,
                     rlb_dim_train_ahead,
                     rlb_dim_train_ahead_aggregate_all,
                     rlb_dim_train_ahead_no_noise,
                     rlb_featdrop_drop_prob_p_init,
                     rlb_featdrop_model_hid_sizes,
                     rlb_featdrop_entropy_num_bins,
                     rlb_featdrop_temperature,
                     rlb_int_rew_type,
                     rlb_no_entropy_term,
                     rlb_num_z_variables,
                     rlb_prediction_term_num_samples,
                     rlb_prediction_type,
                     rlb_target_dynamics,
                     rlb_z_dim,

                     ph_set_for_embedding_net=None,
                     ph_set_for_epimem_ir=None,
                     ):
    logger.info("Using Dynamics Bottleneck ****************************************************")
    yes_gpu = True

    self = EmptyClass()
    self.stats_histo = EmptyClass()
    self.stats_sc = EmptyClass()


    pdparamsize = pdtype.param_shape()[0]
    convfeat = 32

    assert (ph_set is not None) + (ph_set_for_embedding_net is not None) + (ph_set_for_epimem_ir is not None) == 1
    if ph_set is not None:
        num_valid = tf.shape(ph_set.ph_ob_unscaled)[0]
        obs_shape = ph_set.ph_ob_unscaled.shape.as_list()[1:]

        logger.info('define_rlb_model. ph_set obs_shape: {}'.format(obs_shape))
    elif ph_set_for_embedding_net is not None:
        obs_shape = ph_set_for_embedding_net.ph_ob_unscaled.shape.as_list()[1:]

        rlb_prediction_term_num_samples = 1

        logger.info('define_rlb_model. ph_set_for_embedding_net obs_shape: {}'.format(obs_shape))
    elif ph_set_for_epimem_ir is not None:
        assert rlb_target_dynamics in ['latent_aggmean']
        rlb_prediction_term_num_samples = 1
    else:
        assert False

    obtain_deterministic_z = (ph_set_for_embedding_net is not None and rlb_ot_deterministic_z_for_ir)


    rlb_all_z_dim = rlb_z_dim * rlb_num_z_variables

    sched_coef = 1.

    def _preprocess_ob(ob):
        xr = ob
        if xr.dtype != tf.float32:
            xr = tf.cast(xr, tf.float32)
        xr = xr / 255.0
        return xr

    ccg = CContext(verbose=True, print_func=logger.info)

    if ph_set is not None:
        xr_prev_orig = _preprocess_ob(ph_set.ph_ob_unscaled)
        xr_next_orig = _preprocess_ob(ph_set.ph_ob_next_unscaled)
        ccg.register_state('xr_prev', lambda cc: xr_prev_orig)
        ccg.register_state('xr_next', lambda cc: xr_next_orig)
    elif ph_set_for_embedding_net is not None:
        xr_embedding_net = _preprocess_ob(ph_set_for_embedding_net.ph_ob_unscaled)
        ccg.register_state('xr_prev', lambda cc: xr_embedding_net)
        ccg.register_state('xr_next', lambda cc: xr_embedding_net)
    elif ph_set_for_epimem_ir is not None:
        pass
    else:
        assert False


    # {{{
    feat_model_activ_func = tf.nn.leaky_relu
    feat_model_conv_hps = [
        dict(nf=convfeat * 1, rf=8, stride=4, init_scale=np.sqrt(2)),
        dict(nf=convfeat * 2 * 1, rf=4, stride=2, init_scale=np.sqrt(2)),
        dict(nf=convfeat * 2 * 1, rf=3, stride=1, init_scale=np.sqrt(2)),
    ]

    def _rlb_feat_model(xr):
        with tf.variable_scope('drop_feat', reuse=tf.AUTO_REUSE), tf.device('/gpu:0' if yes_gpu else '/cpu:0'):
            for i, hps in enumerate(feat_model_conv_hps):
                xr = feat_model_activ_func(conv(xr, 'c{}r'.format(i+1), **hps))
            xr = to2d(xr)
            for i, size in enumerate(rlb_featdrop_model_hid_sizes):
                xr = feat_model_activ_func(fc(xr, 'fc_hid{}'.format(i+1), nh=size, init_scale=np.sqrt(2)))
            xr = fc(xr, 'fc', nh=rlb_all_z_dim, init_scale=np.sqrt(2))
            return xr

    featdrop_drop_probs_before_sigmoid = tf.get_variable(
            'featdrop_drop_probs_p', shape=[rlb_all_z_dim],
            initializer=tf.random_uniform_initializer(rlb_featdrop_drop_prob_p_init[0], rlb_featdrop_drop_prob_p_init[-1]))
    featdrop_drop_probs = tf.sigmoid(featdrop_drop_probs_before_sigmoid)
    featdrop_mean_retrain_probs = tf.reduce_mean(1.0 - featdrop_drop_probs)
    featdrop_scaler = 1.0 / (featdrop_mean_retrain_probs + 1e-5)

    self.stats_histo.featdrop_drop_probs = featdrop_drop_probs
    self.stats_histo.featdrop_drop_probs_before_sigmoid = featdrop_drop_probs_before_sigmoid
    self.stats_sc.featdrop_scaler = featdrop_scaler

    def _get_retain_mask(drop_probs, shape):
        uni = tf.random.uniform(shape)
        eps = 1e-8
        return 1.0 - tf.sigmoid(
                (tf.log(drop_probs + eps) - tf.log(1 - drop_probs + eps)
                 + tf.log(uni + eps) - tf.log(1 - uni + eps)) / rlb_featdrop_temperature)

    def _rlb_perform_compression(cc):
        # {{{
        out_dict = {}

        r_feat_prev = _rlb_feat_model(cc.xr_prev)
        r_feat_next = _rlb_feat_model(cc.xr_next)

        self.stats_histo.r_feat_next_mean = tf.reduce_mean(r_feat_next, axis=0)
        self.stats_histo.r_feat_next_std = tf.math.reduce_std(r_feat_next, axis=0)

        r_feat_prev_ex = r_feat_prev[:, None, :]
        r_feat_next_ex = r_feat_next[:, None, :]

        featdrop_drop_probs_prev = featdrop_drop_probs
        featdrop_drop_probs_next = featdrop_drop_probs

        if obtain_deterministic_z:
            drop_mask_prev = tf.cast(featdrop_drop_probs_prev < 0.5, tf.float32)[None, None, :]
            drop_mask_next = tf.cast(featdrop_drop_probs_next < 0.5, tf.float32)[None, None, :]
            featdrop_scaler_prev = float(rlb_all_z_dim) / (tf.reduce_sum(drop_mask_prev) + 1e-5)
            featdrop_scaler_next = float(rlb_all_z_dim) / (tf.reduce_sum(drop_mask_next) + 1e-5)
        else:
            drop_mask_prev = _get_retain_mask(
                    featdrop_drop_probs_prev[None, None, :],
                    tf.shape(r_feat_prev_ex) * tf.constant([1, rlb_prediction_term_num_samples, 1]))
            drop_mask_next = _get_retain_mask(
                    featdrop_drop_probs_next[None, None, :],
                    tf.shape(r_feat_next_ex) * tf.constant([1, rlb_prediction_term_num_samples, 1]))
            featdrop_scaler_prev = featdrop_scaler
            featdrop_scaler_next = featdrop_scaler

        self.stats_histo.drop_mask_next = drop_mask_next
        self.stats_sc.featdrop_scaler_next = featdrop_scaler_next

        z_prev = r_feat_prev_ex * drop_mask_prev * featdrop_scaler_prev
        z_next = r_feat_next_ex * drop_mask_next * featdrop_scaler_next

        if rlb_target_dynamics in ['latent_aggmean']:
            # {{{
            assert rlb_featdrop_entropy_num_bins is not None

            eps = 1e-5

            r_min = tf.reduce_min(r_feat_next, axis=0)
            r_max = tf.reduce_max(r_feat_next, axis=0)
            r_min_e = r_min - eps
            r_max_e = r_max + eps

            bin_widths = (r_max_e - r_min_e) / float(rlb_featdrop_entropy_num_bins)
            # bin_widths: [rlb_all_z_dim]
            assert bin_widths.shape.as_list() == [rlb_all_z_dim]
            edge_indices = tf.range(rlb_featdrop_entropy_num_bins + 1, dtype=tf.float32)

            bin_edges = bin_widths[None, :] * edge_indices[:, None] + r_min_e[None, :]
            # bin_edges: [rlb_featdrop_entropy_num_bins + 1, rlb_all_z_dim]
            # For `tfp.stats.histogram`, the edges dimension must be the first dimension.
            assert bin_edges.shape.as_list() == [rlb_featdrop_entropy_num_bins + 1, rlb_all_z_dim]

            import tensorflow_probability as tfp
            r_histogram = tfp.stats.histogram(
                    r_feat_next, bin_edges, axis=0,
                    extend_lower_interval=True, extend_upper_interval=True)

            # NOTE: We can avoid CPU-side computation for counting over fixed-width bins.

            # r_histogram: [rlb_featdrop_entropy_num_bins, rlb_all_z_dim]
            assert r_histogram.shape.as_list()[0] == rlb_featdrop_entropy_num_bins
            r_probabilities = r_histogram / tf.cast(tf.shape(r_feat_next)[0], tf.float32)

            self.stats_histo.r_histogram = r_histogram
            self.stats_histo.r_probabilities = r_probabilities

            minus_p_log_p_raw = (- r_probabilities * tf.log(r_probabilities))
            # Fix so that `- 0 log 0 = 0` is computed correctly.
            minus_p_log_p = tf.where(
                    tf.is_nan(minus_p_log_p_raw),
                    tf.zeros_like(minus_p_log_p_raw),
                    minus_p_log_p_raw)
            r_entropy_next = tf.reduce_sum(minus_p_log_p, axis=0)
            # r_entropy_next: [rlb_all_z_dim]
            # }}}

            r_entropy_next = tf.stop_gradient(r_entropy_next)

            compression_term = r_entropy_next * (1.0 - featdrop_drop_probs_next)
        else:
            raise Exception('Unknown rlb_target_dynamics: {}'.format(rlb_target_dynamics))

        self.stats_histo.r_entropy_next = r_entropy_next

        deterministic_mask_b = (featdrop_drop_probs_next < 0.5)  # boolean
        deterministic_mask = tf.cast(deterministic_mask_b, tf.float32)
        self.stats_sc.featdrop_det_num_preserved_feats = tf.reduce_sum(
                tf.cast(deterministic_mask_b, tf.int32))
        self.stats_sc.featdrop_det_full_entropy = tf.reduce_sum(r_entropy_next)
        self.stats_sc.featdrop_det_preserved_entropy = tf.reduce_sum(r_entropy_next * deterministic_mask)
        self.stats_sc.featdrop_det_dropped_entropy = self.stats_sc.featdrop_det_full_entropy - self.stats_sc.featdrop_det_preserved_entropy
        self.stats_sc.featdrop_det_compression_ratio = self.stats_sc.featdrop_det_preserved_entropy / self.stats_sc.featdrop_det_full_entropy

        z_prev = tf.reshape(z_prev, [-1] + z_prev.shape.as_list()[2:])
        z_next = tf.reshape(z_next, [-1] + z_next.shape.as_list()[2:])

        out_dict.update({
            'z_prev': z_prev,
            'z_next': z_next,
            'compression_term': compression_term,
        })

        return out_dict
        # }}}
    ccg.register_state('compression_inner', _rlb_perform_compression)
    # }}}

    if ph_set_for_embedding_net is not None:
        self.z = ccg.compression_inner['z_next']
        return self

    def _rlb_compression_wrapper(cc):
        compression_term = cc.compression_inner['compression_term']
        assert len(compression_term.shape.as_list()) == 1
        cd = None
        with tf.control_dependencies(cd):
            self.stats_histo.compression_terms = compression_term
        return cc.compression_inner
    ccg.register_state('compression', _rlb_compression_wrapper)

    if ph_set is not None:
        ac_valid = ph_set.ph_ac

        ac_valid = tf.expand_dims(ac_valid, axis=1)
        if rlb_target_dynamics in ['latent_aggmean']:
            ac_valid = tf.expand_dims(ac_valid, axis=1)
            ac_valid = tf.expand_dims(ac_valid, axis=1)
            ac_valid = tf.tile(ac_valid, [1, rlb_prediction_term_num_samples, rlb_num_z_variables, rlb_num_z_variables] + [1] * len(pdtype.sample_shape()))
        else:
            raise Exception('Unknown rlb_target_dynamics: {}'.format(rlb_target_dynamics))
        ac_valid = tf.reshape(ac_valid, [-1] + pdtype.sample_shape())
        with tf.control_dependencies([tf.assert_equal(tf.shape(ccg.compression['z_next'])[0] * rlb_num_z_variables * rlb_num_z_variables, tf.shape(ac_valid)[0])]):
            ac_valid = tf.identity(ac_valid)

    if rlb_prediction_type == 'deepinfomax':
        # {{{
        if rlb_target_dynamics in ['latent_aggmean']:
            def _rlb_dim_discriminator_latent(z):
                # Assume the two one-hot inputs for the parameterization of the discriminator are already part of z.
                if rlb_num_z_variables > 1:
                    assert z.shape.as_list()[-1] == (rlb_z_dim + rlb_num_z_variables) * 2
                else:
                    assert z.shape.as_list()[-1] == (rlb_z_dim) * 2
                with tf.variable_scope('deepinfomax_latent', reuse=tf.AUTO_REUSE), tf.device('/gpu:0' if yes_gpu else '/cpu:0'):
                    for i, size in enumerate(rlb_dim_discriminator_hid_sizes):
                        z = tf.nn.relu(fc(z, 'fc_hid{}'.format(i+1), nh=size, init_scale=np.sqrt(2)))
                    z = fc(z, 'fc_out', nh=1, init_scale=np.sqrt(2))
                    return z
            if ph_set is not None:
                z_s_shape = [num_valid, rlb_prediction_term_num_samples, rlb_num_z_variables, rlb_z_dim]
                ac_s_shape = [num_valid, rlb_prediction_term_num_samples, rlb_num_z_variables, rlb_num_z_variables] + pdtype.sample_shape()
        else:
            raise Exception('Unknown rlb_target_dynamics: {}'.format(rlb_target_dynamics))

        if ph_set is not None:
            def _rlb_perform_prediction(cc, for_dim_training, for_int_rew):
                # {{{
                force_aggregate_all = (for_dim_training and rlb_dim_train_ahead_aggregate_all)

                if for_dim_training and rlb_dim_train_ahead_no_noise:
                    z_next = cc.compression['z_next_mu']
                    z_prev = cc.compression['z_prev_mu']
                else:
                    z_next = cc.compression['z_next']
                    z_prev = cc.compression['z_prev']

                if for_dim_training:
                    # For Deep Infomax discriminator training, only the parameters of Deep Infomax should be updated.
                    z_next = tf.stop_gradient(z_next)
                    z_prev = tf.stop_gradient(z_prev)

                out_dict = {}

                z_next_s = tf.reshape(z_next, z_s_shape)
                z_prev_s = tf.reshape(z_prev, z_s_shape)
                ac_valid_s = tf.reshape(ac_valid, ac_s_shape)
                ac_valid_s = tf.one_hot(ac_valid_s, ac_space.n, axis=-1)
                half_batch_size = tf.cast(num_valid / 2, tf.int32)

                if rlb_target_dynamics in ['latent_aggmean']:
                    z_next_s = tf.tile(z_next_s[:, :, :, None, :], [1, 1, 1, rlb_num_z_variables, 1])
                    z_prev_s = tf.tile(z_prev_s[:, :, None, :, :], [1, 1, rlb_num_z_variables, 1, 1])

                    num_z_eye = tf.eye(rlb_num_z_variables)
                    z_next_idx_one_hot = tf.tile(num_z_eye[:, None, :], [1, rlb_num_z_variables, 1])
                    z_prev_idx_one_hot = tf.tile(num_z_eye[None, :, :], [rlb_num_z_variables, 1, 1])
                    z_idxes_one_hot = tf.concat([z_next_idx_one_hot, z_prev_idx_one_hot], axis=-1)

                    if rlb_dim_marginal_strategy == 'half_batch':
                        out_dict.update({
                            'num_prediction_terms': half_batch_size,
                        })
                        loss_batch_dim = half_batch_size * rlb_prediction_term_num_samples * rlb_num_z_variables
                        input_batch_dim = loss_batch_dim * rlb_num_z_variables

                        z_idxes_one_hot = tf.tile(z_idxes_one_hot[None, None, :, :, :], [half_batch_size, rlb_prediction_term_num_samples, 1, 1, 1])
                    elif rlb_dim_marginal_strategy in ['shuffle_combinationwise', 'reshuffle_combinationwise', 'shuffle_full', 'reshuffle_full']:
                        out_dict.update({
                            'num_prediction_terms': num_valid,
                        })
                        loss_batch_dim = num_valid * rlb_prediction_term_num_samples * rlb_num_z_variables
                        input_batch_dim = loss_batch_dim * rlb_num_z_variables

                        z_idxes_one_hot = tf.tile(z_idxes_one_hot[None, None, :, :, :], [num_valid, rlb_prediction_term_num_samples, 1, 1, 1])
                    else:
                        raise Exception('Unknown rlb_dim_marginal_strategy: {}'.format(rlb_dim_marginal_strategy))
                    if rlb_num_z_variables > 1:
                        input_feat_dim = (rlb_z_dim + rlb_num_z_variables) * 2 + ac_space.n
                    else:
                        input_feat_dim = (rlb_z_dim) * 2 + ac_space.n
                else:
                    raise Exception('Unknown rlb_target_dynamics: {}'.format(rlb_target_dynamics))

                if rlb_dim_marginal_strategy == 'half_batch':
                    pass
                elif rlb_dim_marginal_strategy in ['shuffle_combinationwise', 'reshuffle_combinationwise', 'shuffle_full', 'reshuffle_full']:
                    assert rlb_target_dynamics in ['latent_aggmean']
                    # {{{
                    combinationwise = (rlb_dim_marginal_strategy in ['shuffle_combinationwise', 'reshuffle_combinationwise'])

                    def _get_shuffled_data(data, rand_perm):
                        # {{{
                        # data: [num_valid, rlb_prediction_term_num_samples, rlb_num_z_variables, rlb_num_z_variables, ...]
                        if combinationwise:
                            remaining_axes = list(range(len(data.shape.as_list())))[4:]
                            perm_for = [2, 3, 0, 1] + remaining_axes
                            perm_inv = [2, 3, 0, 1] + remaining_axes
                        else:
                            data_remaining_shape = data.shape.as_list()[2:]
                            data = tf.reshape(
                                    data,
                                    [num_valid * rlb_prediction_term_num_samples] + data_remaining_shape,
                            )
                            remaining_axes = list(range(len(data.shape.as_list())))[3:]
                            perm_for = [1, 2, 0] + remaining_axes
                            perm_inv = [2, 0, 1] + remaining_axes

                        shuffled = tf.gather(
                                tf.transpose(data, perm=perm_for),
                                indices=rand_perm,
                                axis=2,
                                batch_dims=2)
                        shuffled = tf.transpose(shuffled, perm=perm_inv)
                        with tf.control_dependencies([tf.assert_equal(tf.shape(data), tf.shape(shuffled))]):
                            shuffled = tf.identity(shuffled)
                        if not combinationwise:
                            shuffled = tf.reshape(
                                    shuffled,
                                    [num_valid, rlb_prediction_term_num_samples] + data_remaining_shape,
                            )
                        return shuffled
                        # }}}

                    if rlb_dim_marginal_strategy in ['shuffle_combinationwise', 'shuffle_full']:
                        z_prev_s_shuffled_lat = z_prev_s_shuffled
                    elif rlb_dim_marginal_strategy in ['reshuffle_combinationwise', 'reshuffle_full']:
                        if combinationwise:
                            random_perm_lat = tf.argsort(
                                    tf.random.uniform([rlb_num_z_variables, rlb_num_z_variables, num_valid]),
                                    axis=2)
                        else:
                            random_perm_lat = tf.argsort(
                                    tf.random.uniform([rlb_num_z_variables, rlb_num_z_variables, num_valid * rlb_prediction_term_num_samples]),
                                    axis=2)
                        z_prev_s_shuffled_lat = _get_shuffled_data(z_prev_s, random_perm_lat)


                    if rlb_target_dynamics in ['latent_aggmean']:
                        pass
                    else:
                        raise Exception('Unexpected rlb_target_dynamics: {}'.format(rlb_target_dynamics))
                    # }}}
                else:
                    raise Exception('Unknown rlb_dim_marginal_strategy: {}'.format(rlb_dim_marginal_strategy))

                if rlb_target_dynamics in ['latent_aggmean']:
                    prediction_term = tf.constant(0.0)

                if rlb_target_dynamics in ['latent_aggmean']:
                    def _convert_neg_discriminator_output_to_int_rew(values):
                        values = tf.reshape(
                                values,
                                [loss_batch_dim, rlb_num_z_variables])
                        if rlb_target_dynamics in ['latent_aggmean'] or force_aggregate_all:
                            values_weights = 1.0 / float(rlb_num_z_variables)
                        values = tf.reshape(
                                tf.reduce_sum(values_weights * values, axis=-1),
                                [num_valid, rlb_prediction_term_num_samples * rlb_num_z_variables])
                        return tf.stop_gradient(tf.reduce_mean(values, axis=-1))

                    if for_int_rew:
                        if rlb_int_rew_type in ['prediction_dim_direct', 'prediction_dim_direct_sp']:
                            if rlb_int_rew_type in ['prediction_dim_direct']:
                                self.int_rew = _convert_neg_discriminator_output_to_int_rew(tf.negative(joint_output))
                            elif rlb_int_rew_type in ['prediction_dim_direct_sp']:
                                self.int_rew = _convert_neg_discriminator_output_to_int_rew(tf.nn.softplus(tf.negative(joint_output)))

                    # Latent dynamics prediction terms.
                    if rlb_num_z_variables > 1:
                        input_feat_dim = (rlb_z_dim + rlb_num_z_variables) * 2
                    else:
                        input_feat_dim = (rlb_z_dim) * 2

                    if rlb_dim_marginal_strategy == 'half_batch':
                        if rlb_num_z_variables > 1:
                            joint_input = tf.concat([
                                z_next_s[:half_batch_size],
                                z_prev_s[:half_batch_size],
                                z_idxes_one_hot], axis=-1)
                            marginal_input = tf.concat([
                                z_next_s[:half_batch_size],
                                z_prev_s[half_batch_size:half_batch_size*2],
                                z_idxes_one_hot], axis=-1)
                        else:
                            joint_input = tf.concat([
                                z_next_s[:half_batch_size],
                                z_prev_s[:half_batch_size]], axis=-1)
                            marginal_input = tf.concat([
                                z_next_s[:half_batch_size],
                                z_prev_s[half_batch_size:half_batch_size*2]], axis=-1)
                        joint_input = tf.reshape(joint_input, [input_batch_dim, input_feat_dim])
                        marginal_input = tf.reshape(marginal_input, [input_batch_dim, input_feat_dim])
                    elif rlb_dim_marginal_strategy in ['shuffle_combinationwise', 'reshuffle_combinationwise', 'shuffle_full', 'reshuffle_full']:
                        if rlb_num_z_variables > 1:
                            joint_input = tf.concat([
                                z_next_s,
                                z_prev_s,
                                z_idxes_one_hot], axis=-1)
                            marginal_input = tf.concat([
                                z_next_s,
                                z_prev_s_shuffled_lat,
                                z_idxes_one_hot], axis=-1)
                        else:
                            joint_input = tf.concat([
                                z_next_s,
                                z_prev_s], axis=-1)
                            marginal_input = tf.concat([
                                z_next_s,
                                z_prev_s_shuffled_lat], axis=-1)
                        joint_input = tf.reshape(joint_input, [input_batch_dim, input_feat_dim])
                        marginal_input = tf.reshape(marginal_input, [input_batch_dim, input_feat_dim])
                    else:
                        raise Exception('Unknown rlb_dim_marginal_strategy: {}'.format(rlb_dim_marginal_strategy))

                    joint_output_latent = _rlb_dim_discriminator_latent(joint_input)
                    marginal_output_latent = _rlb_dim_discriminator_latent(marginal_input)
                    prediction_term_latent = 0.5 * (- (tf.nn.softplus(tf.negative(joint_output_latent)) + tf.nn.softplus(marginal_output_latent)) + tf.log(4.0))
                    prediction_term_latent = tf.squeeze(prediction_term_latent, axis=[-1])

                    if for_int_rew:
                        if rlb_int_rew_type in ['prediction_dim_latent', 'prediction_dim_latent_sp']:
                            if rlb_int_rew_type in ['prediction_dim_latent']:
                                self.int_rew = _convert_neg_discriminator_output_to_int_rew(tf.negative(joint_output_latent))
                            elif rlb_int_rew_type in ['prediction_dim_latent_sp']:
                                self.int_rew = _convert_neg_discriminator_output_to_int_rew(tf.nn.softplus(tf.negative(joint_output_latent)))
                        elif rlb_int_rew_type in ['prediction_dim_both', 'prediction_dim_both_sp']:
                            if rlb_int_rew_type in ['prediction_dim_both']:
                                self.int_rew = (
                                        _convert_neg_discriminator_output_to_int_rew(tf.negative(joint_output)) +
                                        _convert_neg_discriminator_output_to_int_rew(tf.negative(joint_output_latent)))
                            elif rlb_int_rew_type in ['prediction_dim_both_sp']:
                                self.int_rew = (
                                        _convert_neg_discriminator_output_to_int_rew(tf.nn.softplus(tf.negative(joint_output))) +
                                        _convert_neg_discriminator_output_to_int_rew(tf.nn.softplus(tf.negative(joint_output_latent))))

                    prediction_term_latent = tf.reshape(
                            prediction_term_latent,
                            [loss_batch_dim, rlb_num_z_variables])
                    if rlb_target_dynamics in ['latent_aggmean'] or force_aggregate_all:
                        prediction_term_latent_weights = 1.0 / float(rlb_num_z_variables)
                    prediction_term_latent = tf.reduce_sum(prediction_term_latent_weights * prediction_term_latent, axis=-1)

                    out_dict.update({
                        'prediction_term_latent': prediction_term_latent,
                    })

                out_dict.update({
                    'prediction_term': prediction_term,
                })

                if rlb_int_rew_type in [
                        'prediction_dim_det_direct', 'prediction_dim_det_latent', 'prediction_dim_det_both',
                        'prediction_dim_det_direct_sp', 'prediction_dim_det_latent_sp', 'prediction_dim_det_both_sp',
                    ]:
                    raise NotImplementedError

                return out_dict
                # }}}

            ccg.register_state('prediction_inner', functools.partial(_rlb_perform_prediction, for_dim_training=False, for_int_rew=False))
            ccg.register_state('prediction_inner_disc', functools.partial(_rlb_perform_prediction, for_dim_training=True, for_int_rew=False))
            if rlb_int_rew_type in [
                'prediction_dim_direct', 'prediction_dim_latent', 'prediction_dim_both',
                'prediction_dim_direct_sp', 'prediction_dim_latent_sp', 'prediction_dim_both_sp',
                'prediction_dim_det_direct', 'prediction_dim_det_latent', 'prediction_dim_det_both',
                'prediction_dim_det_direct_sp', 'prediction_dim_det_latent_sp', 'prediction_dim_det_both_sp',
            ]:
                ccg.register_state('prediction_inner_int_rew', functools.partial(_rlb_perform_prediction, for_dim_training=False, for_int_rew=True))
        elif ph_set_for_epimem_ir is not None:
            def _rlb_compute_epimem_ir(cc):
                # {{{
                z_memory = ph_set_for_epimem_ir.ph_emb_memory
                z_target = ph_set_for_epimem_ir.ph_emb_target

                assert len(z_memory.shape.as_list()) == 3
                assert len(z_target.shape.as_list()) == 3

                num_envs = tf.shape(z_target)[0]
                memory_size = tf.shape(z_memory)[1]
                target_size = tf.shape(z_target)[1]
                batch_size = num_envs * memory_size * target_size

                assert rlb_prediction_term_num_samples == 1, 'Technically rlb_prediction_term_num_samples > 1 is possible, but not using it yet.'
                z_memory_s = tf.tile(z_memory[:, None, :, :], [1, target_size, 1, 1])
                z_memory_s = tf.reshape(
                        z_memory_s,
                        [batch_size, rlb_prediction_term_num_samples, rlb_num_z_variables, rlb_z_dim])
                z_target_s = tf.tile(z_target[:, :, None, :], [1, 1, memory_size, 1])
                z_target_s = tf.reshape(
                        z_target_s,
                        [batch_size, rlb_prediction_term_num_samples, rlb_num_z_variables, rlb_z_dim])

                del z_memory
                del z_target

                z_memory_as_next = tf.tile(z_memory_s[:, :, :, None, :], [1, 1, 1, rlb_num_z_variables, 1])
                z_memory_as_prev = tf.tile(z_memory_s[:, :, None, :, :], [1, 1, rlb_num_z_variables, 1, 1])
                z_target_as_next = tf.tile(z_target_s[:, :, :, None, :], [1, 1, 1, rlb_num_z_variables, 1])
                z_target_as_prev = tf.tile(z_target_s[:, :, None, :, :], [1, 1, rlb_num_z_variables, 1, 1])

                num_z_eye = tf.eye(rlb_num_z_variables)
                z_next_idx_one_hot = tf.tile(num_z_eye[:, None, :], [1, rlb_num_z_variables, 1])
                z_prev_idx_one_hot = tf.tile(num_z_eye[None, :, :], [rlb_num_z_variables, 1, 1])
                z_idxes_one_hot = tf.concat([z_next_idx_one_hot, z_prev_idx_one_hot], axis=-1)
                z_idxes_one_hot = tf.tile(z_idxes_one_hot[None, None, :, :, :], [batch_size, rlb_prediction_term_num_samples, 1, 1, 1])

                loss_batch_dim = batch_size * rlb_prediction_term_num_samples * rlb_num_z_variables
                input_batch_dim = loss_batch_dim * rlb_num_z_variables
                if rlb_num_z_variables > 1:
                    input_feat_dim = (rlb_z_dim + rlb_num_z_variables) * 2
                else:
                    input_feat_dim = (rlb_z_dim) * 2

                def _convert_neg_discriminator_output_to_int_rew(values):
                    values = tf.reshape(
                            values,
                            [loss_batch_dim, rlb_num_z_variables])
                    if rlb_target_dynamics in ['latent_aggmean'] or force_aggregate_all:
                        values_weights = 1.0 / float(rlb_num_z_variables)
                    values = tf.reshape(
                            tf.reduce_sum(values_weights * values, axis=-1),
                            [num_envs, target_size, memory_size * rlb_prediction_term_num_samples * rlb_num_z_variables])
                    return tf.stop_gradient(tf.reduce_mean(values, axis=-1))

                if rlb_num_z_variables > 1:
                    from_memory_to_target = tf.concat([
                            z_target_as_next,
                            z_memory_as_prev,
                            z_idxes_one_hot], axis=-1)
                else:
                    from_memory_to_target = tf.concat([
                            z_target_as_next,
                            z_memory_as_prev], axis=-1)
                from_memory_to_target = tf.reshape(from_memory_to_target, [input_batch_dim, input_feat_dim])
                output_from_memory_to_target = _rlb_dim_discriminator_latent(from_memory_to_target)

                if rlb_num_z_variables > 1:
                    from_target_to_memory = tf.concat([
                            z_memory_as_next,
                            z_target_as_prev,
                            z_idxes_one_hot], axis=-1)
                else:
                    from_target_to_memory = tf.concat([
                            z_memory_as_next,
                            z_target_as_prev], axis=-1)
                from_target_to_memory = tf.reshape(from_target_to_memory, [input_batch_dim, input_feat_dim])
                output_from_target_to_memory = _rlb_dim_discriminator_latent(from_target_to_memory)

                if rlb_int_rew_type in ['epimem_dim_latent', 'epimem_dim_latent_sp']:
                    if rlb_int_rew_type in ['epimem_dim_latent']:
                        ir_from_memory_to_target = _convert_neg_discriminator_output_to_int_rew(
                                tf.negative(output_from_memory_to_target))
                        ir_from_target_to_memory = _convert_neg_discriminator_output_to_int_rew(
                                tf.negative(output_from_target_to_memory))
                    elif rlb_int_rew_type in ['epimem_dim_latent_sp']:
                        ir_from_memory_to_target = _convert_neg_discriminator_output_to_int_rew(
                                tf.nn.softplus(tf.negative(output_from_memory_to_target)))
                        ir_from_target_to_memory = _convert_neg_discriminator_output_to_int_rew(
                                tf.nn.softplus(tf.negative(output_from_target_to_memory)))
                    ir_total = ir_from_memory_to_target + ir_from_target_to_memory
                else:
                    from_memory_to_target = tf.stop_gradient(tf.reshape(
                            tf.nn.softplus(tf.negative(output_from_memory_to_target)),
                            [num_envs, target_size, memory_size]))
                    from_target_to_memory = tf.stop_gradient(tf.reshape(
                            tf.nn.softplus(tf.negative(output_from_target_to_memory)),
                            [num_envs, target_size, memory_size]))
                    all_outputs = tf.concat([from_memory_to_target, from_target_to_memory], axis=-1)
                    together = tf.cond(
                            tf.equal(memory_size, 0),
                            lambda: tf.fill([num_envs, target_size, 1], np.nan),
                            lambda: all_outputs)
                    if rlb_int_rew_type in ['epimem_dim_latent_sp_max']:
                        ir_total = tf.reduce_max(together, axis=-1)
                    elif rlb_int_rew_type.startswith('epimem_dim_latent_sp_percentile-'):
                        percentile = float(rlb_int_rew_type.split('-')[-1])
                        ir_total = tf.contrib.distributions.percentile(together, q=percentile, axis=[-1])
                    else:
                        assert False

                return ir_total
                # }}}
            ccg.register_state('epimem_ir', _rlb_compute_epimem_ir)
        else:
            assert False

        # }}}
    else:
        raise Exception('Unknown rlb_prediction_type: {}'.format(rlb_prediction_type))

    if ph_set_for_epimem_ir is not None:
        self.epimem_ir = ccg.epimem_ir
        return self

    def _rlb_prediction_wrapper(cc):
        prediction_inner = cc.prediction_inner
        prediction_term = prediction_inner['prediction_term']

        if rlb_target_dynamics in ['latent_aggmean']:
            prediction_term_latent = prediction_inner['prediction_term_latent']
            assert len(prediction_term_latent.shape.as_list()) == 1
            self.stats_histo.prediction_terms_latent = prediction_term_latent

        return prediction_inner
    ccg.register_state('prediction', _rlb_prediction_wrapper)

    def _rlb_losses(cc):
        compression_term = cc.compression['compression_term']
        prediction_term = cc.prediction['prediction_term']

        aux_prediction_loss = (- tf.reduce_mean(prediction_term))
        aux_compression_loss = tf.reduce_mean(compression_term)
        aux_loss = sched_coef * aux_prediction_loss + rlb_beta * aux_compression_loss

        aux_all_prediction_losses = aux_prediction_loss

        if rlb_target_dynamics in ['latent_aggmean']:
            prediction_term_latent = cc.prediction['prediction_term_latent']
            aux_prediction_latent_loss = (- tf.reduce_mean(prediction_term_latent))
            aux_loss = aux_loss + sched_coef * aux_prediction_latent_loss

            aux_all_prediction_losses = aux_all_prediction_losses + aux_prediction_latent_loss

        return {
            'aux_prediction_loss': aux_prediction_loss,
            'aux_compression_loss': aux_compression_loss,
            'aux_loss': aux_loss,
            'aux_all_prediction_losses': aux_all_prediction_losses,
            'aux_prediction_latent_loss': aux_prediction_latent_loss,
        }
    ccg.register_state('losses', _rlb_losses)

    if rlb_int_rew_type in [
        'prediction_dim_direct', 'prediction_dim_latent', 'prediction_dim_both',
        'prediction_dim_direct_sp', 'prediction_dim_latent_sp', 'prediction_dim_both_sp',
        'prediction_dim_det_direct', 'prediction_dim_det_latent', 'prediction_dim_det_both',
        'prediction_dim_det_direct_sp', 'prediction_dim_det_latent_sp', 'prediction_dim_det_both_sp',
    ]:
        dummy = ccg.prediction_inner_int_rew

    if rlb_prediction_type == 'deepinfomax':
        def _rlb_train_dim_discriminator(cc):
            dim_train_loss = (- tf.reduce_mean(cc.prediction_inner_disc['prediction_term']))
            if rlb_target_dynamics in ['latent_aggmean']:
                dim_train_loss = dim_train_loss + (- tf.reduce_mean(cc.prediction_inner_disc['prediction_term_latent']))
            return dim_train_loss
        ccg.register_state('dim_train_loss', _rlb_train_dim_discriminator)

        self.dim_train_loss = ccg.dim_train_loss
    if rlb_dim_train_ahead > 0:
        dim_params = []
        dim_params.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=outer_scope + '/' + 'deepinfomax/'))
        dim_params.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=outer_scope + '/' + 'deepinfomax_latent/'))
        dim_trainer = optimizer

        dependencies_last = None
        for n in range(rlb_dim_train_ahead):
            with tf.control_dependencies(dependencies_last):
                dim_train_loss = ccg.dim_train_loss
                dim_grads_and_vars = dim_trainer.compute_gradients(dim_train_loss, dim_params)
                dim_train_op = dim_trainer.apply_gradients(dim_grads_and_vars)
                dependencies_last = [dim_train_op]
                # Because now discriminator is updated.
                ccg.invalidate_state('prediction_inner_disc')
        # Redundant, but placed here with a semantic purpose.
        ccg.invalidate_state('prediction_inner')

        loss_dependencies = [dim_train_op]
    else:
        loss_dependencies = None

    with tf.control_dependencies(loss_dependencies):
        for k, v in ccg.losses.items():
            setattr(self, k, v)

    if hasattr(self, 'int_rew'):
        self.stats_histo.int_rew = self.int_rew

    return self

def construct_ph_set(x, x_next, a):
    ph_set = EmptyClass()
    ph_set.ph_ob_unscaled = x
    ph_set.ph_ob_next_unscaled = x_next
    ph_set.ph_ac = a
    return ph_set

def construct_ph_set_for_embedding_net(x):
    ph_set = EmptyClass()
    ph_set.ph_ob_unscaled = x
    return ph_set

def construct_ph_set_for_epimem_ir(memory, x):
    ph_set = EmptyClass()
    ph_set.ph_emb_memory = memory
    ph_set.ph_emb_target = x
    return ph_set

def to2d(x):
    size = 1
    for shapel in x.get_shape()[1:]:
        size *= shapel.value
    return tf.reshape(x, (-1, size))


