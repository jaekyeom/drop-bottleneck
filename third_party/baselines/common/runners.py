# coding=utf-8
import numpy as np
import abc
from abc import abstractmethod

class AbstractEnvRunner(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, env, model, nsteps, obs_dtype=None):
        self.env = env
        self.model = model
        nenv = env.num_envs
        self.batch_ob_shape = (nenv*nsteps,) + env.observation_space.shape
        if obs_dtype is None:
            obs_dtype = env.observation_space.dtype.name
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=obs_dtype)
        self.obs[:] = env.reset()
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    @abstractmethod
    def run(self):
        raise NotImplementedError
