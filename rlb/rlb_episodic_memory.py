# coding=utf-8

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import gin
import numpy as np


#@gin.configurable
class RLBEpisodicMemory(object):
  """Episodic memory."""

  def __init__(self,
               observation_shape,
               #intrinsic_reward_fn,
               replacement='fifo',
               capacity=2000):
    """Creates an episodic memory.

    Args:
      observation_shape: Shape of an observation.
      observation_compare_fn: Function used to measure similarity between
        two observations. This function returns the estimated probability that
        two observations are similar.
      replacement: String to select the behavior when a sample is added
        to the memory when this one is full.
        Can be one of: 'fifo', 'random'.
        'fifo' keeps the last "capacity" samples into the memory.
        'random' results in a geometric distribution of the age of the samples
        present in the memory.
      capacity: Capacity of the episodic memory.

    Raises:
      ValueError: when the replacement scheme is invalid.
    """
    self._capacity = capacity
    self._replacement = replacement
    if self._replacement not in ['fifo', 'random']:
      raise ValueError('Invalid replacement scheme')
    self._observation_shape = observation_shape
    #self._intrinsic_reward_fn = intrinsic_reward_fn
    self.reset(False)

  def reset(self, show_stats=True):
    """Resets the memory."""
    if show_stats:
      size = len(self)
      age_histogram, _ = np.histogram(self._memory_age[:size],
                                      10, [0, self._count])
      age_histogram = age_histogram.astype(np.float32)
      age_histogram = age_histogram / np.sum(age_histogram)
      print('Number of samples added in the previous trajectory: {}'.format(
          self._count))
      print('Histogram of sample freshness (old to fresh): {}'.format(
          age_histogram))

    self._count = 0
    # Stores environment observations.
    self._obs_memory = np.zeros([self._capacity] + self._observation_shape)
    # Stores the infos returned by the environment. For debugging and
    # visualization purposes.
    self._info_memory = [None] * self._capacity
    self._memory_age = np.zeros([self._capacity], dtype=np.int32)

  @property
  def capacity(self):
    return self._capacity

  def __len__(self):
    return min(self._count, self._capacity)

  @property
  def info_memory(self):
    return self._info_memory

  def add(self, observation, info):
    """Adds an observation to the memory.

    Args:
      observation: Observation to add to the episodic memory.
      info: Info returned by the environment together with the observation,
            for debugging and visualization purposes.

    Raises:
      ValueError: when the capacity of the memory is exceeded.
    """
    if self._count >= self._capacity:
      if self._replacement == 'random':
        # By using random replacement, the age of elements inside the memory
        # follows a geometric distribution (more fresh samples compared to
        # old samples).
        index = np.random.randint(low=0, high=self._capacity)
      elif self._replacement == 'fifo':
        # In this scheme, only the last self._capacity elements are kept.
        # Samples are replaced using a FIFO scheme (implemented as a circular
        # buffer).
        index = self._count % self._capacity
      else:
        raise ValueError('Invalid replacement scheme')
    else:
      index = self._count

    self._obs_memory[index] = observation
    self._info_memory[index] = info
    self._memory_age[index] = self._count
    self._count += 1

  def get_data(self):
    return self._obs_memory[:len(self)]


