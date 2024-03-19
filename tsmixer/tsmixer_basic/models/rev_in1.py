# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""Implementation of Reversible Instance Normalization."""

import json


class RevNorm:
  """Reversible Instance Normalization."""

  def __init__(self, axis, eps=1e-5, affine=True):
    self.axis = axis
    self.eps = eps
    self.affine = affine

  def call(self, x, mode, target_slice=None):
    if mode == 'norm':
      self._get_statistics(x)
      x = self._normalize(x)
    elif mode == 'denorm':
      x = self._denormalize(x, target_slice)
    else:
      raise NotImplementedError
    return x

  def _get_statistics(self, x):
    self.mean = x.mean(axis=self.axis, keepdims=True)
    self.stdev = (x.var(axis=self.axis, keepdims=True) + self.eps) ** 0.5

  def _normalize(self, x):
    x = x - self.mean
    x = x / self.stdev
    if self.affine:
      x = x * self.affine_weight
      x = x + self.affine_bias
    return x

  def _denormalize(self, x, target_slice=None):
    if self.affine:
      x = x - self.affine_bias[target_slice]
      x = x / self.affine_weight[target_slice]
    x = x * self.stdev[:, :, target_slice]
    x = x + self.mean[:, :, target_slice]
    return x


# Convert the RevNorm class to a JSON-serializable format
rev_norm_config = {
  'class_name': 'RevNorm',
  'config': {
    'axis': 1,  # Replace with the desired axis value
    'eps': 1e-5,
    'affine': True
  }
}

# Convert the RevNorm instance to a JSON-serializable format
rev_norm_instance = RevNorm(axis=1)
rev_norm_instance_config = rev_norm_instance.__dict__

# Save the JSON configurations to a file
with open('model_config.json', 'w') as f:
  json.dump({'rev_norm_config': rev_norm_config, 'rev_norm_instance_config': rev_norm_instance_config}, f)
