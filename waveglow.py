import os
import math
import copy
import functools
import tensorflow as tf
from tensorflow.python.util import nest

import commons


def init_discrete_bottleneck(bottleneck_bits, bottleneck_dims_per_bit, dtype="float32"):
  """Get lookup table for discrete bottleneck."""
  bottleneck_size = 2 ** bottleneck_bits
  discrete_channels = bottleneck_bits * bottleneck_dims_per_bit
  means = commons.get_variable(
      name="means",
      shape=[bottleneck_size, discrete_channels],
      dtype=dtype)
  return means


def vqvae(x, hparams):
  """Combining EM and VAE"""
  bottleneck_size = 2**hparams.bottleneck_bits
  means = hparams.means

  # Caculate square distance
  def _square_distance(x, means):
    x = tf.cast(x, tf.float32)
    means = tf.cast(means, tf.float32)
    x_sg = tf.stop_gradient(x)
    x_norm_sq = tf.reduce_sum(tf.square(x_sg), axis=-1, keepdims=True) # [b, 1]
    means_norm_sq = tf.reduce_sum(tf.square(means), axis=-1, keepdims=True) # [V, 1]
    scalar_prod = tf.matmul(x_sg, means, transpose_b=True) # [b, V]
    dist_sq = x_norm_sq + tf.transpose(means_norm_sq) - 2 * scalar_prod # [b, V]

    tf.summary.histogram("dist_sq", dist_sq)
    tf.summary.histogram("len_sq", means_norm_sq)
    return tf.cast(dist_sq, x.dtype.base_dtype)
  dist_sq = _square_distance(x, means)

  q = tf.stop_gradient(tf.nn.softmax(-.5 * dist_sq))

  discrete = tf.one_hot(tf.argmax(-dist_sq, axis=-1), depth=bottleneck_size, dtype=means.dtype.base_dtype)
  dense = tf.matmul(discrete, means)
  if hparams.mode == tf.estimator.ModeKeys.TRAIN:
    dense = dense + x - tf.stop_gradient(x)

  def _get_losses(x, dense, dist_sq, q):
    x = tf.cast(x, tf.float32)
    dense = tf.cast(dense, tf.float32)
    dist_sq = tf.cast(dist_sq, tf.float32)
    q = tf.cast(q, tf.float32)
    disc_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - tf.stop_gradient(dense)), -1))
    em_loss = -tf.reduce_mean(tf.reduce_sum(-.5 * dist_sq * q, -1)) # M-step
    return disc_loss, em_loss
  disc_loss, em_loss = _get_losses(x, dense, dist_sq, q)

  losses = {
    "disc_loss": hparams.beta * disc_loss,
    "em_loss": hparams.gamma * em_loss,
  }
  return discrete, dense, losses


def act_fn(act_name):
  if act_name == "leaky_relu":
    return lambda x: tf.nn.leaky_relu(x, 0.4)
  elif act_name == "relu":
    return tf.nn.relu
  elif act_name == "relu6":
    return tf.nn.relu6
  else:
    raise NotImplementedError("Use one among available activations: [leaky_relu, relu, relu6]")


def discrete_bottleneck(x, hparams):
  """Simple vector quantized discrete bottleneck."""
  bottleneck_size = 2 ** hparams.bottleneck_bits
  discrete_channels = hparams.bottleneck_bits * hparams.bottleneck_dims_per_bit

  x = tf.layers.dense(x, discrete_channels)
  x_shape = commons.shape_list(x)
  x = tf.reshape(x, [-1, discrete_channels])

  discrete, dense, losses = vqvae(x, hparams)

  discrete = tf.reshape(discrete, x_shape[:-1] + [bottleneck_size])
  dense = tf.reshape(dense, x_shape[:-1] + [discrete_channels])
  return discrete, dense, losses


def mel_conditioner(x, hparams, infer=False):
  with tf.variable_scope("mel_cond") as scope:
    if not hparams.use_cond_wn:
      scope._custom_getter = commons.float32_variable_storage_getter
    if not hparams.use_vq:
      x = tf.expand_dims(x, 2)
      for s in hparams.upsample_scales:
        x = tf.layers.conv2d_transpose(
            x,
            filters=hparams.hidden_channels,
            kernel_size=(s * 4, 1),
            strides=(s, 1),
            padding="same",
            activation=act_fn(hparams.act_name))
      x = tf.squeeze(x, 2)
      losses = {}
    else:
      # Run compression by strided convs.
      e = x
      for s in hparams.upsample_scales:
        e = tf.layers.conv1d(
            e,
            hparams.hidden_channels,
            s * 2,
            padding="same",
            activation=act_fn(hparams.act_name))

      # bottleneck
      latents_discrete, latents_dense, extra_losses = discrete_bottleneck(e, hparams=hparams)

      # summary
      if not infer:
        tf.summary.image("seq_codes", tf.expand_dims(tf.cast(latents_discrete * 255, tf.uint8), -1), max_outputs=1)
        tf.summary.histogram("codes", tf.argmax(latents_discrete, -1))

      # decode
      d = latents_dense
      d = tf.expand_dims(d, 2)
      for s in hparams.upsample_scales:
        d = tf.layers.conv2d_transpose(
            d,
            filters=hparams.hidden_channels,
            kernel_size=(s * 4, 1),
            strides=(s, 1),
            padding="same",
            activation=act_fn(hparams.act_name))
      d = tf.squeeze(d, 2)

      x = d
      losses = {}
      losses.update(extra_losses)
    return x, losses


def wav_conditioner(x, hparams, infer=False):
  with tf.variable_scope("wav_cond") as scope:
    if not hparams.use_cond_wn:
      scope._custom_getter = commons.float32_variable_storage_getter
    if not hparams.use_vq:
      raise ValueError("Set use_vq as True.")
    else:
      # Run compression by strided convs.
      e = x
      for s in hparams.upsample_scales:
        e = tf.layers.conv1d(
            e,
            hparams.hidden_channels,
            s * 4,
            strides=s,
            padding="same",
            activation=act_fn(hparams.act_name))

      # bottleneck
      latents_discrete, latents_dense, extra_losses = discrete_bottleneck(e, hparams=hparams)

      # summary
      if not infer:
        tf.summary.image("seq_codes", tf.expand_dims(tf.cast(latents_discrete * 255, tf.uint8), -1), max_outputs=1)
        tf.summary.histogram("codes", tf.argmax(latents_discrete, -1))

      # decode
      d = latents_dense
      d = tf.expand_dims(d, 2)
      for s in reversed(hparams.upsample_scales):
        d = tf.layers.conv2d_transpose(
            d,
            filters=hparams.hidden_channels,
            kernel_size=(s * 4, 1),
            strides=(s, 1),
            padding="same",
            activation=act_fn(hparams.act_name))
      d = tf.squeeze(d, 2)

      x = d
      losses = {}
      losses.update(extra_losses)
    return x, losses


class Invertible1x1Conv():
  """
  The layer outputs both the convolution, and the log determinant
  of its weight matrix.  If reverse=True it does convolution with
  inverse
  """
  def __init__(self, c, dtype="float32", name=None):
    self.name = name
    with tf.variable_scope(self.name, default_name="inv1x1conv") as self.scope:
      self.W = commons.get_variable(
          name="w",
          shape=[c, c],
          initializer=tf.initializers.orthogonal(),
          dtype=dtype)

  def __call__(self, z, reverse=False):
    with tf.variable_scope(self.scope):
      # shape
      batch_size, n_of_groups, group_size = commons.shape_list(z)

      if reverse:
        if not hasattr(self, "W_inv"):
          self.W_inv = tf.cast(tf.linalg.inv(tf.cast(self.W, tf.float64)), 
              dtype=self.W.dtype.base_dtype)
        z = tf.tensordot(z, self.W_inv, [[-1], [0]])
        return z
      else:
        # Forward computation
        log_det_W = tf.cast(batch_size * n_of_groups, z.dtype.base_dtype) \
            * tf.cast(tf.math.log(tf.math.abs(tf.linalg.det(tf.cast(self.W, tf.float64)))), 
                z.dtype.base_dtype)
        z = tf.tensordot(z, self.W, [[-1], [0]])
        return z, log_det_W


class WN():
  """
  This is the WaveNet like layer for the affine coupling.  The primary difference
  from WaveNet is the convolutions need not be causal.  There is also no dilation
  size reset.  The dilation only doubles on each layer
  """
  def __init__(self, n_in_channels, n_layers, n_channels, kernel_size, global_cond=False, name=None):
    self.name = name
    with tf.variable_scope(self.name, default_name="WN") as self.scope:
      assert(kernel_size % 2 == 1)
      assert(n_channels % 2 == 0)
      self.n_layers = n_layers
      self.n_channels = n_channels
      self.in_layers = []
      self.res_skip_layers = []
      self.cond_layers = []
      if global_cond:
        self.global_layers = []

      start = tf.layers.Dense(n_channels, name="start")
      self.start = start

      # Initializing last layer to 0 makes the affine coupling layers
      # do nothing at first.  This helps with training stability
      class EndLayer:
        def __init__(self, name=None):
          self.name = name
          with tf.variable_scope(self.name, default_name="end") as self.scope:
            self.scope._custom_getter = commons.float32_variable_storage_getter
            self.layer = tf.layers.Dense(2 * n_in_channels,
                kernel_initializer=tf.initializers.zeros(),
                bias_initializer=tf.initializers.zeros(),
                name="end")
        def __call__(self, x):
          with tf.variable_scope(self.scope):
            x = self.layer(x)
            return x
      end = EndLayer(name="end")
      self.end = end

      for i in range(n_layers):
        dilation_rate = 2 ** i
        in_layer = tf.layers.Conv1D(2 * n_channels, kernel_size,
            dilation_rate=dilation_rate, padding="same", name="conv_%d" % i)
        self.in_layers.append(in_layer)

        cond_layer = tf.layers.Dense(2 * n_channels, name="cond_%d" % i)
        self.cond_layers.append(cond_layer)
        if global_cond:
          global_layer = tf.layers.Dense(2 * n_channels, name="global_%d" % i)
          self.global_layers.append(global_layer)

        # last one is not necessary
        if i < n_layers - 1:
          res_skip_channels = 2 * n_channels
        else:
          res_skip_channels = n_channels
        res_skip_layer = tf.layers.Dense(res_skip_channels, name="res_skip_%d" % i)
        self.res_skip_layers.append(res_skip_layer)

  def __call__(self, forward_input):
    with tf.variable_scope(self.scope):
      audio, spect, g_expand = forward_input
      audio = self.start(audio)

      for i in range(self.n_layers):
        in_act = self.in_layers[i](audio) + self.cond_layers[i](spect)
        if g_expand is not None:
          in_act = in_act + self.global_layers[i](g_expand)
        in_act_0, in_act_1 = tf.split(in_act, 2, -1)
        t_act = tf.nn.tanh(in_act_0)
        s_act = tf.nn.sigmoid(in_act_1)
        acts = t_act * s_act

        res_skip_acts = self.res_skip_layers[i](acts)
        if i < self.n_layers - 1:
          res_skip_acts_0, res_skip_acts_1 = tf.split(res_skip_acts, 2, -1)
          audio = res_skip_acts_0 + audio
          skip_acts = res_skip_acts_1
        else:
          skip_acts = res_skip_acts

        if i == 0:
          output = skip_acts
        else:
          output = skip_acts + output
      output = self.end(output)
      return output
            

class WaveGlow():
  def __init__(self, hparams, mode):
    with tf.variable_scope("WaveGlow") as self.scope:
      self.hparams = copy.copy(hparams)
      self.hparams.mode = mode
      if self.hparams.mode != tf.estimator.ModeKeys.TRAIN:
        # remove dropouts if not training
        for key in self.hparams.keys():
          if key.endswith("dropout"):
            setattr(self.hparams, key, 0.0)

      if self.hparams.use_vq:
        # lookup tables
        means = init_discrete_bottleneck(
            self.hparams.bottleneck_bits, self.hparams.bottleneck_dims_per_bit,
            dtype=self.hparams.ftype)
        self.hparams.means = means

      assert(self.hparams.n_group % 2 == 0)
      self.WN = []
      self.convinv = []
      self.film = []

      class FiLMLayer:
        def __init__(self, n_channels, global_cond=None, name=None):
          self.name = name
          self.n_channels = n_channels
          with tf.variable_scope(self.name, default_name="FiLM") as self.scope:
            self.scope._custom_getter = commons.float32_variable_storage_getter
            self.cond_layer = tf.layers.Conv1D(2 * n_channels, 3,
                padding="same",
                kernel_initializer=tf.initializers.zeros(),
                bias_initializer=tf.initializers.zeros(),
                activation=act_fn(hparams.act_name))
            if global_cond is not None:
              self.global_layer = tf.layers.Conv1D(2 * n_channels, 1,
                  padding="same",
                  kernel_initializer=tf.initializers.zeros(),
                  bias_initializer=tf.initializers.zeros(),
                  activation=act_fn(hparams.act_name))
        def __call__(self, forward_input):
          c, g = forward_input
          with tf.variable_scope(self.scope):
            x = self.cond_layer(c)
            if g is not None:
              x = x + self.global_layer(g)
            return x

      # Set up layers with the right sizes based on how many dimensions
      # have been output already
      n_remaining_channels = self.hparams.n_group
      for k in range(self.hparams.n_flows):
          if k % self.hparams.n_early_every == 0 and k > 0:
              n_remaining_channels = n_remaining_channels - self.hparams.n_early_size
          self.convinv.append(Invertible1x1Conv(n_remaining_channels, 
                                                dtype=self.hparams.ftype,
                                                name="inv1x1conv_%d" % k))
          self.WN.append(WN(n_remaining_channels // 2, self.hparams.n_layers,
              self.hparams.n_channels, self.hparams.kernel_size, self.hparams.global_condition is not None,
              name="WN_%d" % k))
          self.film.append(FiLMLayer(n_remaining_channels, self.hparams.global_condition is not None, name="FiLM_%d" % k))
      self.n_remaining_channels = n_remaining_channels  # Useful during inference

  def body(self, features):
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      hparams = self.hparams
      x, c, g = features["x"], features["c"], features["g"]
      c_expand, extra_losses = self.local_conditioner(c)
      g_expand = self.global_conditioner(g)

      x_shape = commons.shape_list(x)
      c_shape = commons.shape_list(c_expand)
      x_group = tf.reshape(x, 
          [x_shape[0], x_shape[1] // hparams.n_group, x_shape[2] * hparams.n_group])
      c_group = tf.reshape(c_expand, 
          [c_shape[0], c_shape[1] // hparams.n_group, c_shape[2] * hparams.n_group])

      audio = x_group
      spect = c_group
      output_audio = []
      log_s_list = []
      log_det_W_list = []

      for k in range(hparams.n_flows):
        if k % hparams.n_early_every == 0 and k > 0:
          _output_audio, audio = tf.split(audio, [hparams.n_early_size, -1], -1)
          output_audio.append(_output_audio)
        audio, log_det_W = self.convinv[k](audio)
        log_det_W_list.append(log_det_W)

        audio_0, audio_1 = tf.split(audio, 2, -1)

        output = self.WN[k]((audio_0, spect, g_expand))
        log_s, b = tf.split(output, 2, -1)
        audio_1 = audio_1 * tf.math.exp(log_s) + b
        log_s_list.append(log_s)

        audio = tf.concat([audio_0, audio_1], 2)
        if hparams.use_film:
          log_gamma, beta = tf.split(self.film[k]((spect, g_expand)), 2, -1)
          audio = audio * tf.math.exp(log_gamma) + beta
          log_s_list.append(log_gamma)

      output_audio.append(audio)
      outputs = (tf.concat(output_audio, 2), log_s_list, log_det_W_list)

      losses = {}
      losses.update(extra_losses)
      return outputs, losses

  def infer(self, features, sigma=1.0):
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      hparams = self.hparams
      c, g = features["c"], features["g"]
      c_expand, extra_losses = self.local_conditioner(c, infer=True)
      g_expand = self.global_conditioner(g)

      c_shape = commons.shape_list(c_expand)
      c_group = tf.reshape(c_expand, 
          [c_shape[0], c_shape[1] // hparams.n_group, c_shape[2] * hparams.n_group])

      spect = c_group
      audio = tf.random.normal(
          shape=[c_shape[0], c_shape[1] // hparams.n_group, self.n_remaining_channels],
          dtype=hparams.ftype)
      audio *= sigma

      for k in reversed(range(hparams.n_flows)):
        if hparams.use_film:
          log_gamma, beta = tf.split(self.film[k]((spect, g_expand)), 2, -1)
          audio = (audio - beta) * tf.math.exp(-log_gamma)
        audio_0, audio_1 = tf.split(audio, 2, -1)

        output = self.WN[k]((audio_0, spect, g_expand))
        log_s, b = tf.split(output, 2, -1)
        audio_1 = (audio_1 - b) * tf.math.exp(-log_s)
        audio = tf.concat([audio_0, audio_1], 2)

        audio = self.convinv[k](audio, reverse=True)

        if k % hparams.n_early_every == 0 and k > 0:
          z = tf.random.normal(
              shape=[c_shape[0], c_shape[1] // hparams.n_group, hparams.n_early_size],
              dtype=hparams.ftype)
          audio = tf.concat([z * sigma, audio], 2)
      audio = tf.reshape(audio, [c_shape[0], -1, 1])
      return audio

  def local_conditioner(self, c, infer=False):
    hparams = self.hparams
    if hparams.local_condition == "mel":
      return mel_conditioner(c, hparams, infer=infer)
    else:
      return wav_conditioner(c, hparams, infer=infer)

  def global_conditioner(self, g):
    hparams = self.hparams
    if hparams.global_condition == None:
      return None
    else:
      with tf.variable_scope("global_cond"):
        s_emb = commons.get_variable(
            name="speaker_emb",
            shape=[hparams.n_speakers, hparams.emb_channels],
            dtype=hparams.ftype)
        x = tf.nn.embedding_lookup(s_emb, g) # [b, dim]
        x = tf.expand_dims(x, 1) # [b, 1, dim]
        return x


def compute_waveglow_loss(model_output, sigma=1.0):
  with tf.name_scope("compute_loss"):
    z, log_s_list, log_det_W_list = model_output
    # mixed precision training support
    if z.dtype.base_dtype != tf.float32:
      z = tf.cast(z, dtype=tf.float32)
      log_s_list = [tf.cast(x, tf.float32) for x in log_s_list]
      log_det_W_list = [tf.cast(x, tf.float32) for x in log_det_W_list]

    for i, log_s in enumerate(log_s_list):
      if i == 0:
        log_s_total = tf.reduce_sum(log_s)
      else:
        log_s_total += tf.reduce_sum(log_s)

    for i, log_det_W in enumerate(log_det_W_list):
      if i == 0:
        log_det_W_total = log_det_W
      else:
        log_det_W_total += log_det_W

    loss = tf.reduce_sum(z * z) / (2. * tf.math.square(sigma))
    loss -= log_s_total
    loss -= log_det_W_total
    return loss / tf.cast(tf.reduce_prod(commons.shape_list(z)), loss.dtype.base_dtype)


def build_model_fn(hparams):
  def model_fn(features, labels, mode):
    with tf.variable_scope("model", 
        custom_getter=commons.weight_norm_getter,
        initializer=tf.initializers.glorot_uniform()):
      # Input Preparation
      x_org = features["wav"]
      x = features["wav"]
      c = features[hparams["local_condition"]]
      g = features[hparams["global_condition"]] if hparams["global_condition"] is not None else None
      
      # type casting
      if hparams.ftype != tf.float32:
        x = tf.cast(x, hparams.ftype)
        if c is not None:
          c = tf.cast(c, hparams.ftype)

      model = WaveGlow(hparams, mode)
      # decode
      if mode != tf.estimator.ModeKeys.PREDICT:
        outputs, losses = model.body(features={"x": x, "c": c, "g": g})
        losses["neglogp"] = compute_waveglow_loss(outputs, sigma=hparams.sigma)

        if mode == tf.estimator.ModeKeys.TRAIN:
          predictions = None
        else:
          predictions = model.infer(features={"c": c, "g": g}, sigma=0.9)

        # losses
        loss = 0.
        for k, l in losses.items():
          tf.summary.scalar(k, l)
          loss += l
      else:
        predictions = model.infer(features={"c": c, "g": g}, sigma=hparams.sigma)
        loss = None
  
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op, saver = commons.get_train_op(loss, hparams)
    else:
      train_op, saver = None, None
  
    if mode == tf.estimator.ModeKeys.EVAL:
      #tf.summary.audio("org", features["wav"], hparams.sample_rate, max_outputs=1)
      tf.summary.audio("gen", tf.cast(predictions, tf.float32), hparams.sample_rate, max_outputs=1)

      eval_metrics = None

      eval_summary_hook = tf.train.SummarySaverHook(
          save_steps=1,
          output_dir= os.path.join(hparams.model_dir, "eval"),
          summary_op=tf.summary.merge_all())
      eval_summary_hooks = [eval_summary_hook]
    else:
      eval_metrics = None
      eval_summary_hooks = None
  
    return tf.estimator.EstimatorSpec(
        mode,
        predictions=predictions,
        loss=loss,
        eval_metric_ops=eval_metrics,
        evaluation_hooks=eval_summary_hooks,
        scaffold=tf.train.Scaffold(saver=saver),
        train_op=train_op)
  return model_fn
