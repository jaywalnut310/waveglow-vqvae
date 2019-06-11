import math
import numpy as np
import tensorflow as tf
from tensorflow.python.training import distribution_strategy_context
from tensorflow.python.training import slot_creator


"""
Minimized tensor2tensor utils.
Almost codes are drawn from https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor
"""

#== UTILS ==
def float32_variable_storage_getter(getter, name, shape=None, dtype=None,
                                    initializer=None, regularizer=None,
                                    trainable=True,
                                    *args, **kwargs):
  """Custom variable getter that forces trainable variables to be stored in
  float32 precision and then casts them to the training precision.
  """
  storage_dtype = tf.float32 if trainable else dtype
  variable = getter(name, shape, dtype=storage_dtype,
                    initializer=initializer, regularizer=regularizer,
                    trainable=trainable,
                    *args, **kwargs)
  if trainable and dtype != tf.float32:
    variable = tf.cast(variable, dtype)
  return variable

def weight_norm_getter(getter, name, shape=None, dtype=None,
                       initializer=None, regularizer=None,
                       trainable=True,
                       *args, **kwargs):
  if name.endswith("kernel"):
    dims = list(range(len(shape) - 1))
    v = float32_variable_storage_getter(getter, name + "_v", shape, 
        dtype=dtype, initializer=initializer, regularizer=regularizer,
        trainable=trainable, *args, **kwargs)
    v = tf.nn.l2_normalize(v, dims)
    g = float32_variable_storage_getter(getter, name + "_g", (shape[-1],), 
        dtype=dtype, initializer=initializer, regularizer=regularizer,
        trainable=trainable, *args, **kwargs)
    return g * v
  else:
    return float32_variable_storage_getter(getter, name, shape, 
        dtype=dtype, initializer=initializer, regularizer=regularizer,
        trainable=trainable, *args, **kwargs)


def get_variable(name,
                 shape=None,
                 dtype=tf.float32,
                 initializer=None,
                 regularizer=None,
                 trainable=True,
                 weight_norm=False,
                 *args,
                 **kwargs):
  storage_dtype = tf.float32 if trainable else dtype
  if weight_norm:
    dims = list(range(len(shape) - 1))
    v = tf.get_variable(name + "_v", shape, dtype=storage_dtype,
                             initializer=initializer, regularizer=regularizer,
                             trainable=trainable,
                             *args, **kwargs)
    v = tf.nn.l2_normalize(v, dims)
    g = tf.get_variable(name + "_g", (shape[-1],), dtype=storage_dtype,
                             initializer=initializer, regularizer=regularizer,
                             trainable=trainable,
                             *args, **kwargs)
    variable = g * v
  else:
    variable = tf.get_variable(name, shape, dtype=storage_dtype,
                             initializer=initializer, regularizer=regularizer,
                             trainable=trainable,
                             *args, **kwargs)
  if trainable and dtype != tf.float32:
    variable = tf.cast(variable, dtype)
  return variable


def shape_list(x):
  """Shape list"""
  x_shape = tf.shape(x)
  x_get_shape = x.get_shape().as_list()

  res = []
  for i, d in enumerate(x_get_shape):
    if d is not None:
      res.append(d)
    else:
      res.append(x_shape[i])
  return res


def get_optimizer_fn(hparams):
  if hparams.optimizer == "adam":
    return lambda lr: tf.train.AdamOptimizer(
          lr, hparams.adam_beta1, hparams.adam_beta2, hparams.adam_epsilon)
  elif hparams.optimizer == "adamax":
    return lambda lr: tf.contrib.opt.AdaMaxOptimizer(
          lr, hparams.adam_beta1, hparams.adam_beta2, hparams.adam_epsilon)
  elif hparams.optimizer == "sgd":
    return lambda lr: tf.train.GradientDescentOptimizer(lr)

  else:
    raise ValueError("Unknown optimizer: {}".format(hparams.optimizer))


def noam_learning_rate_decay(learning_rate, global_step, hparams):
  step = tf.to_float(global_step)
  return learning_rate * hparams.hidden_size**-0.5 * tf.minimum(
      (step + 1) * hparams.warmup_step**-1.5, (step + 1)**-0.5)


def halve_learning_rate_decay(learning_rate, global_step, hparams):
  step = tf.to_float(global_step)
  ratio = 2**(-1. * (step // hparams.halve_step))
  return tf.maximum(hparams.min_lr, learning_rate * ratio)


def get_learning_rate_decay_fn(hparams):
  if hparams.lr_decay is None:
    return lambda lr, step: lr

  elif hparams.lr_decay == "noam":
    return lambda lr, step: noam_learning_rate_decay(lr, step, hparams)
  elif hparams.lr_decay == "halve":
    return lambda lr, step: halve_learning_rate_decay(lr, step, hparams)
  else:
    raise ValueError("Unknown learing rate decay method: {}".format(hparams.lr_decay))


def get_train_op(loss, hparams, name="train"):
  # 0. summary
  summaries = ["loss", "learning_rate", "global_gradient_norm"]
  global_step = tf.train.get_global_step()

  with tf.variable_scope(name, "OptimizeLoss", [loss, global_step]):
    # 1. get learning rate
    lr = get_learning_rate_decay_fn(hparams)(hparams.learning_rate, global_step)
    tf.summary.scalar("learning_rate", lr)

    # 2. create optimizer
    opt = get_optimizer_fn(hparams)(lr)
    if hparams.exponential_moving_average:
      opt = MovingAverageOptimizer(opt, hparams.ema_decay)

    # 3. multiply scalar to loss
    loss_scale = float(hparams.loss_scale)
    inv_loss_scale = tf.math.reciprocal(loss_scale)
    variables = tf.trainable_variables()
    gradients = opt.compute_gradients(
        loss * loss_scale,
        variables)

    num_finite = []
    num_grads = []
    gv = []
    for g, v in gradients:
      if g is not None:
        g_f = tf.is_finite(g)
        g = tf.where(g_f,
            g * tf.cast(inv_loss_scale, g.dtype.base_dtype),
            tf.zeros_like(g))
        num_finite.append(tf.reduce_sum(tf.to_float(g_f)))
        num_grads.append(tf.reduce_prod(g.get_shape()))
        gv.append((g, v))
      else:
        print("Untrained Trainable Variable: ", v.name)
    tf.summary.scalar("finite_grad_ratio", tf.reduce_sum(num_finite) / tf.to_float(tf.reduce_sum(num_grads)))
    gradients = gv

    tf.summary.scalar("global_norm/gradient_norm",
        tf.global_norm(list(zip(*gradients))[0]))

    # 4. clipping gradients
    if hparams.clip_gradients is not None:
      gs, vs = zip(*gradients)
      gn_inv = tf.rsqrt(tf.reduce_sum([tf.reduce_sum(tf.square(g)) for g in gs]) + 1e-8)
      gs = [g * gn_inv * hparams.clip_gradients for g in gs]
      gradients = list(zip(gs, vs))

    grad_updates = opt.apply_gradients(
        gradients,
        global_step=global_step,
        name="train")

    train_op = grad_updates
    if hparams.exponential_moving_average:
      saver = opt.swapping_saver()
    else:
      saver = None
  return train_op, saver


def assign_moving_average(variable, value, decay, zero_debias=True, name=None):
  """Compute the moving average of a variable.
  https://github.com/tensorflow/tensorflow/blob/c966b5eed60a570f2121cb84ddb4ece84c413719/tensorflow/python/training/moving_averages.py
  """

  def _zero_debias(unbiased_var, value, decay):
    """Compute the delta required for a debiased Variable.
    """
    with tf.variable_scope(
        unbiased_var.op.name, values=[unbiased_var, value, decay]) as scope:
      with tf.init_scope():
        biased_initializer = tf.zeros_initializer(
            dtype=unbiased_var.dtype)(unbiased_var.get_shape())
        local_step_initializer = tf.zeros_initializer()
      def _maybe_get_unique(name):
        """Get name for a unique variable, if not `reuse=True`."""
        if tf.get_variable_scope().reuse:
          return name
        vs_vars = [x.op.name for x in
                   tf.get_variable_scope().global_variables()]
        full_name = tf.get_variable_scope().name + "/" + name
        if full_name not in vs_vars: return name
        idx = 1
        while full_name + ("_%d" % idx) in vs_vars:
          idx += 1
        return name + ("_%d" % idx)
      biased_var = tf.get_variable(
          _maybe_get_unique("biased"), initializer=biased_initializer,
          trainable=False)
      local_step = tf.get_variable(
          _maybe_get_unique("local_step"),
          shape=[],
          dtype=unbiased_var.dtype,
          initializer=local_step_initializer,
          trainable=False)

      # Get an update ops for both shadow variables.
      update_biased = tf.assign_sub(biased_var,
                                           (biased_var - value) * decay,
                                           name=scope.name)
      update_local_step = local_step.assign_add(1)

      # Compute the value of the delta to update the unbiased EMA. Make sure to
      # use the new values of the biased variable and the local step.
      with tf.control_dependencies([update_biased, update_local_step]):
        # This function gets `1 - decay`, so use `1.0 - decay` in the exponent.
        unbiased_ema_delta = (unbiased_var - biased_var.read_value() /
                              (1 - tf.pow(
                                  1.0 - decay, local_step.read_value())))

      return unbiased_ema_delta

  def update_fn(v, value, decay=decay):
    decay = tf.convert_to_tensor(1.0 - decay, name="decay")
    if decay.dtype != v.dtype.base_dtype:
      decay = tf.cast(decay, v.dtype.base_dtype)
    if zero_debias:
      update_delta = _zero_debias(v, value, decay)
    else:
      update_delta = (v - value) * decay
    return tf.assign_sub(v, update_delta, name=scope)

  with tf.name_scope(name, "AssignMovingAvg",
                      [variable, value, decay]) as scope:
    tower_context = distribution_strategy_context.get_tower_context()
    if tower_context:
      # In a tower context, we update variable using the mean of value across
      # towers.
      def merge_fn(strategy, v, value):
        try:
          value = strategy.reduce(
              tf.VariableAggregation.MEAN, value, v)
        except:
          pass # Mirrored variables are loaded
        return strategy.update(v, update_fn, value)

      return tower_context.merge_call(merge_fn, variable, value)
    else:
      strategy = distribution_strategy_context.get_cross_tower_context()
      return strategy.update(variable, update_fn, value)


class ExponentialMovingAverage(object):
  """Maintains moving averages of variables by employing an exponential decay.
  """

  def __init__(self, decay, num_updates=None, zero_debias=False,
               name="ExponentialMovingAverage"):
    """Creates a new ExponentialMovingAverage object.
    """
    self._decay = decay
    self._num_updates = num_updates
    self._zero_debias = zero_debias
    self._name = name
    self._averages = {}

  @property
  def name(self):
    """The name of this ExponentialMovingAverage object."""
    return self._name

  def apply(self, var_list=None):
    """Maintains moving averages of variables.
    """
    # TODO(touts): op_scope
    if var_list is None:
      var_list = tf.trainable_variables()
    zero_debias_true = set()  # set of vars to set `zero_debias=True`

    def _create_slots(var_list):
      for var in var_list:
        if var.dtype.base_dtype not in [
            tf.bfloat16, tf.float16, tf.float32, tf.float64
        ]:
          raise TypeError("The variables must be half, float, or double: %s" %
                          var.name)

        if var not in self._averages:
          # For variables: to lower communication bandwidth across devices we keep
          # the moving averages on the same device as the variables. For other
          # tensors, we rely on the existing device allocation mechanism.
          with tf.init_scope():
            try:
              prefix = var._primary_var.op.name
            except:
              prefix = var.op.name
            
            with tf.variable_scope(None, prefix + "/" + self.name):
              if isinstance(var, tf.Variable):
                avg = tf.get_variable("",
                    initailizer=var.initialized_value(),
                    trainable=False)
                # NOTE(mrry): We only add `tf.Variable` objects to the
                # `MOVING_AVERAGE_VARIABLES` collection.
                tf.add_to_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES, var)
              else:
                avg = tf.get_variable("",
                    initializer=tf.zeros_initializer(),
                    shape=var.get_shape(),
                    dtype=var.dtype)
                if self._zero_debias:
                  zero_debias_true.add(avg)
          self._averages[var] = avg

    _create_slots(var_list)

    with tf.name_scope(self.name) as scope:
      decay = tf.convert_to_tensor(self._decay, name="decay")
      if self._num_updates is not None:
        num_updates = tf.cast(self._num_updates,
                                    tf.float32,
                                    name="num_updates")
        decay = tf.minimum(decay, (1.0 + num_updates) / (10.0 + num_updates))
      updates = []
      for var in var_list:
        zero_debias = self._averages[var] in zero_debias_true
        updates.append(assign_moving_average(
            self._averages[var], var, decay, zero_debias=zero_debias))
        break
      return tf.group(*updates, name=scope)

  def average(self, var):
    """Returns the `Variable` holding the average of `var`.
    """
    return self._averages.get(var, None)


class MovingAverageOptimizer(tf.train.Optimizer):
  def __init__(self, opt, average_decay=0.9999, num_updates=None,
               sequential_update=True):
    """Construct a new MovingAverageOptimizer.
    """
    self._optimizer = opt
    self._ema = ExponentialMovingAverage(
        average_decay, num_updates=num_updates, zero_debias=True)
    self._swapped_variable_name_map = None
    self._sequential_update = sequential_update

  def compute_gradients(self, *args, **kwargs):
    return self._optimizer.compute_gradients(*args, **kwargs)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    train_op = self._optimizer.apply_gradients(
        grads_and_vars, global_step=global_step, name=name)
    var_list = [x[1] for x in grads_and_vars if x[0] is not None]
    self._swapped_variable_name_map = {}
    if self._sequential_update:
      with tf.control_dependencies([train_op]):
        ma_op = self._ema.apply(var_list)
    else:
      ma_op = self._ema.apply(var_list)

    for v in var_list:
      v_avg = self._ema.average(v)
      self._swapped_variable_name_map[v.op.name] = v_avg.op.name
      self._swapped_variable_name_map[v_avg.op.name] = v.op.name
    return tf.group(train_op, ma_op, name='train_with_avg')

  def swapping_saver(self, var_list=None, name='swapping_saver', **kwargs):
    """Create a saver swapping moving averages and variables.
    """
    if self._swapped_variable_name_map is None:
      raise RuntimeError('Must call apply_gradients or minimize before '
                         'creating the swapping_saver')
    if var_list is None:
      var_list = tf.global_variables()
    v_name_to_tensor = {v.op.name: v for v in var_list}

    # Now swap variables and moving averages
    swapped_var_list = {}
    for v_name, v in v_name_to_tensor.items():
      swapped_v_name = self._swapped_variable_name_map.get(v_name, None)
      v_to_save = v
      if swapped_v_name is not None:
        if swapped_v_name in v_name_to_tensor:
          v = v_name_to_tensor[swapped_v_name]
        else:
          raise ValueError(
              ('Variable to swap %s is not part of variables to save. '
               'This breaks MovingAverageOptimizer.') % swapped_v_name)
      swapped_var_list[v_name] = v

    # Build the swapping saver.
    return tf.train.Saver(swapped_var_list, name=name, **kwargs)
