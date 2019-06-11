import os
import yaml
import tensorflow as _tf


_config_file_name = "config.yml"
_source_dir = os.path.dirname(os.path.realpath(__file__))


class Hparams():
  """Dummy class"""
  def keys(self):
    return self.__dict__.keys()

  def items(self):
    return self.__dict__.items()

  def values(self):
    return self.__dict__.values()

  def __len__(self):
    return len(self.__dict__)

  def __getitem__(self, key):
    return getattr(self, key)

  def __setitem__(self, key, value):
    return setattr(self, key, value)

  def __contains__(self, key):
    return key in self.__dict__

  def __repr__(self):
    return self.__dict__.__repr__()


def _update(d, key, value):
  if key in d and d[key] != value:
    _tf.logging.info("The value of '{}' is '{}', but is overwritten by '{}'.".format(key, d[key], value))
  d[key] = value


def _save(model_dir, d):
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)

  config_yml = yaml.dump(d, default_flow_style=False)
  with open(os.path.join(model_dir, _config_file_name), "w+") as f:
    f.write(config_yml)


def _print(d):
  _tf.logging.info("------------------- All configurations --------------------")
  for k, v in d.items():
    _tf.logging.info("    %s = %s", k, v)
  _tf.logging.info("------------------------------------------------------------")

def _parse_value(v):
  try:
     v = int(v)
  except ValueError:
    try:
      v = float(v)
    except ValueError:
      if len(v.split(",")) > 1:
        v = [_parse_value(x.strip()) for x in v.split(",") if x != ""]
      else:
        v_norm = v.lower().strip()
        if v_norm == "true":
          v = True
        elif v_norm == "false":
          v = False
        elif v_norm == "null":
          v = None
  return v

def import_configs(hparams, configs):
  configs = [os.path.join(_source_dir, _config_file_name)] + configs
  for cfg in configs:
    kv = [s.strip() for s in cfg.split("=", 1)]
    if len(kv) == 1:
      if not os.path.exists(cfg):
        raise ValueError("The configuration file doesn't exist; {}".format(cfg))
      obj = yaml.load(open(cfg).read())
      for k, v in obj.items():
        _update(hparams, k, v)
    else:
      k, v = kv
      v = _parse_value(v)
      _update(hparams, k, v)


def create_hparams(model_dir, configs, initialize=False, print=True):
  saved_config_file = os.path.join(model_dir, _config_file_name)

  if not os.path.isdir(model_dir):
    os.mkdir(os.path.relpath(model_dir))
  if os.path.exists(saved_config_file):
    configs = [saved_config_file] + configs

  hparams = Hparams()
  import_configs(hparams, configs) 
  _update(hparams, "model_dir", model_dir)

  if not os.path.exists(saved_config_file) and initialize:
    _save(model_dir, hparams)

  if print:
    _print(hparams)

  return hparams
