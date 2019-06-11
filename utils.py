import os
import argparse
import logging
import subprocess

import tensorflow as tf


def get_argument_parser(description=None):
  parser = argparse.ArgumentParser(description)
  parser.add_argument("-m", "--model_dir", type=str, required=True,
      help="The directory for a trained model is saved.")
  parser.add_argument("-c", "--conf", dest="configs", default=[], nargs="*",
      help="A list of configuration items. "
           "An item is a file path or a 'key=value' formatted string. "
           "The type of a value is determined by applying int(), float(), and str() "
           "to it sequencially.")
  return parser


def parse_args(description=None):
  parser = get_argument_parser(description)
  args = parser.parse_args()
  return args


def check_git_hash(model_dir):
  source_dir = os.path.dirname(os.path.realpath(__file__))
  if not os.path.exists(os.path.join(source_dir, ".git")):
      tf.logging.warn("{} is not a git repository, therefore hash value comparison will be ignored.")
      return

  cur_hash = subprocess.getoutput("git rev-parse HEAD")

  path = os.path.join(model_dir, "githash")
  if os.path.exists(path):
    saved_hash = open(path).read()
    if saved_hash != cur_hash:
      tf.logging.warn("git hash values are different. {}(saved) != {}(current)".format(
          saved_hash[:8], cur_hash[:8]))
  else:
    open(path, "w").write(cur_hash)


def redirect_log_to_file(model_dir, filename="train.log"):
  logger = logging.getLogger("tensorflow")

  formatter = logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s")

  if not os.path.exists(model_dir):
    os.makedirs(model_dir)

  h = logging.FileHandler(os.path.join(model_dir, "train.log"))
  h.setLevel(logging.DEBUG)
  h.setFormatter(formatter)
  logger.addHandler(h)
