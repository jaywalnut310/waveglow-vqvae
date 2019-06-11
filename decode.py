import argparse
import os
import time
from scipy.io.wavfile import write

import tensorflow as tf
import numpy as np

from hparams import create_hparams
import data
import utils
import commons
import waveglow as model


def main():
  parser = utils.get_argument_parser("Decode by using the trained model")
  parser.add_argument("--checkpoint", dest="checkpoint", help="Path to a checkpoint file. Default is the latest version.")
  parser.add_argument("--limit", type=int, default=1, help="The number of sentences to be decoded. (0=unlimited)")
  parser.add_argument("--use_eval", action="store_true", help="Use evaluation dataset for prediction")
  parser.add_argument("--predict_dir", type=str, default="", help="Path to a local condition file dir")
  parser.add_argument("--out_dir", type=str, default="", help="Path to a wav file dir to write")
  args = parser.parse_args()

  hparams = create_hparams(args.model_dir, args.configs, initialize=False)
  utils.check_git_hash(args.model_dir)

  if not os.path.isdir(args.out_dir):
    os.mkdir(os.path.relpath(args.out_dir))

  data.load_vocab(hparams)
  if args.use_eval:
    input_fn = data.InputPipeline(hparams, tf.estimator.ModeKeys.EVAL)
  else:
    raise NotImplementedError("File to mel or wav is not avaliable now.")

  estimator = tf.estimator(model_fn=model.build_model_fn(hparams), model_dir=args.model_dir)

  for i, prediction in enumerate(estimator.predict(input_fn, checkpoint_path=args.checkpoint)):
    for j, wav in predictions.tolist():
      wav = wav.astype(np.float32)
      write(os.path.join(args.out_dir, "{}.wav".format(i*hparams.infer_batch_size + i)), hparams.sample_rate, wav)

    if args.limit and i + 1 == args.limit:
      break
