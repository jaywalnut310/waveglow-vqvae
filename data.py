import os
import glob
import tensorflow as tf


def load_vocab(hparams):
  if hparams.vocab_file is None:
    return
  raise NotImplementedError("Text Input is not valid currently.")


class InputPipeline:
  def __init__(self, hparams, mode):
    self.hparams = hparams
    self.mode = mode

  def __call__(self):
    hparams = self.hparams
    with tf.name_scope("input_pipeline"):
      if self.mode != tf.estimator.ModeKeys.PREDICT:
        record_files = os.path.join(hparams.tfr_dir, 
            "%s_%s_*_of_*.tfrecord" % (hparams.tfr_prefix, self.mode))
        record_files = glob.glob(record_files)
        dataset = tf.data.TFRecordDataset(record_files)
        if self.mode == tf.estimator.ModeKeys.TRAIN:
          dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(2**10))
        dataset = dataset.map(self._parse, num_parallel_calls=8)
        dataset = dataset.batch(hparams["{}_batch_size".format(self.mode)])
        dataset = dataset.prefetch(buffer_size=8)
        
        return dataset
      else:
        raise NotImplementedError("Prediction mode is not implemented.")

  def _parse(self, example):
    hparams = self.hparams

    # feature template
    feature = {}
    for k in hparams.load_features:
      if k in ["wav", "mel"]:
        feature[k] = tf.FixedLenFeature([], tf.string)
      elif k == "txt":
        feature[k] = tf.VarLenFeature(tf.int64)
      else:
        feature[k] = tf.FixedLenFeature([], tf.int64)

    #parse
    features = tf.parse_single_example(example, features=feature)

    # sparse to dense for text
    if features.get("txt") is not None:
      features["txt"] = tf.sparse.to_dense(features["txt"])

    assert hparams.max_input_length % hparams.hop_size == 0, \
        "max_input_length should be a multiple of hop_size"
    # reshape
    wav = tf.decode_raw(features["wav"], tf.float32)
    wav = tf.expand_dims(wav, -1)
    features["wav"] = wav # assign
    wav_len = tf.size(wav)
    if features.get("mel") is not None:
      mel = tf.decode_raw(features["mel"], tf.float32)
      mel = tf.reshape(mel, [wav_len // hparams.hop_size, hparams.mel_channels])
      features["mel"] = mel # assign

    if self.mode == tf.estimator.ModeKeys.TRAIN:
      # wave segment
      str_idx_mel = tf.random_uniform((), 0, wav_len-hparams.max_input_length, tf.int32) // hparams.hop_size
      str_idx_wav = str_idx_mel * hparams.hop_size
      seg_len_wav = hparams.max_input_length
      wav_seg = wav[str_idx_wav:str_idx_wav + seg_len_wav]
      features["wav"] = wav_seg

      # mel segment
      if features.get("mel") is not None:
        seg_len_mel = hparams.max_input_length // hparams.hop_size
        mel_seg = mel[str_idx_mel:str_idx_mel + seg_len_mel]
        features["mel"] = mel_seg

    return features, {}


if __name__ == "__main__":
  import time
  import argparse
  from hparams import Hparams, import_configs
  
  parser = argparse.ArgumentParser("Load datasets")
  parser.add_argument("-c", "--conf", dest="configs", default=[], nargs="*",
            help="A list of configuration items. "
                 "An item is a file path or a 'key=value' formatted string. "
                 "The type of a value is determined by applying int(), float(), and str() "
                 "to it sequencially.")
  args = parser.parse_args()

  hparams = Hparams()
  import_configs(hparams, args.configs)

  load_vocab(hparams)

  # Save tfrecords
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.enable_eager_execution()

  ds = InputPipeline(hparams, tf.estimator.ModeKeys.TRAIN)()
  itr = ds.make_one_shot_iterator()

  itr.get_next()
  t_str = time.time()
  for i in range(1000):
    itr.get_next()
  print(time.time() - t_str)
