import os
import math
from typing import Iterable
from scipy.io.wavfile import read
import numpy as np

import tensorflow as tf

import data

"""
Generate audio datasets
"""
_NUM_SHARDS = {
  "train": 4,
  "eval": 1,
}


def _int64_feature(value):
  if not isinstance(value, Iterable):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  if not isinstance(value, Iterable):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def save_tfrecords(save_dir, save_prefix, data, wav_load_fn, txt_encode_fn, n_shards):
  """write to TFRecords
  """
  n_data = math.ceil(len(data) / n_shards)
  for idx_shard in range(n_shards):
    tf.logging.info("Save shards: %d of %d" % (idx_shard, n_shards))
    save_path = os.path.join(save_dir, save_prefix + ("_%04d_of_%04d.tfrecord" % (idx_shard, n_shards)))
    with tf.python_io.TFRecordWriter(save_path) as writer:
      for fname, txt, sid in data[n_data*idx_shard:n_data*(idx_shard + 1)]:
        # load wav
        wav, mel = wav_load_fn(fname)
        # create a feature
        feature = {"wav": _bytes_feature(tf.compat.as_bytes(wav.tobytes())),
                   "mel": _bytes_feature(tf.compat.as_bytes(mel.tobytes()))}
        if txt:
          feature["txt"] = _int64_feature(txt_encode_fn(txt))
        if sid:
          feature["sid"] = _int64_feature(sid)

        # create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # serialize to string and write on the file
        writer.write(example.SerializeToString())


def load_wav(fname, hparams):
  sample_rate, audio = read(fname)
  if sample_rate != hparams.sample_rate: 
    raise ValueError("source {} SR doesn't match target {} SR".format(
        sample_rate, hparams.sample_rate))

  audio_norm = audio / hparams.max_wav_value

  # pad audio
  pad_len = hparams.hop_size * math.ceil(len(audio_norm) / hparams.hop_size) - len(audio_norm)
  audio_norm = np.pad(audio_norm, ((0, pad_len)), "constant").astype("float32")

  # Need tf eager execution
  signal = tf.reshape(audio_norm, [1, -1])
  stft = tf.contrib.signal.stft(signal,
      frame_length=hparams.fft_size,
	    frame_step=hparams.hop_size,
	    fft_length=hparams.fft_size,
      pad_end=True)
  magnitude = tf.abs(stft)

  linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
      hparams.mel_channels, 
      magnitude.shape[-1].value,
      hparams.sample_rate, 
      hparams.fmin,
      hparams.fmax)
  
  mel = tf.tensordot(magnitude, linear_to_mel_weight_matrix, 1)
  mel = tf.log(tf.maximum(mel, 1e-5)) # log scaling with clamping
  mel = mel.numpy()[0] # For visualisation, use mel.T[::-1]

  assert len(audio_norm) / hparams.hop_size == len(mel), \
      "Wave length {} is different from mel length * hop size {} * {}".format( \
          len(audio_norm), len(mel), hparams.hop_size)

  return audio_norm, mel


def encode_txt(s, vocab):
  if vocab is None:
    return None
  return vocab.encode(s.strip())


def get_value_from_query(query, keys, values):
  if query not in keys:
    return None
  idx = keys.index(query)
  return values[idx]


def audio2tfr(hparams):
  wav_load_fn = lambda x: load_wav(x, hparams)
  txt_encode_fn = lambda x: encode_txt(x, hparams.vocab)

  for mode in ["eval", "train"]:
    with open(hparams["{}_files".format(mode)], "r") as f:
      meta_data = [x.strip().split("|") for x in f.readlines()]
    head = meta_data[0]
    body = meta_data[1:]
    
    data = []
    for d in body:
      fname = get_value_from_query("fname", head, d)
      txt = get_value_from_query("txt", head, d)
      sid = int(get_value_from_query("sid", head, d))
      data.append((fname, txt, sid))

    save_tfrecords(hparams.tfr_dir, "%s_%s" % (hparams.tfr_prefix, mode), data, wav_load_fn, txt_encode_fn, _NUM_SHARDS[mode])


if __name__ == "__main__":
  import argparse
  from hparams import Hparams, import_configs

  parser = argparse.ArgumentParser("Generate and save datasets")
  parser.add_argument("-c", "--conf", dest="configs", default=[], nargs="*",
            help="A list of configuration items. "
                 "An item is a file path or a 'key=value' formatted string. "
                 "The type of a value is determined by applying int(), float(), and str() "
                 "to it sequencially.")
  args = parser.parse_args()
  # python generate_data.py -c tfr_dir=datasets/vctk tfr_prefix=vctk train_files=filelists/vctk_sid_audio_text_train_filelist.txt eval_files=filelists/vctk_sid_audio_text_eval_filelist.txt 

  hparams = Hparams()
  import_configs(hparams, args.configs)

  if not os.path.isdir(hparams.tfr_dir):
    os.mkdir(os.path.relpath(hparams.tfr_dir))
  data.load_vocab(hparams)
  
  # Save tfrecords
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.enable_eager_execution()
  audio2tfr(hparams)
