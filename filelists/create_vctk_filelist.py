import os
import glob
import random

from scipy.io import wavfile
from scipy import interpolate
import numpy as np


min_signal_length = 16384
power_threshold = 3.


def resample(fname, sample_rate=22050):
  old_samplerate, old_audio = wavfile.read(fname)

  if old_samplerate != sample_rate:
    duration = old_audio.shape[0] / old_samplerate

    time_old  = np.linspace(0, duration, old_audio.shape[0])
    time_new  = np.linspace(0, duration, int(old_audio.shape[0] * sample_rate / old_samplerate))

    interpolator = interpolate.interp1d(time_old, old_audio.T)
    new_audio = interpolator(time_new).T

    return sample_rate, np.round(new_audio).astype(old_audio.dtype)
  return old_samplerate, old_audio


def power(signal, duration=16384):
  power = (signal/ 32768.0) ** 2
  power_list = np.cumsum(power)
  power_list = power_list[duration-1:] - np.concatenate([[0], power_list[:-duration]], 0)
  return power_list


data_path = "../datasets/VCTK-Corpus"
file_list = []
sid = [int(x[1:]) for x in os.listdir(os.path.join(data_path, "wav48"))]
for x in sid:
  if not os.path.isdir(os.path.join(data_path, "wav22/p%s" % x)):
    os.makedirs(os.path.relpath(os.path.join(data_path, "wav22/p%s" % x)))
  wav = glob.glob(os.path.join(data_path, "wav48/p%s/*.wav" % x))
  txt = glob.glob(os.path.join(data_path, "txt/p%s/*.txt" % x))

  for i, (y, z) in enumerate(zip(wav, txt)):
    z = " ".join(" ".join(open(z, "r").readlines()).split())

    sr, wav = resample(y, sample_rate=22050)
    if len(wav) < min_signal_length:
      print("%s is not long enough to train." % y)
      continue
    if power(wav).max() < power_threshold:
      print("%s is not loud enough to train." % y)
      continue

    y = y.replace("wav48", "wav22")
    wavfile.write(y, sr, wav)

    file_list.append([x, y, z])
    if i % 100 == 0:
      print("READ: %4d, %9d" % (x, i), end='\r')
    

random.shuffle(file_list)
with open("vctk_sid_audio_text_eval_filelist.txt", "w") as f:
  flag = ""
  for x, y, z in file_list[:100]:
    f.write("%s%d|%s|%s" % (flag, x, y, z))
    flag = "\n"
with open("vctk_sid_audio_text_test_filelist.txt", "w") as f:
  flag = ""
  for x, y, z in file_list[100:600]:
    f.write("%s%d|%s|%s" % (flag, x, y, z))
    flag = "\n"

with open("vctk_sid_audio_text_train_filelist.txt", "w") as f:
  file_list[600:]
  flag = ""
  for x, y, z in file_list[600:]:
    f.write("%s%d|%s|%s" % (flag, x, y, z))
    flag = "\n"
