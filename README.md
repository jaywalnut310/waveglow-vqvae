# WaveGlow vocoder with VQVAE

Tensorflow implementation of [WaveGlow: A Flow-based Generative Network for Speech Synthesis](https://arxiv.org/abs/1811.00002)
and [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937).

This implementation includes **multi-gpu** and **mixed precision**(unstable yet) support.
It is highly based on some github repositories:[waveglow](https://github.com/NVIDIA/waveglow).
Data used here are the [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/) and [VCTK Corpus](https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html).

You can choose local conditions among mel-spectrogram or vector-quantized representations and also choose whether to use speaker identity as a global condition.
As more options, polyak-averaging, FiLM and weight normalization are implemented.


## Audio Samples
### LJ dataset
Mel spectrogram condition (original WaveGlow): https://drive.google.com/open?id=1HuV51fnhEZG_6vGubXVrer6lAtZK7py9

VQVAE condition: https://drive.google.com/open?id=1xcGSelMycn2g-72noZH4vPiPpG0d7pZq

### VCTK Corpus (Voice conversion)
It does not work well at now :(

Source (360): https://drive.google.com/open?id=1CfEvnQS_dVYRhsvj8NDqogOJlzK7npTd

Target (303): https://drive.google.com/open?id=1-kcSglimKgJrRjLDfPbD7s5KxZuFRY-i


## My Humble Contribution
I slightly modify the original VQVAE optimization technique to increase robustness w.r.t hyperparameter choices and diversity of latent code usage without index-collapse.
That is,
- the original technique contains 1) finding neareast latent codes given encoded vectors and 2) updating latent codes according to matching encoded vectors.
- I modify them as 1) finding distribution of latent codes given encoded vectors and 2) updating latent codes to increase the likelihood given distribution of matching encoded vectors.
- By replacing EMA with the gradient descent method, it can give additional gradient signals to latent codes to reduce reconstruction loss (which is impossible in the EMA setting.).

It resembles Soft-EM method a lot. The difference between Soft-EM is to replace closed form Maximization step with a gradient descent method.
For more information, please see em_toy.ipynb or contact me(jaywalnut310@gmail.com).

As I haven't investigated this method thoroughly, I cannot say it is better than previous methods in almost every cases.
But I found this novel method works pretty well in all of my experimental settings (no index-collapse).


## Pre-requisites
1. Tensorflow 1.12 (1.13 would work with some deprecation warnings)
2. (If fp16 training is needed) Volta GPUs


## Setup
```sh
# 1. create dataset folder
mkdir datasets
cd datasets

# 2. Download and extract datasets
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -jxvf LJSpeech-1.1.tar.bz2

# Additionally, download VCTK Corpus
wget http://homepages.inf.ed.ac.uk/jyamagis/release/VCTK-Corpus.tar.gz
tar -zxvf VCTK-Corpus.tar.gz
cd ../filelists
python resample_vctk.py # Change sample rate

# 3. Create TFRecords
python generate_data.py

# Additionally, create VCTK TFRecords
python generate_data.py -c tfr_dir=datasets/vctk tfr_prefix=vctk train_files=filelists/vctk_sid_audio_text_train_filelist.txt eval_files=filelists/vctk_sid_audio_text_eval_filelist.txt
```


## Training
```sh
# 1. Create log directory
mkdir ~/your-log-dir

# 2. (Optional) Copy configs
cp ./config.yml ~/your-log-dir

# 3. Run training
python train.py -m ~/your-log-dir
```

If you want to change hparams, then you can do it by choosing one of two options.
* modify config.yml
* add arguments as below:
  ```sh
  python train.py -m ~/your-log-dir --c hidden_size=512 num_heads=8
  ```

Example configs:
- fp32 training: `python train.py -m ~/your-log-dir --c ftype=float32 loss_scale=1`
- mel condition: `python train.py -m ~/your-log-dir --c local_condition=mel use_vq=false`
- remove FiLM layers: `python train.py -m ~/your-log-dir --c use_film=false`


## Pre-trained models
Compressed model directories with pretrained weights are available: WILL BE UPLOADED SOON!

You can generate samples with those models in inference.ipynb.

You may have to change tfr_dir and model_dir to work on your settings.


## Disclaimer
- For fp16 settings, you need 1 week to train 1M steps with 4 V100 GPUs.
- I haven't tried fp32 training, so there might be some issues to train high quality models.
- As fp16 training is not robust enough (at now), I usually train FiLM enabled model and unabled model consequently and choose one which survives.
- For a single speaker dataset(LJ Speech dataset), trained model vocoding quality is good enough compared to mel-spectrogram conditioned one.
- For multi-speaker dataset(VCTK Corpus), disentangling between speaker identity and local condition does not work well (at now). I am investigating reasons though.
- The next step would be training Text-to-LatentCodes model(as Transformer) so that fully TTS is possible.
- If you're interested in this project, please improve models with me!
