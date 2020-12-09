#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/8 17:40
# @Author  : 兮嘉
# @File    : tacotron_wavernn_gen.py
# @Software: PyCharm


import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = ""

cwd = os.getcwd()

import sys

sys.path.append(cwd)

import wave
from datetime import datetime

import numpy as np
import tensorflow as tf
from tacotron.datasets import audio
from tacotron.utils.infolog import log
from librosa import effects
from tacotron.models import create_model
from tacotron.utils import plot
from tacotron.utils.text import text_to_sequence
import os
from tacotron_hparams import hparams
import shutil
import hashlib
import time
from tacotron.pinyin.parse_text_to_pyin import get_pyin

from wavernn.utils.dataset import get_vocoder_datasets, gen_testset
from wavernn.utils.dsp import *
from wavernn.models.fatchord_version import WaveRNN
from wavernn.utils.paths import Paths
from wavernn.utils.display import simple_table
import torch
import argparse


def padding_targets(target, r, padding_value):
    lens = target.shape[0]
    if lens % r == 0:
        return target
    else:
        target = np.pad(target, [(0, r - lens % r), (0, 0)], mode='constant', constant_values=padding_value)
        return target


class Synthesizer:
    def load(self, checkpoint_path, hparams, gta=False, model_name='Tacotron'):
        log('Constructing model: %s' % model_name)
        # Force the batch size to be known in order to use attention masking in batch synthesis
        inputs = tf.placeholder(tf.int32, (1, None), name='inputs')
        input_lengths = tf.placeholder(tf.int32, (1), name='input_lengths')

        targets = None  # tf.placeholder(tf.float32, (None, None, hparams.num_mels), name='mel_targets')
        target_lengths = None  # tf.placeholder(tf.int32, (1), name='target_length')
        # gta = True

        with tf.variable_scope('Tacotron_model', reuse=tf.AUTO_REUSE) as scope:
            self.model = create_model(model_name, hparams)
            self.model.initialize(inputs=inputs, input_lengths=input_lengths)
            # mel_targets=targets,  targets_lengths=target_lengths, gta=gta, is_evaluating=True)

            self.mel_outputs = self.model.mel_outputs
            self.alignments = self.model.alignments
            if hparams.predict_linear:
                self.linear_outputs = self.model.linear_outputs
            self.stop_token_prediction = self.model.stop_token_prediction

        self._hparams = hparams

        self.inputs = inputs
        self.input_lengths = input_lengths
        # self.targets = targets
        # self.target_lengths = target_lengths

        log('Loading checkpoint: %s' % checkpoint_path)
        # Memory allocation on the GPUs as needed
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        self.session = tf.Session(config=config)
        self.session.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(self.session, checkpoint_path)

    def synthesize(self, text, out_dir, idx, step):
        hparams = self._hparams

        T2_output_range = (-hparams.max_abs_value, hparams.max_abs_value) if hparams.symmetric_mels else (
        0, hparams.max_abs_value)

        # pyin, text = get_pyin(text)
        print(text.split(' '))

        inputs = [np.asarray(text_to_sequence(text.split(' ')))]
        print(inputs)
        input_lengths = [len(inputs[0])]

        feed_dict = {
            self.inputs: np.asarray(inputs, dtype=np.int32),
            self.input_lengths: np.asarray(input_lengths, dtype=np.int32),
        }

        mels, alignments, stop_tokens = self.session.run([self.mel_outputs,
                                                          self.alignments, self.stop_token_prediction],
                                                         feed_dict=feed_dict)

        mel = mels[0]
        alignment = alignments[0]

        print('pred_mel.shape', mel.shape)
        stop_token = np.round(stop_tokens[0]).tolist()
        target_length = stop_token.index(1) if 1 in stop_token else len(stop_token)

        mel = np.clip(mel, T2_output_range[0], T2_output_range[1])
        mel = mel[:target_length, :]
        mel = (mel + T2_output_range[1]) / (2 * T2_output_range[1])
        mel = np.clip(mel, 0.0, 1.0)  # 0~1.0

        pred_mel_path = os.path.join(out_dir, 'mel-{}-pred.npy'.format(idx))
        np.save(pred_mel_path, mel, allow_pickle=False)
        plot.plot_spectrogram(mel, pred_mel_path.replace('.npy', '.png'), title='')

        alignment_path = os.path.join(out_dir, 'align-{}.png'.format(idx))
        plot.plot_alignment(alignment, alignment_path, title='')

        return pred_mel_path, alignment_path


def gen_from_file(model: WaveRNN, load_path, save_path, batched, target, overlap):
    k = model.get_step() // 1000

    if ".wav" in load_path:
        wav = load_wav(load_path)
        save_wav(wav, save_path / f'__{file_name}__{k}k_steps_target.wav')
        mel = melspectrogram(wav)
    elif ".npy" in load_path:
        mel = np.load(load_path).T
        if mel.ndim != 2 or mel.shape[0] != hp.num_mels:
            raise ValueError(f'Expected a numpy array shaped (n_mels, n_hops), but got {wav.shape}!')
        _max = np.max(mel)
        _min = np.min(mel)
        if _max >= 1.01 or _min <= -0.01:
            raise ValueError(f'Expected spectrogram range in [0,1] but was instead [{_min}, {_max}]')
    else:
        raise ValueError(f"Expected an extension of .wav or .npy, but got {suffix}!")

    mel = torch.tensor(mel).unsqueeze(0)

    batch_str = f'gen_batched_target{target}_overlap{overlap}' if batched else 'gen_NOT_BATCHED'

    # idx = load_path.split('/')[-1].strip().split('-')[1].strip()
    idx = load_path.split('/')[-1].strip().split('.')[0]
    save_str = os.path.join(save_path, idx + '_' + batch_str + '_' + 'step={}k'.format(k) + '.wav')

    _ = model.generate(mel, save_str, batched, target, overlap, hp.mu_law)

    print('\n\nstep = {}'.format(k * 1000))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Tacktron+WaveRNN Samples From Text')
    parser.add_argument('--text', default='', help='text to synthesis.')
    parser.add_argument('--batched', '-b', dest='batched', action='store_false', help='Fast Batched Generation')
    parser.add_argument('--unbatched', '-u', dest='batched', action='store_false', help='Slow Unbatched Generation')
    parser.add_argument('--samples', '-s', type=int, help='[int] number of utterances to generate')
    parser.add_argument('--target', '-t', type=int, help='[int] number of samples in each batch index')
    parser.add_argument('--overlap', '-o', type=int, help='[int] number of crossover samples')
    # parser.add_argument('--file', '-f', type=str, help='[string/path] for testing a wav outside dataset')
    parser.add_argument('--voc_weights', '-w', type=str, help='[string/path] Load in different WaveRNN weights')
    parser.add_argument('--gta', '-g', dest='gta', action='store_true', help='Generate from GTA testset')
    parser.add_argument('--force_cpu', '-c', action='store_true', help='Forces CPU-only training, even when in CUDA capable environment')
    parser.add_argument('--hp_file', metavar='FILE', default='wavernn_hparams.py', help='The file to use for the hyperparameters')

    parser.set_defaults(batched=None)
    args = parser.parse_args()

    past = time.time()

    # Tacotron symthesizes Mel from texts
    print('Tacotron symthesizes Mel from texts')
    synth = Synthesizer()

    ckpt_path = f'logs-Tacotron-2/{hparams.dataset}/taco_pretrained'  # finetune(D8)
    # ckpt_path = 'logs-Tacotron-2/taco_pretrained' # pretrained_tacotron_input
    checkpoint_path = tf.train.get_checkpoint_state(ckpt_path).model_checkpoint_path
    # checkpoint_path = 'logs-Tacotron-2/BQ/taco_pretrained/tacotron_model.ckpt-5000'  # TODO: 直接指定checkpoint path

    synth.load(checkpoint_path, hparams)
    print('succeed in loading checkpoint')

    out_dir = os.path.join(cwd, 'predicted_mel', 'temp')
    # if os.path.exists(out_dir):
    #    shutil.rmtree(out_dir)
    # os.makedirs(out_dir, exist_ok=True)

    text = '哈尔滨今天晴，十度到二十二度，南风三级，空气质量良。'

    text = args.text if args.text != '' else text
    pyin, text = get_pyin(text)

    m = hashlib.md5()
    m.update(text.encode('utf-8'))
    idx = m.hexdigest()
    step = checkpoint_path.split('/')[-1].split('-')[-1].strip()

    # mel_path = os.path.join(out_dir, idx+'_mel.npy')
    pred_mel_path, alignment_path = synth.synthesize(pyin, out_dir, idx, step)

    print(text)
    print(checkpoint_path)
    print(idx)
    print('Tacotron finishes symthesizing Mel from texts')

    # WaveRNN synthesizes wav from Mel
    print('WaveRNN synthesizes wav from Mel')
    hp.configure(args.hp_file)  # Load hparams from file
    # set defaults for any arguments that depend on hparams
    if args.target is None:
        args.target = hp.voc_target
    if args.overlap is None:
        args.overlap = hp.voc_overlap
    if args.batched is None:
        args.batched = hp.voc_gen_batched
    if args.samples is None:
        args.samples = hp.voc_gen_at_checkpoint

    batched = args.batched
    batched = True

    samples = args.samples
    target = args.target
    overlap = args.overlap
    file = pred_mel_path
    gta = args.gta
    gta = False

    if not args.force_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.set_device(1)
    else:
        device = torch.device('cpu')

    # device = torch.device('cpu')

    print('Using device:', device)

    print('\nInitialising Model...\n')

    model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
                    fc_dims=hp.voc_fc_dims,
                    bits=hp.bits,
                    pad=hp.voc_pad,
                    upsample_factors=hp.voc_upsample_factors,
                    feat_dims=hp.num_mels,
                    compute_dims=hp.voc_compute_dims,
                    res_out_dims=hp.voc_res_out_dims,
                    res_blocks=hp.voc_res_blocks,
                    hop_length=hp.hop_length,
                    sample_rate=hp.sample_rate,
                    mode=hp.voc_mode).to(device)

    paths = Paths(hp.dataset)

    voc_weights = args.voc_weights if args.voc_weights else paths.voc_latest_weights
    print(voc_weights)

    model.load(voc_weights)

    simple_table([('Generation Mode', 'Batched' if batched else 'Unbatched'),
                  ('Target Samples', target if batched else 'N/A'),
                  ('Overlap Samples', overlap if batched else 'N/A')])
    print('WaveRNN is Generating wav file based on: ' + file)

    if file:
        out_dir = './tacotron_wavernn_output'
        os.makedirs(out_dir, exist_ok=True)
        gen_from_file(model, file, out_dir, batched, target, overlap)

    print('\n\nExiting...\n')

    print('last: {} seconds'.format(time.time() - past))