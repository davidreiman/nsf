#!/usr/bin/env python
# coding: utf-8
import os
import sys
import PIL
import time
import pickle
import argparse
import datetime
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import collections as col

from torch import optim
from torch.utils import data
from torch.nn.utils import clip_grad_norm_
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter

sys.path.append('./nsf')
import nsf.nn as nn_
import nsf.utils as utils
from nsf.experiments import cutils

from model import ConditionalFlow
from data import QSODataset


def train():
    model.train()
    global metrics, global_step, start_time
    start_time = time.time()
    train_loss, train_acc = 0, 0

    for epoch in range(args.n_epochs):
        for batch, (data, context) in enumerate(train_iter):
            data, context = data.to(device), context.to(device)

            model.zero_grad()
            loss = model(data, context)
            train_loss += loss.float().item()
            loss.backward()

            clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            scheduler.step()

            global_step += 1

            if global_step % args.log_interval == 0:
                lr = scheduler.get_lr()[0]
                cur_loss = train_loss / args.log_interval
                metrics['train_loss'].append(cur_loss)
                metrics['train_gs'].append(global_step)
                print(f'Epoch: {epoch:03d} | '
                    f'Batch: {(batch + 1):06d} | '
                    f'LR: {lr:.3E} | '
                    f'Current Loss: {cur_loss:0.6f}')
                writer.add_scalar('loss/train', cur_loss, global_step)
                writer.add_scalar('learning_rate', lr, global_step)
                train_loss, train_acc = 0, 0

            if global_step % args.eval_interval == 0:
                evaluate()

    with open(os.path.join(args.log_dir, 'metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)


def evaluate():
    model.eval()
    global best_mae, best_val_loss, metrics, start_time
    val_loss, val_acc, mae = 0, 0, []

    with torch.no_grad():
        for batch, (data, context) in enumerate(valid_iter):
            data, context = data.to(device), context.to(device)
            loss = model(data, context)
            val_loss += loss.float().item()

            for i in range(data.size(0)):
                truth = data[i].cpu().numpy()
                truth = valid_dset.inverse_transform(truth[None, :]).flatten()
                samples, _ = model.sample(context[i], args.eval_n_samples)
                samples = valid_dset.inverse_transform(samples)
                pred = samples.mean(0)
                mae.append(np.abs(pred - truth).mean())

    mae = np.mean(mae)
    val_loss = val_loss / (batch + 1)
    t_elapsed = time.time() - start_time
    t_min = int(t_elapsed // 60)
    t_sec = int(t_elapsed % 60)
    bps = args.eval_interval / t_elapsed

    print('=' * 100)
    print(f'Validation loss: {val_loss} | '
        f'Time elapsed: {t_min:d}m {t_sec:d}s | '
        f'Batches per second: {bps:0.3f}')
    print('=' * 100)

    for name, fig in make_figures():
        writer.add_figure(f'{name}/valid', fig, global_step)
    writer.add_scalar('loss/valid', val_loss, global_step)
    writer.add_scalar('mae/valid', mae, global_step)

    error_bias()

    metrics['valid_loss'].append(val_loss)
    metrics['valid_gs'].append(global_step)

    with open(os.path.join(args.log_dir, 'metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        with open(os.path.join(args.log_dir, 'model.pt'), 'wb') as f:
            torch.save(model, f)
    if mae < best_mae:
        best_mae = mae

    start_time = time.time()
    model.train()


def make_figures():
    model.eval()
    figures = []
    data, context = next(iter(valid_iter))
    data, context = data.to(device), context.to(device)

    idx = np.random.choice(data.size(0))
    truth = data[idx].cpu().numpy()
    truth = valid_dset.inverse_transform(truth[None, :]).flatten()

    context = context[idx]
    samples, _ = model.sample(context, args.eval_n_samples)
    samples = valid_dset.inverse_transform(samples)

    pred = samples.mean(0)
    std = samples.std(0)

    fig = plt.figure(1)
    plt.plot(valid_dset.lya_wave, pred, color='b', label="Spectre")
    plt.fill_between(valid_dset.lya_wave, pred - 2*std, pred + 2*std,
        color='b', alpha=0.15, label=r'$2\sigma$')
    plt.fill_between(valid_dset.lya_wave, pred - std, pred + std,
        color='b', alpha=0.3, label=r'$1\sigma$')
    plt.plot(valid_dset.lya_wave, truth, color='r', label="Truth")
    plt.xlabel(r'Rest-frame Wavelength ($\AA$)')
    plt.ylabel('Normalized Flux')
    plt.legend()
    figures.append(('prediction_std', fig))

    fig = plt.figure(2)
    alpha = 2. / args.eval_n_samples
    [plt.plot(valid_dset.lya_wave, s, color='k', alpha=alpha) for s in samples]
    plt.plot(valid_dset.lya_wave, pred, color='b', label='Spectre')
    plt.plot(valid_dset.lya_wave, truth, color='r', label='Truth')
    plt.xlabel(r'Rest-frame Wavelength ($\AA$)')
    plt.ylabel('Normalized Flux')
    plt.legend()
    figures.append(('prediction_all', fig))

    fig = plt.figure(3)
    ax = plt.gca()
    ax.set_aspect('equal')
    sns.heatmap(samples, ax=ax, cbar_kws={'shrink': 0.6})
    plt.axis('off')
    figures.append(('sunrise', fig))

    return figures


def error_bias():
    model.eval()
    global best_error_ratio
    error, bias = [], []

    with torch.no_grad():
        for data, context in valid_iter:
            data, context = data.to(device), context.to(device)
            for i in range(data.size(0)):
                d, c = data[i], context[i]

                samples, _ = model.sample(c, args.eval_n_samples)
                samples = valid_dset.inverse_transform(samples)
                d = valid_dset.inverse_transform(d[None, :]).flatten()

                mean = samples.mean(0)
                b = (mean - d) / d
                e = np.abs((mean - d) / d)

                bias.append(b)
                error.append(e)

    error, bias = np.array(error).mean(0), np.array(bias).mean(0)
    error_ratio = (error / epca_error).mean()

    writer.add_scalar('mean_bias/valid', bias.mean(), global_step)
    writer.add_scalar('mean_error/valid', error.mean(), global_step)
    writer.add_scalar('epca_error_ratio/valid', error_ratio, global_step)

    if error_ratio < best_error_ratio:
        best_error_ratio = error_ratio
        with open(os.path.join(args.log_dir, 'model_err_ratio.pt'), 'wb') as f:
            torch.save(model, f)


if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parser = argparse.ArgumentParser(description='Spectre')
    parser.add_argument('--data_filepath', type=str,
                        default='./data/qsanndra_data.npz',
                        help='Filepath for data')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Base directory for logs')
    parser.add_argument('--n_layers', type=int, default=10,
                        help='Number of spline layers in flow')
    parser.add_argument('--hidden_units', type=int, default=256,
                        help='Number of hidden units in spline layer')
    parser.add_argument('--n_blocks', type=int, default=1,
                        help='Number of residual blocks in each spline layer')
    parser.add_argument('--tail_bound', type=float, default=5.,
                        help='Bounds of spline region')
    parser.add_argument('--tails', type=str, default='linear',
                        help='Function type outside spline region')
    parser.add_argument('--n_bins', type=int, default=5,
                        help='Number of bins in piecewise spline transform')
    parser.add_argument('--min_bin_height', type=float, default=1e-3,
                        help='Minimum bin height of piecewise transform')
    parser.add_argument('--min_bin_width', type=float, default=1e-3,
                        help='Minimum bin width of piecewise transform')
    parser.add_argument('--min_derivative', type=float, default=1e-3,
                        help='Minimum derivative at bin edges')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability in flow')
    parser.add_argument('--use_batch_norm', type=int, default=1,
                        help='Use batch norm in spline layers')
    parser.add_argument('--unconditional_transform', type=int, default=0,
                        help='Unconditionally transform identity features')
    parser.add_argument('--use_cnn_encoder', type=int, default=0,
                        help='Use 1-D CNN to encode conditioning information')
    parser.add_argument('--encoder_units', type=int, default=128,
                        help='Number of hidden units in encoder layers')
    parser.add_argument('--n_encoder_layers', type=int, default=2,
                        help='Number of layers in encoder')
    parser.add_argument('--encoder_dropout', type=float, default=0.,
                        help='Dropout probability in encoder')
    parser.add_argument('--wavelength_threshold', type=float, default=1290.,
                        help='Wavelength threshold between blue/red sides')
    parser.add_argument('--subsample', type=int, default=3,
                        help='Subsample spectra for dimensionality reduction')
    parser.add_argument('--log_transform', type=int, default=0,
                        help='Log transform spectra before standardization')
    parser.add_argument('--standardize', type=int, default=1,
                        help='Standardize spectra (by wavelength)')
    parser.add_argument('--drop_outliers', type=int, default=1,
                        help='Drop spectra with outlying flux.')
    parser.add_argument('--learning_rate', type=float, default=2.5e-4,
                        help='Initial learning rate during annealing')
    parser.add_argument('--min_learning_rate', type=float, default=1e-8,
                        help='Minimum learning rate during annealing')
    parser.add_argument('--anneal_period', type=int, default=10000,
                        help='Learning rate annealing period')
    parser.add_argument('--anneal_mult', type=int, default=2,
                        help='Warm restart period multiplier')
    parser.add_argument('--n_restarts', type=int, default=3,
                        help='Number of annealing restarts')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=256,
                        help='Evaluation batch size')
    parser.add_argument('--eval_n_samples', type=int, default=200,
                        help='Number of samples to draw from flow')
    parser.add_argument('--grad_clip', type=float, default=5.,
                        help='Gradient norm clipping threshold')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Training loss/accuracy logging interval')
    parser.add_argument('--eval_interval', type=int, default=200,
                        help='Validation loss/accuracy logging interval')
    parser.add_argument('--n_epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--exp_name', type=str, default=now,
                        help='Optional experiment name')
    args = parser.parse_args()

    # =========================================================================

    cuda = torch.cuda.is_available()
    device = torch.device('cuda') if cuda else torch.device('cpu')

    train_dset = QSODataset(filepath=args.data_filepath, partition='train',
        wavelength_threshold=args.wavelength_threshold, subsample=args.subsample,
        log_transform=args.log_transform, standardize=args.standardize,
        drop_outliers=args.drop_outliers)
    valid_dset = QSODataset(filepath=args.data_filepath, partition='valid',
        wavelength_threshold=args.wavelength_threshold, subsample=args.subsample,
        log_transform=args.log_transform, standardize=args.standardize,
        drop_outliers=args.drop_outliers, scaler=train_dset.scaler)

    train_iter = data.DataLoader(train_dset, batch_size=args.batch_size,
        drop_last=True, shuffle=True, num_workers=8)
    valid_iter = data.DataLoader(valid_dset, batch_size=args.eval_batch_size,
        drop_last=True, shuffle=True, num_workers=8)

    model = ConditionalFlow(dim=train_dset.data_dim,
        context_dim=train_dset.context_dim, n_layers=args.n_layers,
        hidden_units=args.hidden_units, n_blocks=args.n_blocks,
        dropout=args.dropout, encoder_dropout=args.encoder_dropout,
        n_encoder_layers=args.n_encoder_layers, encoder_units=args.encoder_units,
        use_batch_norm=args.use_batch_norm, tails=args.tails,
        tail_bound=args.tail_bound, n_bins=args.n_bins,
        min_bin_height=args.min_bin_height, min_bin_width=args.min_bin_width,
        min_derivative=args.min_derivative,
        unconditional_transform=args.unconditional_transform,
        use_cnn_encoder=args.use_cnn_encoder, subsample=args.subsample,
        device=device)

    n_params = utils.get_num_parameters(model)
    print(f"There are {n_params} trainable parameters in this model.")

    last_step = np.sum([args.anneal_period*args.anneal_mult**i
        for i in range(args.n_restarts+1)])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=args.anneal_period,
        T_mult=args.anneal_mult,
        eta_min=args.min_learning_rate,
        last_epoch=last_step)

    metrics = col.defaultdict(list)
    global_step = 0
    best_val_loss = np.inf
    best_mae = np.inf
    best_error_ratio = np.inf
    start_time = None

    if 'SLURM_JOB_ID' in os.environ:
        id = os.environ['SLURM_JOB_ID']
        args.exp_name += f'-SLURM-{id}'
    args.log_dir = os.path.join(args.log_dir, args.exp_name)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    with open(os.path.join(args.log_dir, 'args.txt'), 'wb') as f:
        pickle.dump(vars(args), f)
    with open(os.path.join(args.log_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(train_dset.scaler, f)

    epca = np.load('./notebooks/epca_bias_error.npz')
    epca_error = epca['error'][::args.subsample]

    writer = SummaryWriter(log_dir=args.log_dir, flush_secs=5)
    os.system(f'cp *.py {args.log_dir}')

    try:
        train()
    except KeyboardInterrupt:
        pass
    finally:
        metric_dict = {
            'hparams/best_validation_loss': best_val_loss,
            'hparams/best_mae': best_mae,
            'hparams/best_error_ratio': best_error_ratio}
        writer.add_hparams(vars(args), metric_dict)
