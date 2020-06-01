#!/usr/bin/env python
# coding: utf-8
import os
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler


class QSODataset(torch.utils.data.Dataset):
    """QSO spectra iterator."""
    def __init__(self, filepath, partition, wavelength_threshold=1290.,
                 subsample=1, log_transform=False, standardize=True,
                 drop_outliers=False, scaler=None):
        self.log_transform = log_transform
        self.standardize = standardize
        self.scaler = scaler

        print(f"Creating {partition} dataset from file: {filepath}")
        data = np.load(filepath)[partition].astype(np.float32)
        wave = np.load(filepath)['wave'].astype(np.float32)
        data = data[:, (wave >= 1191.5) & (wave < 2900.)]
        wave = wave[(wave >= 1191.5) & (wave < 2900.)]
        data, wave = data[:, ::subsample], wave[::subsample]

        # Drop spectra with negative flux values
        n = len(data)
        mask = ~np.any(data < 0, axis=1)
        data = data[mask]
        print(f"Dropped {n - len(data)} spectra with negative continua values.")

        if log_transform:
            data = np.log(data)
        if standardize:
            if not self.scaler:
                self.scaler = StandardScaler()
                self.scaler.fit(data)
            data = self.scaler.transform(data)

            # Drop spectra with flux >5 sig from dataset mean by wavelength
            if drop_outliers:
                n = len(data)
                mask = ~np.any(np.abs(data) > 5., axis=1)
                data = data[mask]
                print(f"Dropped {n - len(data)} spectra as outliers.")

        print("Data shape:", data.shape)
        self.data = torch.from_numpy(data)
        self.idx = int(np.sum(wave < wavelength_threshold))

        self.wave = wave
        self.lya_wave = wave[:self.idx]

        self.mean_ = self.scaler.mean_[:self.idx]
        self.scale_ = self.scaler.scale_[:self.idx]

        self.data_dim = self.idx
        self.context_dim = len(wave) - self.idx

    def inverse_transform(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if self.standardize:
            if x.shape[1] == self.data_dim + self.context_dim:
                x = self.scaler.inverse_transform(x)
            elif x.shape[1] == self.data_dim:
                x = x * self.scale_ + self.mean_
        if self.log_transform:
            x = np.exp(x)
        return x

    def __getitem__(self, i):
        example = self.data[i]
        data = example[:self.idx]
        context = example[self.idx:]
        return data, context

    def __len__(self):
        return len(self.data)
