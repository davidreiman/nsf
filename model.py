#!/usr/bin/env python
# coding: utf-8
import sys
import numpy as np
import torch
import torch.nn as nn
sys.path.append('./nsf')
import nsf.nn as nn_
import nsf.utils as utils
from nsf.nde import distributions, flows, transforms


class Encoder(nn.Module):
    """An encoder for conditioning information."""
    def __init__(self, context_dim, encoder_units, n_encoder_layers,
        encoder_dropout):
        super().__init__()
        self.context_dim = context_dim
        self.encoder_units = encoder_units
        self.n_encoder_layers = n_encoder_layers
        self.encoder_dropout = encoder_dropout

        self.layers = nn.ModuleList()
        self.layers.append(self.encoder_layer(context_dim, encoder_units))
        self.layers.extend(
            [self.encoder_layer(encoder_units, encoder_units)
            for _ in range(n_encoder_layers - 1)])

    def encoder_layer(self, in_units, out_units):
        layer = nn.Sequential(
            nn.Linear(in_units, out_units),
            nn.PReLU(),
            nn.BatchNorm1d(out_units),
            nn.Dropout(self.encoder_dropout))
        return layer

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class CNNEncoder(nn.Module):
    """A 1-D CNN encoder for conditioning information."""
    def __init__(self, context_dim, encoder_units, n_encoder_layers,
        encoder_dropout, subsample):
        super().__init__()
        self.context_dim = context_dim
        self.encoder_units = encoder_units
        self.n_encoder_layers = n_encoder_layers
        self.encoder_dropout = encoder_dropout
        self.subsample = subsample

        # Approximate width of strong emission features
        kernel_size = 39 // subsample
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(self.encoder_layer(1, 32, kernel_size, 5))
        self.conv_layers.extend(
            [self.encoder_layer(32, 32, 3, 1)
            for _ in range(n_encoder_layers - 1)])
        output_size = self.compute_output_size(context_dim, 1, 1, kernel_size, 5)
        self.fc = nn.Sequential(
            nn.Linear(32*output_size, encoder_units),
            nn.PReLU(),
            nn.BatchNorm1d(encoder_units),
            nn.Dropout(self.encoder_dropout))

    def compute_output_size(self, l_in, padding, dilation, kernel_size, stride):
        arg = (l_in + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1
        return int(np.floor(arg))

    def encoder_layer(self, in_channels, out_channels, kernel_size, stride):
        layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels,
                kernel_size, stride, padding=1),
            nn.PReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(self.encoder_dropout))
        return layer

    def forward(self, x):
        bsz = x.size(0)
        x = x.unsqueeze(1)
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(bsz, -1)
        x = self.fc(x)
        return x


class ConditionalFlow(nn.Module):
    """A conditional rational quadratic neural spline flow."""
    def __init__(self, dim, context_dim, n_layers, n_encoder_layers,
        encoder_units, hidden_units, n_blocks, dropout, encoder_dropout,
        use_batch_norm, tails, tail_bound, n_bins, min_bin_height,
        min_bin_width, min_derivative, unconditional_transform,
        use_cnn_encoder, subsample, device):
        super().__init__()
        self.dim = dim
        self.context_dim = context_dim
        self.n_layers = n_layers
        self.n_encoder_layers = n_encoder_layers
        self.encoder_units = encoder_units
        self.hidden_units = hidden_units
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.encoder_dropout = encoder_dropout
        self.use_batch_norm = use_batch_norm
        self.tails = tails
        self.tail_bound = tail_bound
        self.n_bins = n_bins
        self.min_bin_height = min_bin_height
        self.min_bin_width = min_bin_width
        self.min_derivative = min_derivative
        self.unconditional_transform = unconditional_transform
        self.use_cnn_encoder = use_cnn_encoder
        self.subsample = subsample
        self.device = device

        distribution = distributions.StandardNormal([dim]).to(device)
        transform = transforms.CompositeTransform([
            self.create_transform() for _ in range(self.n_layers)], device)
        self.flow = flows.Flow(transform, distribution).to(device)

        if use_cnn_encoder:
            self.encoder = CNNEncoder(context_dim=context_dim,
                encoder_units=encoder_units, n_encoder_layers=n_encoder_layers,
                encoder_dropout=encoder_dropout, subsample=subsample).to(device)
        else:
            self.encoder = Encoder(context_dim=context_dim,
                encoder_units=encoder_units, n_encoder_layers=n_encoder_layers,
                encoder_dropout=encoder_dropout).to(device)

    def create_transform(self):
        """Create invertible rational quadratic transformations."""
        linear = transforms.RandomPermutation(features=self.dim).to(self.device)
        base = transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=utils.create_mid_split_binary_mask(features=self.dim),
            transform_net_create_fn=lambda in_features, out_features:
                nn_.ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    context_features=self.encoder_units,
                    hidden_features=self.hidden_units,
                    num_blocks=self.n_blocks,
                    dropout_probability=self.dropout,
                    use_batch_norm=self.use_batch_norm,
                ),
            tails=self.tails,
            tail_bound=self.tail_bound,
            num_bins=self.n_bins,
            min_bin_height=self.min_bin_height,
            min_bin_width=self.min_bin_width,
            min_derivative=self.min_derivative,
            apply_unconditional_transform=self.unconditional_transform,
        )
        t = transforms.CompositeTransform([linear, base], self.device)
        return t

    def _forward(self, inputs, context):
        """Forward pass in density estimation direction.

        Args:
            inputs (torch.Tensor): [N, dim] tensor of data.
            context (torch.Tensor): [N, context_dim] tensor of context."""
        context = self.encoder(context)
        log_density = self.flow.log_prob(inputs, context)
        return log_density

    def forward(self, inputs, context):
        """Forward pass to negative log likelihood (NLL).

        Args:
            inputs (torch.Tensor): [N, dim] tensor of data.
            context (torch.Tensor): [N, context_dim] tensor of context."""
        log_density = self._forward(inputs, context)
        loss = -torch.mean(log_density)
        return loss

    def sample(self, context, n_samples):
        """Draw samples from the conditional flow.

        Args:
            context (torch.Tensor): [context_dim] tensor of conditioning info.
            n_samples (int): Number of samples to draw."""
        context = self.encoder(context.unsqueeze(0)).expand(n_samples, -1)
        noise = self.flow._distribution.sample(1, context)
        noise = noise.squeeze(1).to(self.device)
        samples, log_density = self.flow._transform.inverse(noise, context)
        return samples, log_density


def initialize_model(data_dim, context_dim, args, device):
        model = ConditionalFlow(dim=data_dim,
                                context_dim=context_dim, n_layers=args.n_layers,
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
        
        return model
