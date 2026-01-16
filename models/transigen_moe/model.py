#!/usr/bin/env python3
"""
TranSiGen model adapted for PDGrapher data (10716 genes).

Original TranSiGen: Dual VAE for predicting gene expression changes from chemical perturbations.
This version is adapted to work with PDGrapher's 10716-gene expression data.
"""
import torch
import torch.nn.functional as F
from torch import nn, optim
import copy
import numpy as np
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr


class TranSiGen_PDG(torch.nn.Module):
    """
    TranSiGen adapted for PDGrapher's 10716 genes.

    Architecture:
    - Encoder x1: diseased expression -> latent z1
    - Encoder x2: treated expression -> latent z2
    - Feature embedding: molecular features -> embedded features
    - Fusion: z1 + mol_embed -> predicted z2
    - Decoder: z2 -> predicted treated expression
    """

    def __init__(self, n_genes=10716, n_latent=100, n_en_hidden=[1200, 600],
                 n_de_hidden=[600, 1200], features_dim=2304, features_embed_dim=[400], **kwargs):
        """
        Args:
            n_genes: Number of genes (10716 for PDGrapher)
            n_latent: Latent dimension
            n_en_hidden: Encoder hidden layer sizes
            n_de_hidden: Decoder hidden layer sizes
            features_dim: Molecular feature dimension (2304 for KPGT, 768 for ChemBERTa)
            features_embed_dim: Feature embedding hidden sizes
        """
        super(TranSiGen_PDG, self).__init__()
        self.n_genes = n_genes
        self.n_latent = n_latent
        self.n_en_hidden = n_en_hidden
        self.n_de_hidden = n_de_hidden
        self.features_dim = features_dim
        self.feat_embed_dim = features_embed_dim
        self.init_w = kwargs.get('init_w', False)
        self.beta = kwargs.get('beta', 0.1)
        self.path_model = kwargs.get('path_model', 'trained_model/')
        self.dev = kwargs.get('device', torch.device('cpu'))
        self.dropout = kwargs.get('dropout', 0.1)
        self.random_seed = kwargs.get('random_seed', 42)
        self.use_wandb = kwargs.get('use_wandb', False)

        # Encoder for x2 (treated expression)
        encoder = [
            nn.Linear(self.n_genes, self.n_en_hidden[0]),
            nn.BatchNorm1d(self.n_en_hidden[0]),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        ]
        if len(n_en_hidden) > 1:
            for i in range(len(n_en_hidden)-1):
                en_hidden = [
                    nn.Linear(self.n_en_hidden[i], self.n_en_hidden[i+1]),
                    nn.BatchNorm1d(self.n_en_hidden[i+1]),
                    nn.ReLU(),
                    nn.Dropout(self.dropout)
                ]
                encoder = encoder + en_hidden
        self.encoder_x2 = nn.Sequential(*encoder)

        # Encoder for x1 (diseased expression) - same architecture
        encoder_x1 = copy.deepcopy(encoder)
        self.encoder_x1 = nn.Sequential(*encoder_x1)

        # Latent layers for z2
        self.mu_z2 = nn.Sequential(nn.Linear(self.n_en_hidden[-1], self.n_latent))
        self.logvar_z2 = nn.Sequential(nn.Linear(self.n_en_hidden[-1], self.n_latent))

        # Latent layers for z1
        self.mu_z1 = nn.Sequential(nn.Linear(self.n_en_hidden[-1], self.n_latent))
        self.logvar_z1 = nn.Sequential(nn.Linear(self.n_en_hidden[-1], self.n_latent))

        # Decoder for x2
        if len(n_de_hidden) == 0:
            decoder = [nn.Linear(self.n_latent, self.n_genes)]
        else:
            decoder = [
                nn.Linear(self.n_latent, self.n_de_hidden[0]),
                nn.BatchNorm1d(self.n_de_hidden[0]),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ]
            if len(n_de_hidden) > 1:
                for i in range(len(self.n_de_hidden)-1):
                    de_hidden = [
                        nn.Linear(self.n_de_hidden[i], self.n_de_hidden[i+1]),
                        nn.BatchNorm1d(self.n_de_hidden[i+1]),
                        nn.ReLU(),
                        nn.Dropout(self.dropout)
                    ]
                    decoder = decoder + de_hidden
            decoder.append(nn.Linear(self.n_de_hidden[-1], self.n_genes))
        self.decoder_x2 = nn.Sequential(*decoder)

        # Decoder for x1 - same architecture
        decoder_x1 = copy.deepcopy(decoder)
        self.decoder_x1 = nn.Sequential(*decoder_x1)

        # Feature embedding network
        if self.feat_embed_dim is None:
            self.mu_z2Fz1 = nn.Sequential(nn.Linear(self.n_latent + self.features_dim, self.n_latent))
            self.logvar_z2Fz1 = nn.Sequential(nn.Linear(self.n_latent + self.features_dim, self.n_latent))
        else:
            feat_embeddings = [
                nn.Linear(self.features_dim, self.feat_embed_dim[0]),
                nn.BatchNorm1d(self.feat_embed_dim[0]),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ]
            if len(self.feat_embed_dim) > 1:
                for i in range(len(self.feat_embed_dim)-1):
                    feat_hidden = [
                        nn.Linear(self.feat_embed_dim[i], self.feat_embed_dim[i+1]),
                        nn.BatchNorm1d(self.feat_embed_dim[i+1]),
                        nn.ReLU(),
                        nn.Dropout(self.dropout)
                    ]
                    feat_embeddings = feat_embeddings + feat_hidden
            self.feat_embeddings = nn.Sequential(*feat_embeddings)

            self.mu_z2Fz1 = nn.Sequential(nn.Linear(self.n_latent + self.feat_embed_dim[-1], self.n_latent))
            self.logvar_z2Fz1 = nn.Sequential(nn.Linear(self.n_latent + self.feat_embed_dim[-1], self.n_latent))

        if self.init_w:
            self.encoder_x1.apply(self._init_weights)
            self.decoder_x1.apply(self._init_weights)
            self.encoder_x2.apply(self._init_weights)
            self.decoder_x2.apply(self._init_weights)
            self.mu_z1.apply(self._init_weights)
            self.logvar_z1.apply(self._init_weights)
            self.mu_z2.apply(self._init_weights)
            self.logvar_z2.apply(self._init_weights)

    def _init_weights(self, layer):
        if type(layer) == nn.Linear:
            torch.nn.init.xavier_uniform_(layer.weight)

    def encode_x1(self, X):
        y = self.encoder_x1(X)
        mu, logvar = self.mu_z1(y), self.logvar_z1(y)
        z = self.sample_latent(mu, logvar)
        return z, mu, logvar

    def encode_x2(self, X):
        y = self.encoder_x2(X)
        mu, logvar = self.mu_z2(y), self.logvar_z2(y)
        z = self.sample_latent(mu, logvar)
        return z, mu, logvar

    def decode_x1(self, z):
        return self.decoder_x1(z)

    def decode_x2(self, z):
        return self.decoder_x2(z)

    def sample_latent(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(self.dev)
        eps = eps.mul_(std).add_(mu)
        return eps

    def forward(self, x1, features):
        """
        Forward pass.

        Args:
            x1: diseased expression [batch, n_genes]
            features: molecular features [batch, features_dim]

        Returns:
            x1_rec, mu1, logvar1, x2_pred, mu_pred, logvar_pred, z2_pred
        """
        z1, mu1, logvar1 = self.encode_x1(x1)
        x1_rec = self.decode_x1(z1)

        if self.feat_embed_dim is not None:
            feat_embed = self.feat_embeddings(features)
        else:
            feat_embed = features

        z1_feat = torch.cat([z1, feat_embed], 1)
        mu_pred, logvar_pred = self.mu_z2Fz1(z1_feat), self.logvar_z2Fz1(z1_feat)
        z2_pred = self.sample_latent(mu_pred, logvar_pred)
        x2_pred = self.decode_x2(z2_pred)

        return x1_rec, mu1, logvar1, x2_pred, mu_pred, logvar_pred, z2_pred

    def loss(self, x1_train, x1_rec, mu1, logvar1, x2_train, x2_rec, mu2, logvar2,
             x2_pred, mu_pred, logvar_pred):
        """Compute total loss."""
        mse_x1 = F.mse_loss(x1_rec, x1_train, reduction="sum")
        mse_x2 = F.mse_loss(x2_rec, x2_train, reduction="sum")
        mse_pert = F.mse_loss(x2_pred - x1_train, x2_train - x1_train, reduction="sum")

        kld_x1 = -0.5 * torch.sum(1. + logvar1 - mu1.pow(2) - logvar1.exp())
        kld_x2 = -0.5 * torch.sum(1. + logvar2 - mu2.pow(2) - logvar2.exp())
        kld_pert = -0.5 * torch.sum(1 + (logvar_pred - logvar2) -
                                     ((mu_pred - mu2).pow(2) + logvar_pred.exp()) / logvar2.exp())

        return (mse_x1 + mse_x2 + mse_pert + self.beta * kld_x1 + self.beta * kld_x2 + self.beta * kld_pert,
                mse_x1, mse_x2, mse_pert, kld_x1, kld_x2, kld_pert)

    def compute_deg_correlations(self, x1_data, x2_data, x2_pred):
        """Compute DEG correlations."""
        x1_np = x1_data.cpu().numpy()
        x2_np = x2_data.cpu().numpy()
        x2_pred_np = x2_pred.cpu().numpy()

        deg_actual = x2_np - x1_np
        deg_predicted = x2_pred_np - x1_np

        pearson_corrs = []
        spearman_corrs = []

        for i in range(deg_actual.shape[0]):
            try:
                p, _ = pearsonr(deg_actual[i], deg_predicted[i])
                s, _ = spearmanr(deg_actual[i], deg_predicted[i])
                if not np.isnan(p):
                    pearson_corrs.append(p)
                if not np.isnan(s):
                    spearman_corrs.append(s)
            except:
                continue

        return {
            'pearson_mean': np.mean(pearson_corrs) if pearson_corrs else 0,
            'spearman_mean': np.mean(spearman_corrs) if spearman_corrs else 0,
        }


# MoE variants
class SparseMoE(nn.Module):
    """Sparse Mixture of Experts (top-k=1)."""

    def __init__(self, latent_dim, embed_dim, num_experts=4, k=1, sparsity_coef=0.1):
        super(SparseMoE, self).__init__()
        self.input_dim = latent_dim + embed_dim
        self.num_experts = num_experts
        self.k = k
        self.sparsity_coef = sparsity_coef

        self.experts = nn.ModuleList([
            nn.Linear(self.input_dim, self.input_dim) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(self.input_dim, num_experts)
        self.noise_epsilon = 1e-2
        self.auxiliary_loss = torch.tensor(0.0)

    def forward(self, x, y):
        inp = torch.cat([x, y], dim=1)

        gate_logits = self.gate(inp)
        if self.training:
            noise = torch.randn_like(gate_logits) * self.noise_epsilon
            gate_logits = gate_logits + noise

        topk_val, topk_indices = torch.topk(gate_logits, self.k, dim=1)
        topk_gates = F.softmax(topk_val, dim=1)

        all_expert_outputs = torch.stack([expert(inp) for expert in self.experts], dim=1)

        batch_size = inp.size(0)
        batch_indices = torch.arange(batch_size, device=inp.device).unsqueeze(1).expand(-1, self.k)
        topk_expert_outputs = all_expert_outputs[batch_indices, topk_indices]

        topk_gates = topk_gates.unsqueeze(2)
        fused_output = torch.sum(topk_expert_outputs * topk_gates, dim=1)

        # Sparsity loss
        gate_weights = self.gate.weight
        sparsity_loss = torch.mean(torch.norm(gate_weights, dim=1))
        self.auxiliary_loss = self.sparsity_coef * sparsity_loss

        return fused_output

    def get_auxiliary_loss(self):
        return self.auxiliary_loss


class BalancedMoE(nn.Module):
    """Balanced Mixture of Experts (top-k=2 with load balancing)."""

    def __init__(self, latent_dim, embed_dim, num_experts=4, k=2, sparsity_coef=0.1):
        super(BalancedMoE, self).__init__()
        self.input_dim = latent_dim + embed_dim
        self.num_experts = num_experts
        self.k = k
        self.sparsity_coef = sparsity_coef

        self.experts = nn.ModuleList([
            nn.Linear(self.input_dim, self.input_dim) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(self.input_dim, num_experts)
        self.noise_epsilon = 1e-2
        self.auxiliary_loss = torch.tensor(0.0)

    def forward(self, x, y):
        inp = torch.cat([x, y], dim=1)

        gate_logits = self.gate(inp)
        if self.training:
            noise = torch.randn_like(gate_logits) * self.noise_epsilon
            gate_logits = gate_logits + noise

        topk_val, topk_indices = torch.topk(gate_logits, self.k, dim=1)
        topk_gates = F.softmax(topk_val, dim=1)

        all_expert_outputs = torch.stack([expert(inp) for expert in self.experts], dim=1)

        batch_size = inp.size(0)
        batch_indices = torch.arange(batch_size, device=inp.device).unsqueeze(1).expand(-1, self.k)
        topk_expert_outputs = all_expert_outputs[batch_indices, topk_indices]

        topk_gates = topk_gates.unsqueeze(2)
        fused_output = torch.sum(topk_expert_outputs * topk_gates, dim=1)

        # Balance + sparsity loss
        balance_loss = self._compute_balance_loss(inp)
        gate_weights = self.gate.weight
        sparsity_loss = torch.mean(torch.norm(gate_weights, dim=1))
        self.auxiliary_loss = balance_loss + self.sparsity_coef * sparsity_loss

        return fused_output

    def _compute_balance_loss(self, inp):
        gate_probs = F.softmax(self.gate(inp), dim=1)
        importance = torch.sum(gate_probs, dim=0)
        one_hot = torch.zeros_like(gate_probs)
        one_hot.scatter_(1, torch.topk(gate_probs, self.k, dim=1)[1], 1)
        load = torch.sum(one_hot, dim=0).float()

        importance_mean, importance_std = torch.mean(importance), torch.std(importance)
        load_mean, load_std = torch.mean(load), torch.std(load)
        balance_loss = (importance_std / (importance_mean + 1e-6)) + (load_std / (load_mean + 1e-6))
        return balance_loss

    def get_auxiliary_loss(self):
        return self.auxiliary_loss


class TranSiGen_MoE_PDG(TranSiGen_PDG):
    """
    TranSiGen with MoE fusion for PDGrapher data.
    """

    def __init__(self, n_genes=10716, n_latent=100, n_en_hidden=[1200, 600],
                 n_de_hidden=[600, 1200], features_dim=2304, features_embed_dim=[400],
                 moe_style='balanced', num_experts=4, **kwargs):
        # Initialize base class
        super(TranSiGen_MoE_PDG, self).__init__(
            n_genes=n_genes, n_latent=n_latent, n_en_hidden=n_en_hidden,
            n_de_hidden=n_de_hidden, features_dim=features_dim,
            features_embed_dim=features_embed_dim, **kwargs
        )

        self.moe_style = moe_style
        self.num_experts = num_experts

        # Replace simple fusion with MoE
        embed_dim = self.feat_embed_dim[-1] if self.feat_embed_dim else features_dim

        if moe_style == 'sparse':
            self.moe_fusion = SparseMoE(n_latent, embed_dim, num_experts=num_experts, k=1)
        else:  # balanced
            self.moe_fusion = BalancedMoE(n_latent, embed_dim, num_experts=num_experts, k=2)

        # Prediction head after MoE
        moe_output_dim = n_latent + embed_dim
        self.mu_z2Fz1 = nn.Sequential(nn.Linear(moe_output_dim, n_latent))
        self.logvar_z2Fz1 = nn.Sequential(nn.Linear(moe_output_dim, n_latent))

    def forward(self, x1, features):
        z1, mu1, logvar1 = self.encode_x1(x1)
        x1_rec = self.decode_x1(z1)

        if self.feat_embed_dim is not None:
            feat_embed = self.feat_embeddings(features)
        else:
            feat_embed = features

        # MoE fusion instead of simple concat
        z1_feat = self.moe_fusion(z1, feat_embed)

        mu_pred, logvar_pred = self.mu_z2Fz1(z1_feat), self.logvar_z2Fz1(z1_feat)
        z2_pred = self.sample_latent(mu_pred, logvar_pred)
        x2_pred = self.decode_x2(z2_pred)

        return x1_rec, mu1, logvar1, x2_pred, mu_pred, logvar_pred, z2_pred

    def get_auxiliary_loss(self):
        return self.moe_fusion.get_auxiliary_loss()


if __name__ == "__main__":
    print("Testing TranSiGen PDG models...")

    # Test base model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TranSiGen_PDG(n_genes=10716, device=device).to(device)

    x1 = torch.randn(4, 10716).to(device)
    features = torch.randn(4, 2304).to(device)

    out = model(x1, features)
    print(f"TranSiGen_PDG output shapes: x1_rec={out[0].shape}, x2_pred={out[3].shape}")

    # Test MoE model
    model_moe = TranSiGen_MoE_PDG(n_genes=10716, moe_style='balanced', device=device).to(device)
    out_moe = model_moe(x1, features)
    print(f"TranSiGen_MoE_PDG output shapes: x1_rec={out_moe[0].shape}, x2_pred={out_moe[3].shape}")

    print("Models work correctly!")
