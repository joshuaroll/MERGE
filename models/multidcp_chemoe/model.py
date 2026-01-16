"""
MultiDCP_CheMoE: Mixture-of-Experts variant of MultiDCP for gene expression prediction

This model integrates CheMoE's Mixture-of-Experts architecture with MultiDCP's
multi-modal data fusion approach. Key features:

1. Hybrid Encoders:
   - Drug: Neural fingerprint (from MultiDCP) → 128-dim
   - Cell: Linear/Transformer encoder (from MultiDCP) → 50-dim
   - Dose: MLP encoder (CheMoE-style) → 128-dim
   - Gene: Learnable gene embeddings (from MultiDCP) → 128-dim

2. Global Gating Network:
   - Input: Concatenated drug+cell+dose → 306-dim
   - Output: Expert weights for 4 experts
   - Top-k selection (k=2)

3. Gene-Aware Expert Networks:
   - Each expert sees: global features (306) + gene embeddings (128) = 434-dim
   - 4 experts with shared MLP architecture per gene position
   - Output: Weighted combination → predicted gene expression (10,716 genes)

4. Differential Expression Focus:
   - Maintains MultiDCP's DE evaluation pipeline
   - Predicts full 10,716 gene profile
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Import from same package
from .neural_fingerprint import NeuralFingerprint


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class LinearEncoder(nn.Module):
    """Linear encoder for cell expression."""
    def __init__(self, input_dim, output_dim, dropout=0.3):
        super(LinearEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(100, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)


class TransformerEncoder(nn.Module):
    """Transformer encoder for cell expression."""
    def __init__(self, input_dim, output_dim, nhead=2, num_layers=2, dropout=0.3):
        super(TransformerEncoder, self).__init__()
        self.input_proj = nn.Linear(input_dim, output_dim)
        self.pos_encoder = PositionalEncoding(output_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=output_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(output_dim, output_dim)

    def forward(self, x, **kwargs):
        # **kwargs allows epoch parameter from training script to be ignored
        x = self.input_proj(x.unsqueeze(1))
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.output_proj(x.squeeze(1))
        return x


class DoseEncoder(nn.Module):
    """
    MLP encoder for dose information (CheMoE-style)

    Converts dose one-hot encoding to dense embedding
    Architecture: dose_dim → 64 → 128
    """
    def __init__(self, dose_input_dim, dose_emb_dim=128, dropout=0.3):
        super(DoseEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dose_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, dose_emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, dose_input):
        """
        Args:
            dose_input: [batch, dose_input_dim] one-hot encoded dose
        Returns:
            dose_embed: [batch, 128] dose embedding
        """
        return self.encoder(dose_input)


class GatingNetwork(nn.Module):
    """
    Gating network for expert selection (CheMoE-style)

    Takes global features (drug+cell+dose) and produces expert weights.
    Supports top-k sparse gating.
    """
    def __init__(self, input_dim, num_experts=4, dropout=0.3):
        super(GatingNetwork, self).__init__()
        self.num_experts = num_experts

        self.gate = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_experts)
        )

    def forward(self, global_features, top_k=2):
        """
        Args:
            global_features: [batch, input_dim] concatenated drug+cell+dose features
            top_k: number of experts to select (default: 2)
        Returns:
            gates: [batch, num_experts] expert weights (sparse, only top-k non-zero)
            expert_indices: [batch, top_k] indices of selected experts
        """
        # Compute raw gating scores
        logits = self.gate(global_features)  # [batch, num_experts]

        # Top-k selection
        top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=1)  # [batch, top_k]

        # Softmax over selected experts only
        top_k_gates = F.softmax(top_k_logits, dim=1)  # [batch, top_k]

        # Create sparse gate tensor
        gates = torch.zeros_like(logits)  # [batch, num_experts]
        gates.scatter_(1, top_k_indices, top_k_gates)  # Assign top-k weights

        return gates, top_k_indices


class ExpertNetwork(nn.Module):
    """
    Single expert network (shared MLP across genes)

    Processes concatenated global features + gene embeddings
    Architecture: 434 → 128 → 1 (per gene position) - Simplified to match baseline
    """
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super(ExpertNetwork, self).__init__()

        # Simplified to 2-layer MLP matching baseline MultiDCP depth
        self.expert_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features):
        """
        Args:
            features: [batch, num_gene, input_dim] gene-aware features
        Returns:
            predictions: [batch, num_gene, 1] per-gene predictions
        """
        return self.expert_mlp(features)


class MultiDCP_CheMoE(nn.Module):
    """
    Main MultiDCP-CheMoE model combining MoE architecture with MultiDCP encoders

    Architecture:
        1. Encode modalities independently (drug, cell, dose)
        2. Global gating network selects top-k experts based on drug+cell+dose
        3. Concatenate global features with gene embeddings
        4. Each expert processes gene-aware features
        5. Weighted combination of expert outputs
    """
    def __init__(self, device, model_param_registry):
        super(MultiDCP_CheMoE, self).__init__()

        # Set attributes from config
        self.set_attr_from_dict(model_param_registry)

        # Validate dimensions
        assert self.drug_emb_dim == self.gene_emb_dim == 128, \
            'Drug and gene embedding dimensions must be 128'

        # ===== ENCODERS (Hybrid: MultiDCP + CheMoE) =====

        # Drug encoder: Neural fingerprint (MultiDCP)
        self.drug_fp = NeuralFingerprint(
            self.drug_input_dim['atom'],
            self.drug_input_dim['bond'],
            self.conv_size,
            self.drug_emb_dim,  # 128
            self.degree,
            device
        )

        # Cell encoder: Linear or Transformer (MultiDCP)
        self.cell_id_emb_dim = 50
        if self.linear_encoder_flag:
            self.cell_encoder = LinearEncoder(self.cell_id_input_dim, self.cell_id_emb_dim, dropout=self.dropout)
        else:
            self.cell_encoder = TransformerEncoder(self.cell_id_input_dim, self.cell_id_emb_dim, dropout=self.dropout)

        # Dose encoder: MLP (CheMoE-style, NEW)
        self.dose_emb_dim = 128
        self.dose_encoder = DoseEncoder(
            self.pert_idose_input_dim,
            self.dose_emb_dim,
            dropout=self.dropout
        )

        # Gene embeddings: Learnable per-gene (MultiDCP)
        self.gene_index_embed = nn.Embedding(self.num_gene, self.gene_emb_dim)

        # ===== MIXTURE-OF-EXPERTS COMPONENTS (CheMoE) =====

        # Global features dimension: drug (128) + cell (50) + dose (128) = 306
        self.global_dim = self.drug_emb_dim + self.cell_id_emb_dim + self.dose_emb_dim

        # Gating network
        self.num_experts = 4
        self.top_k = 2
        self.gating_network = GatingNetwork(
            self.global_dim,
            self.num_experts,
            dropout=self.dropout
        )

        # Expert input dimension: global (306) + gene_emb (128) = 434
        self.expert_input_dim = self.global_dim + self.gene_emb_dim

        # Create 4 expert networks (simplified 2-layer architecture)
        self.experts = nn.ModuleList([
            ExpertNetwork(
                self.expert_input_dim,
                hidden_dim=128,  # Simplified to match baseline depth
                dropout=self.dropout
            )
            for _ in range(self.num_experts)
        ])

    def set_attr_from_dict(self, model_param_registry):
        """Set model attributes from config dictionary"""
        for k, v in model_param_registry.items():
            self.__setattr__(k, v)

    def forward(self, input_drug, input_gene, mask, input_cell_gex, input_pert_idose, epoch=0):
        """
        Forward pass with Mixture-of-Experts fusion

        Args:
            input_drug: dict with 'molecules', 'atom', 'bond' for neural fingerprint
            input_gene: gene features (not used in current architecture)
            mask: attention mask (not used in current architecture)
            input_cell_gex: [batch, cell_id_input_dim] baseline gene expression
            input_pert_idose: [batch, pert_idose_input_dim] one-hot dose encoding
            epoch: current training epoch (for logging)

        Returns:
            predictions: [batch, num_gene] predicted perturbed gene expression
            cell_hidden: [batch, cell_id_emb_dim] cell embedding (for auxiliary tasks)
        """
        num_batch = input_drug['molecules'].batch_size
        device = input_cell_gex.device

        # ===== STEP 1: ENCODE MODALITIES =====

        # Drug encoding: SMILES → neural fingerprint → 128-dim
        drug_atom_embed = self.drug_fp(input_drug)  # [batch, num_node, 128]
        drug_embed = torch.sum(drug_atom_embed, dim=1)  # [batch, 128]

        # Cell encoding: basal GE → linear/transformer → 50-dim
        cell_hidden = self.cell_encoder(input_cell_gex, epoch=epoch)  # [batch, 50]

        # Dose encoding: one-hot → MLP → 128-dim
        dose_embed = self.dose_encoder(input_pert_idose)  # [batch, 128]

        # Gene embeddings: learnable per-gene embeddings
        gene_indices = torch.arange(self.num_gene, device=device)  # [num_gene]
        gene_embed = self.gene_index_embed(gene_indices)  # [num_gene, 128]
        gene_embed = gene_embed.unsqueeze(0).repeat(num_batch, 1, 1)  # [batch, num_gene, 128]

        # ===== STEP 2: GLOBAL GATING =====

        # Concatenate global features: drug + cell + dose
        global_features = torch.cat([drug_embed, cell_hidden, dose_embed], dim=1)  # [batch, 306]

        # Gating network: select top-k experts
        expert_weights, expert_indices = self.gating_network(
            global_features,
            top_k=self.top_k
        )  # [batch, num_experts], [batch, top_k]

        # ===== STEP 3: GENE-AWARE EXPERT PROCESSING =====

        # Expand global features to gene dimension
        global_features_expanded = global_features.unsqueeze(1).repeat(1, self.num_gene, 1)
        # [batch, num_gene, 306]

        # Concatenate with gene embeddings
        expert_input = torch.cat([global_features_expanded, gene_embed], dim=2)
        # [batch, num_gene, 434]

        # Process through all experts
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(expert_input)  # [batch, num_gene, 1]
            expert_outputs.append(expert_out)
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch, num_experts, num_gene, 1]

        # ===== STEP 4: WEIGHTED COMBINATION =====

        # Reshape expert weights for broadcasting
        expert_weights = expert_weights.view(num_batch, self.num_experts, 1, 1)
        # [batch, num_experts, 1, 1]

        # Weighted sum of expert outputs
        weighted_output = (expert_outputs * expert_weights).sum(dim=1)  # [batch, num_gene, 1]
        predictions = weighted_output.squeeze(2)  # [batch, num_gene]

        # Store expert weights for load balancing loss
        self._last_expert_weights = expert_weights.squeeze(-1).squeeze(-1)  # [batch, num_experts]
        self._last_expert_indices = expert_indices

        return predictions, cell_hidden


class MultiDCP_CheMoEBase(nn.Module):
    """
    Base class for MultiDCP_CheMoE variants (with loss functions)
    """
    def __init__(self, device, model_param_registry):
        super(MultiDCP_CheMoEBase, self).__init__()
        self.model = MultiDCP_CheMoE(device, model_param_registry)
        self.device = device
        self.initializer = model_param_registry.get('initializer', None)

    def init_weights(self, pretrained=None):
        """
        Initialize model weights with diverse expert initialization

        Args:
            pretrained: Path to pretrained weights (not supported in initial version)
        """
        print('Initializing MultiDCP_CheMoE weights...')

        if pretrained:
            print(f'WARNING: Pretrained weights not supported for CheMoE yet: {pretrained}')
            print('Training from scratch with random initialization')

        # Diverse initialization for experts to prevent collapse
        # Each expert gets different random seed
        for expert_idx, expert in enumerate(self.model.experts):
            torch.manual_seed(42 + expert_idx)  # Different seed per expert
            for param in expert.parameters():
                if param.dim() == 1:
                    nn.init.constant_(param, 10**-7)
                elif param.dim() >= 2:
                    # Xavier initialization with expert-specific gain
                    gain = 1.0 + 0.1 * expert_idx  # Slight variation per expert
                    nn.init.xavier_uniform_(param, gain=gain)

        # Initialize other components normally
        if self.initializer is not None:
            for name, parameter in self.named_parameters():
                if 'experts' not in name:  # Skip experts (already initialized)
                    if parameter.dim() == 1:
                        nn.init.constant_(parameter, 10**-7)
                    else:
                        self.initializer(parameter)

        print('Weights initialized successfully with diverse expert initialization')

    def forward(self, input_drug, input_gene, mask, input_cell_gex,
                input_pert_idose, job_id='perturbed', epoch=0):
        """Forward pass through the model"""
        # job_id parameter for compatibility with MultiDCP_AE training script
        # CheMoE doesn't use autoencoder mode, always predicts perturbed expression
        return self.model(input_drug, input_gene, mask, input_cell_gex,
                         input_pert_idose, epoch)

    def loss(self, label, predict, load_balance_weight=0.01):
        """
        Combined loss: MSE + Load Balancing

        Args:
            label: [batch, num_gene] true gene expression
            predict: [batch, num_gene] predicted gene expression
            load_balance_weight: weight for load balancing loss (default: 0.01)
        Returns:
            total_loss: scalar combined loss
        """
        # Main prediction loss
        mse_loss = F.mse_loss(predict, label)

        # Load balancing loss: encourage equal expert usage
        # Compute importance (fraction of total weight) per expert
        if hasattr(self.model, '_last_expert_weights'):
            expert_weights = self.model._last_expert_weights  # [batch, num_experts]

            # Mean importance per expert across batch
            importance = expert_weights.mean(dim=0)  # [num_experts]

            # Target: equal usage (1/num_experts for each)
            num_experts = importance.size(0)
            target_importance = torch.ones_like(importance) / num_experts

            # L2 distance from uniform distribution
            load_balance_loss = F.mse_loss(importance, target_importance)

            # Combined loss
            total_loss = mse_loss + load_balance_weight * load_balance_loss
        else:
            total_loss = mse_loss

        return total_loss


# Alias for compatibility with training scripts
MultiDCP_CheMoE_AE = MultiDCP_CheMoEBase
