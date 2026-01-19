# Phase 5: CheMoE PDGrapher Adaptation - Research

**Researched:** 2026-01-19
**Domain:** CheMoE Mixture-of-Experts architecture adaptation for 10,716-gene prediction
**Confidence:** HIGH

## Summary

CheMoE is a Mixture-of-Experts model for predicting gene expression responses to chemical perturbations. The original model predicts 978 L1000 genes; this adaptation scales to PDGrapher's 10,716 genes while preserving the core MoE architecture. The repository already has successful patterns from TranSiGen-CheMoE (`models/transigen_chemoe/model.py`) and MultiDCP-CheMoE (`models/multidcp_chemoe/model.py`) that demonstrate proper MoE integration with gene expression prediction.

Key findings:
1. Original CheMoE uses 4 experts with top-k=2 sparse gating
2. Expert architecture: concatenated embeddings -> MLP -> output_dim
3. Gating based on global features (drug + cell + dose + basal expression)
4. PDGrapher data lacks dose variation (all 'x'), requiring fixed embedding or removal
5. Can reuse `moe_modules.py` for gating/expert components

**Primary recommendation:** Create a standalone CheMoE_PDG model (`models/chemoe_pdg/model.py`) following the TranSiGen_CheMoE pattern, but with CheMoE's architecture (basal expression as input, categorical cell embedding, 4 modal experts).

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.x | Model implementation | Project standard |
| RDKit | Latest | Morgan fingerprint generation | CheMoE original |
| KPGT embeddings | 2304-dim | Molecular embeddings | TranSiGen pattern, pre-computed |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| moe_modules.py | Local | MoELayer, FusionMoE | Consistent MoE implementation |
| data_loader.py | Local | PDGrapherDataLoader, TopKEvaluator | Shared evaluation |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Morgan FP (1024-dim) | KPGT (2304-dim) | Morgan is CheMoE native, KPGT better for similarity |
| Dose embedding | Skip dose | PDGrapher has no dose variation |

## Architecture Patterns

### Recommended Project Structure
```
models/chemoe_pdg/
    __init__.py             # Export CheMoE_PDG
    model.py                # Main model class
train_chemoe_pdg.py         # Training script following train_transigen_pdg.py pattern
```

### Pattern 1: CheMoE Core Architecture
**What:** Global gating on multimodal features, 4 experts predict gene expression
**When to use:** This is the defining CheMoE pattern

**Original CheMoE forward pass (from HeirCheMoE.py SparseCheMoE):**
```python
# Encode inputs to embed_dim each
drug_embed = self.smiles_encoder(smiles)           # [batch, 128]
dose_embed = self.dose_encoder(dose).squeeze(1)    # [batch, 128]
cell_embed = self.cell_encoder(cell).squeeze(1)    # [batch, 128]
basal_embed = self.basal_encoder(basal_gex)        # [batch, 128]

# Combine for gating
combined = torch.cat([drug_embed, dose_embed, cell_embed, basal_embed], dim=-1)  # [batch, 512]

# Sparse gating
gate_weights = self.gating(combined)  # [batch, num_experts]
top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)
top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

# Expert processing (4 experts, each predicts full output_dim)
output = torch.zeros(batch_size, output_dim, device=combined.device)
for i in range(self.top_k):
    expert_idx = top_k_indices[:, i]
    expert_output = torch.stack([self.experts[idx](combined[j:j+1])
                                  for j, idx in enumerate(expert_idx)], dim=0).squeeze(1)
    output += top_k_weights[:, i:i+1] * expert_output

return output
```

### Pattern 2: PDGrapher Adaptation (MultiDCP-CheMoE Style)
**What:** Gene-aware expert processing instead of global expert
**When to use:** When predicting 10,716 genes (more efficient than global expert)

**Recommended for CheMoE_PDG (inspired by multidcp_chemoe/model.py):**
```python
# Expert processes [batch, num_gene, input_dim] -> [batch, num_gene, 1]
# Instead of [batch, input_dim] -> [batch, output_dim]

# Expand global features to gene dimension
global_features_expanded = global_features.unsqueeze(1).repeat(1, self.num_gene, 1)
# [batch, num_gene, global_dim]

# Concatenate with gene embeddings (optional)
# gene_embed: [batch, num_gene, gene_emb_dim]
expert_input = torch.cat([global_features_expanded, gene_embed], dim=2)
# [batch, num_gene, global_dim + gene_emb_dim]

# Expert output per gene
expert_out = expert(expert_input)  # [batch, num_gene, 1]
```

### Pattern 3: Dual Encoder Support (Morgan FP + KPGT)
**What:** Support both molecular representations via CLI flag
**When to use:** Per CONTEXT.md decision

```python
class MolecularEncoder(nn.Module):
    def __init__(self, encoder_type='kpgt', embed_dim=128):
        super().__init__()
        self.encoder_type = encoder_type

        if encoder_type == 'morgan':
            # Morgan fingerprint input: 1024-dim
            self.encoder = nn.Sequential(
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, embed_dim)
            )
        else:  # kpgt
            # KPGT embedding input: 2304-dim
            self.encoder = nn.Sequential(
                nn.Linear(2304, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, embed_dim)
            )

    def forward(self, x):
        return self.encoder(x)
```

### Anti-Patterns to Avoid
- **Per-sample expert loop:** CheMoE original loops per sample for sparse routing. Use vectorized ops instead.
- **No load balancing:** Expert usage can collapse without auxiliary loss.
- **Dose embedding with no variation:** PDGrapher data has no dose info; using dummy dose wastes capacity.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| MoE gating | Custom gating | `moe_modules.py` MoELayer | Has load balancing, tested |
| Top-k evaluation | Custom loop | `data_loader.py` TopKEvaluator | Consistent with other baselines |
| Data loading | Custom Dataset | `PDGrapherDataset` from train_transigen_pdg.py | Handles SMILES to embedding |
| Morgan fingerprints | RDKit from scratch | Existing pattern in HeirCheMoE.py | Proven implementation |

**Key insight:** The repo already has `moe_modules.py` that provides `SparseGatingNetwork`, `MoELayer`, and `FusionMoE`. Use these rather than reimplementing.

## Common Pitfalls

### Pitfall 1: Expert Collapse
**What goes wrong:** One or two experts dominate, others never used
**Why it happens:** Without load balancing, gating network finds local minimum
**How to avoid:** Include load balancing auxiliary loss (weight ~0.01)
**Warning signs:** Expert usage stats show >60% on single expert

### Pitfall 2: Dose Encoding with No Variation
**What goes wrong:** PDGrapher data has `pert_idose = 'x'` for all samples (no dose information)
**Why it happens:** PDGrapher aggregated across doses
**How to avoid:** Either (a) remove dose encoding entirely, or (b) use fixed learned embedding
**Warning signs:** Dose encoder learns nothing, wastes parameters

### Pitfall 3: Memory Explosion with Gene-Aware Experts
**What goes wrong:** `[batch, 10716, 434] * 4 experts` consumes massive memory
**Why it happens:** Gene-aware processing scales with num_genes
**How to avoid:** Use shared expert MLP (same weights across genes), not per-gene parameters
**Warning signs:** CUDA OOM on forward pass

### Pitfall 4: Inconsistent Evaluation
**What goes wrong:** Model seems better/worse due to metric mismatch
**Why it happens:** Different top-k selection or metric computation
**How to avoid:** Use `TopKEvaluator` from `data_loader.py` exactly as other baselines
**Warning signs:** Results don't compare to other models

## Code Examples

### CheMoE_PDG Model Structure (Recommended)
```python
class CheMoE_PDG(nn.Module):
    """
    CheMoE adapted for PDGrapher's 10,716-gene prediction.

    Architecture (per CONTEXT.md):
    - Drug: Morgan FP (1024) or KPGT (2304) -> embed_dim
    - Cell: Learned categorical embedding (9 cell lines) -> embed_dim
    - Basal: MLP encoder (10716 genes) -> embed_dim
    - (No dose - PDGrapher data lacks variation)

    Gating: Global features -> 4 experts, top-k=2
    Experts: Shared MLP per gene position
    Output: 10,716 genes
    """

    def __init__(self, n_genes=10716, embed_dim=128, num_experts=4, top_k=2,
                 mol_encoder_type='kpgt', num_cell_lines=9, dropout=0.1, device='cuda'):
        super().__init__()

        self.n_genes = n_genes
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = top_k

        # === Encoders ===
        # Molecular encoder (KPGT or Morgan)
        mol_input_dim = 2304 if mol_encoder_type == 'kpgt' else 1024
        self.mol_encoder = nn.Sequential(
            nn.Linear(mol_input_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Cell line embedding (categorical)
        self.cell_embedding = nn.Embedding(num_cell_lines, embed_dim)

        # Basal expression encoder
        self.basal_encoder = nn.Sequential(
            nn.Linear(n_genes, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Optional: learnable gene embeddings
        self.gene_embedding = nn.Embedding(n_genes, embed_dim)

        # === MoE Components ===
        # Global dim: drug (128) + cell (128) + basal (128) = 384
        self.global_dim = embed_dim * 3

        # Gating network
        self.gating = nn.Sequential(
            nn.Linear(self.global_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_experts)
        )

        # Expert input: global (384) + gene_emb (128) = 512
        expert_input_dim = self.global_dim + embed_dim

        # 4 Expert networks (shared MLP across genes)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_input_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1)
            )
            for _ in range(num_experts)
        ])

        self._last_expert_weights = None

    def forward(self, basal_expr, mol_embed, cell_idx):
        """
        Args:
            basal_expr: [batch, n_genes] diseased expression
            mol_embed: [batch, mol_dim] KPGT (2304) or Morgan (1024)
            cell_idx: [batch] cell line indices

        Returns:
            pred_treated: [batch, n_genes] predicted treated expression
        """
        batch_size = basal_expr.size(0)
        device = basal_expr.device

        # Encode modalities
        drug_embed = self.mol_encoder(mol_embed)        # [batch, 128]
        cell_embed = self.cell_embedding(cell_idx)      # [batch, 128]
        basal_embed = self.basal_encoder(basal_expr)    # [batch, 128]

        # Global features for gating
        global_features = torch.cat([drug_embed, cell_embed, basal_embed], dim=-1)  # [batch, 384]

        # Sparse gating (top-k=2)
        gate_logits = self.gating(global_features)  # [batch, 4]
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # [batch, 2]

        # Create sparse weight tensor
        expert_weights = torch.zeros(batch_size, self.num_experts, device=device)
        expert_weights.scatter_(1, top_k_indices, top_k_weights)
        self._last_expert_weights = expert_weights  # For load balancing loss

        # Gene embeddings
        gene_indices = torch.arange(self.n_genes, device=device)
        gene_embed = self.gene_embedding(gene_indices)  # [n_genes, 128]
        gene_embed = gene_embed.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, n_genes, 128]

        # Expand global features for gene-aware processing
        global_expanded = global_features.unsqueeze(1).expand(-1, self.n_genes, -1)  # [batch, n_genes, 384]
        expert_input = torch.cat([global_expanded, gene_embed], dim=-1)  # [batch, n_genes, 512]

        # Process through all experts
        expert_outputs = torch.stack([
            expert(expert_input).squeeze(-1) for expert in self.experts
        ], dim=1)  # [batch, 4, n_genes]

        # Weighted combination
        expert_weights_expanded = expert_weights.unsqueeze(-1)  # [batch, 4, 1]
        pred_treated = (expert_outputs * expert_weights_expanded).sum(dim=1)  # [batch, n_genes]

        return pred_treated

    def compute_load_balance_loss(self, weight=0.01):
        """Auxiliary loss to encourage equal expert usage."""
        if self._last_expert_weights is None:
            return torch.tensor(0.0)

        importance = self._last_expert_weights.mean(dim=0)  # [num_experts]
        target = torch.ones_like(importance) / self.num_experts
        return weight * F.mse_loss(importance, target)
```

### Training Script Pattern (from train_transigen_pdg.py)
```python
# Key training loop elements:

# 1. Dataset returns (diseased, treated, mol_embed, idx)
class PDGrapherDataset(Dataset):
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.diseased[idx]),   # basal expression
            torch.FloatTensor(self.treated[idx]),    # target
            torch.FloatTensor(mol_embed),            # KPGT or Morgan
            idx
        )

# 2. Forward pass
pred_treated = model(diseased, mol_embed, cell_idx)

# 3. Loss with load balancing
mse_loss = F.mse_loss(pred_treated, treated)
aux_loss = model.compute_load_balance_loss(weight=0.01)
total_loss = mse_loss + aux_loss

# 4. Evaluation on differential expression
true_de = treated - diseased
pred_de = pred_treated - diseased
metrics = evaluator.compute_metrics(treated, pred_treated, diseased)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Dense gating | Sparse top-k gating | CheMoE original | Better specialization |
| Global expert (978-dim) | Gene-aware expert | MultiDCP-CheMoE | Scales to 10K+ genes |
| Single encoder type | Morgan + KPGT dual support | This phase | Flexibility for experiments |

**Deprecated/outdated:**
- Original CheMoE per-sample expert loop: Replaced with batched vectorized ops
- Dose encoding: Not useful for PDGrapher data (no dose variation)

## Data Pipeline Integration

### PDGrapher Data Format
```
- Total samples: 182,246
- Genes: 10,716
- Cell lines: 10 (A375, A549, BT20, HA1E, HELA, HT29, MCF7, MDAMB231, PC3, VCAP)
- Dose: ALL samples have pert_idose='x' (no variation)
- SMILES: Available in 'smiles' column (KPGT embeddings available for ~8K molecules)
```

### Cell Line to Index Mapping
```python
CELL_LINE_TO_IDX = {
    'A375': 0, 'A549': 1, 'BT20': 2, 'HA1E': 3, 'HELA': 4,
    'HT29': 5, 'MCF7': 6, 'MDAMB231': 7, 'PC3': 8, 'VCAP': 9
}
```

### Molecular Embeddings
- **KPGT (recommended):** 2304-dim, pre-computed at `/raid/home/joshua/projects/cbio_032024/zenodo_TranSiGen/TranSiGen/data/LINCS2020/KPGT_emb2304.pickle` (8,316 molecules)
- **Morgan FP (alternative):** 1024-dim, compute on-the-fly using RDKit (radius=2)

## Open Questions

Things that couldn't be fully resolved:

1. **Cross-Modal Attention Variant**
   - What we know: CheMoE has CrossModalMoE with multi-head attention
   - What's unclear: Whether it improves PDGrapher performance
   - Recommendation: Start with SparseCheMoE style, CrossModalMoE is deferred

2. **Gene Embedding Necessity**
   - What we know: MultiDCP-CheMoE uses gene embeddings, TranSiGen-CheMoE doesn't
   - What's unclear: Impact on 10,716-gene prediction
   - Recommendation: Include gene embeddings (follows MultiDCP-CheMoE pattern)

## Files to Create/Modify

### New Files
| File | Purpose |
|------|---------|
| `models/chemoe_pdg/__init__.py` | Export CheMoE_PDG |
| `models/chemoe_pdg/model.py` | Main model class |
| `train_chemoe_pdg.py` | Training script |

### Modified Files
| File | Change |
|------|--------|
| `models/__init__.py` | Add CheMoE_PDG import |
| `data/topk_r2_results.csv` | Training results appended |

### Reference Files (read-only)
| File | How to Use |
|------|------------|
| `/raid/home/joshua/projects/CheMoE/model/HeirCheMoE.py` | SparseCheMoE architecture reference |
| `models/transigen_chemoe/model.py` | MoE integration pattern |
| `models/multidcp_chemoe/model.py` | Gene-aware expert pattern |
| `models/moe_modules.py` | Reusable MoE components |
| `train_transigen_pdg.py` | Training script pattern |
| `data_loader.py` | TopKEvaluator, PDGrapherDataLoader |

## Risks and Considerations

### Technical Risks
1. **Memory:** Gene-aware processing with 4 experts on 10,716 genes may need batch size reduction
2. **Expert collapse:** Without careful load balancing, experts may collapse
3. **SMILES coverage:** Not all PDGrapher molecules have KPGT embeddings (need fallback)

### Mitigation Strategies
1. Use `gradient_checkpointing` if memory issues arise
2. Log expert usage per epoch, adjust `load_balance_weight` if needed
3. Fall back to zero vector for missing molecules (same as TranSiGen)

## Sources

### Primary (HIGH confidence)
- `/raid/home/joshua/projects/CheMoE/model/HeirCheMoE.py` - SparseCheMoE, CrossModalMoE classes
- `/raid/home/joshua/projects/CheMoE/model/CheMoE.py` - Basic CheMoE architecture
- `/raid/home/joshua/projects/PDGrapher_Baseline_Models/models/transigen_chemoe/model.py` - TranSiGen_CheMoE_PDG
- `/raid/home/joshua/projects/PDGrapher_Baseline_Models/models/multidcp_chemoe/model.py` - MultiDCP_CheMoE

### Secondary (MEDIUM confidence)
- `/raid/home/joshua/projects/CheMoE/CLAUDE.md` - CheMoE project documentation
- `/raid/home/joshua/projects/PDGrapher_Baseline_Models/.planning/phases/05-chemoe-pdg-adaptation/05-CONTEXT.md` - User decisions

### Tertiary (LOW confidence)
- CheMoE training data format analysis (dose encoding uncertain for PDGrapher)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Direct code inspection
- Architecture: HIGH - Source code verified in multiple files
- Pitfalls: HIGH - Based on existing MultiDCP-CheMoE experience in this repo
- Data format: HIGH - Direct pickle inspection

**Research date:** 2026-01-19
**Valid until:** 2026-02-19 (stable codebase, no external dependencies changing)
