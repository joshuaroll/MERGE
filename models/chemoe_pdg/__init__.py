"""
CheMoE adapted for PDGrapher's 10,716-gene prediction task.

This module implements the CheMoE (Chemical Mixture of Experts) architecture
adapted for PDGrapher data, featuring:
- 4 expert networks with sparse top-k=2 routing
- Global gating based on drug + cell + basal expression features
- Gene-aware expert processing for efficient 10,716-gene prediction
- Load balancing loss to prevent expert collapse
"""

from .model import CheMoE_PDG, CELL_LINE_TO_IDX, IDX_TO_CELL_LINE

__all__ = ['CheMoE_PDG', 'CELL_LINE_TO_IDX', 'IDX_TO_CELL_LINE']
