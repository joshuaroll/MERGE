#!/usr/bin/env python
import torch
import numpy as np

# Load PDGrapher format data
print("Loading data_forward_A549.pt...")
data = torch.load("/raid/home/public/chemoe_collab_102025/PDGrapher/data/processed/torch_data/chemical/real_lognorm/data_forward_A549.pt")

print(f"Type: {type(data)}")
if hasattr(data, 'keys'):
    print(f"Keys: {list(data.keys())[:10]}")
elif hasattr(data, '__len__'):
    print(f"Length: {len(data)}")
    if len(data) > 0:
        item = data[0]
        print(f"First element type: {type(item)}")
        # Check if it's a PyG Data object
        if hasattr(item, 'x'):
            print(f"  x shape: {item.x.shape if hasattr(item.x, 'shape') else 'N/A'}")
        if hasattr(item, 'y'):
            print(f"  y shape: {item.y.shape if hasattr(item.y, 'shape') else 'N/A'}")
        if hasattr(item, 'healthy'):
            print(f"  healthy shape: {item.healthy.shape if hasattr(item.healthy, 'shape') else 'N/A'}")
        if hasattr(item, 'diseased'):
            print(f"  diseased shape: {item.diseased.shape if hasattr(item.diseased, 'shape') else 'N/A'}")
        if hasattr(item, 'treated'):
            print(f"  treated shape: {item.treated.shape if hasattr(item.treated, 'shape') else 'N/A'}")
        # List all attributes
        attrs = [a for a in dir(item) if not a.startswith('_')]
        print(f"  All attributes: {attrs[:15]}")
