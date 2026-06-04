#!/bin/bash
set -e
echo "=== GNN-RouteOpt: Multi-Objective NAS ==="
python -m src.nas_search --config configs/config.yaml
echo "=== Pareto Front Visualization ==="
python -m src.visualize
echo "=== Done ==="
