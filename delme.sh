#!/bin/bash
cd /home/ani/code/SD-GNN
source venvSD/bin/activate

# ZINC ablations (arch_24)
gxl -c configs/arch_24/ZINC/ablations/no_inter_conv.json
gxl -c configs/arch_24/ZINC/ablations/no_bfs_pe.json
gxl -c configs/arch_24/ZINC/ablations/no_logp_pe.json
gxl -c configs/arch_24/ZINC/ablations/no_inter_no_pe.json

# MolHIV HT ablations
gxl -c configs/arch_24/OGB/molhiv_ht_inter.json
gxl -c configs/arch_24/OGB/molhiv_ht_pool.json

# Multi-seed ZINC
gxl -c configs/arch_24/ZINC/m32.json --multi-seed --seeds 42 43 44 45
