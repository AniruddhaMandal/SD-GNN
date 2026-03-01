# SD-GNN

Graph neural network framework with subgraph sampling for graph, node, and link prediction tasks.

---

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.11 or 3.12 |
| CUDA toolkit | 12.8 (or match your GPU driver) |
| g++ | any modern version |

> **Note:** `torch-scatter` has no pre-built wheels for PyTorch 2.10 and is compiled from source during setup (~10 min). A C++ compiler is required.

---

## Setup

***Download this repository. Run `bash install.sh --cuda CUDA-VERSION`***

This script will:
1. Create a virtual environment at `./venvSD`
2. Install PyTorch 2.10.0 (CUDA 12.8)
3. Build `torch-scatter` from source
4. Install all Python dependencies
5. Compile the graphlet sampler extension (C++/CUDA)
6. Install the `gxl` library in editable mode

### Options

```bash
bash install.sh --cuda cu126              # CUDA 12.6
bash install.sh --cuda cu121              # CUDA 12.1
bash install.sh --cuda cpu                # CPU only
bash install.sh --python python3.12       # use Python 3.12
bash install.sh --cuda cu126 --python python3.12  # combine flags
```

### Activate the environment

```bash
source venvSD/bin/activate
```

---

## Running Experiments

Experiments are configured with JSON files under `configs/`.

```bash
gxl -c configs/sd_gnn/TUData/mutag-gin.json
```

Override config values with `-o key=value`:

```bash
gxl -c configs/sd_gnn/TUData/mutag-gin.json -o train.lr=0.001 model.hidden_dim=128
```

Run with multiple seeds:

```bash
gxl -c configs/sd_gnn/TUData/mutag-gin.json --multi-seed --seeds 0 1 2 42
```

---

## Project Structure

```
configs/       # JSON experiment configs (sd_gnn/, vanilla/, sle_gnn/)
src/
  gxl/         # Main training framework (GXL library)
  samplers/
    rwr_sampler/       # Random-walk with restart sampler (C++)
    ugs_sampler/       # Unbiased graphlet sampler (C++)
    uniform_sampler/   # Uniform subgraph sampler (C++)
    graphlet_sampler/  # Graphlet sampler (CUDA/C++)
data/          # Datasets (downloaded automatically on first run)
experiments/   # Output logs and checkpoints
```
