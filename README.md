# Multi-Agent Guided Policy Search (MA-GPS) for Ergodic Search

This repository builds on the [Multi-Agent Guided Policy Search (MA-GPS)](https://arxiv.org/pdf/2509.24226) framework by J. Li\*, G. Qu\*, J. Choi, S. Sojoudi, and C. Tomlin, which learns Nash equilibrium policies for multi-agent non-cooperative dynamic games using model-based LQ game guidance to stabilize multi-agent policy gradients.

**Our goal** is to extend MA-GPS to the problem of **multi-agent ergodic search** — training multiple agents to learn coverage policies over a target spatial distribution, where agents may have competing objectives. This combines game-theoretic multi-agent RL with ergodic control theory to enable intelligent, coordinated (or adversarial) exploration of a search space.

## Current Status

We have reproduced the two benchmark experiments from the original MA-GPS paper:

- **Three-Vehicle Unicycle Platooning** — three cars learn to merge into a single-file platoon (Figure 2b in the paper)
- **Six-Player Basketball Formation** — six players from two teams learn strategic positioning near the basket (Figure 3b in the paper)

All six algorithm-environment combinations (MA-GPS, MAPPO, MADDPG × both environments) have been trained and verified. Trained checkpoints are included in `experiment_script/log/`.

Next steps involve designing a new Gym environment for multi-agent ergodic search, with a reward function based on the [ergodic metric](https://www.sciencedirect.com/science/article/pii/S016727891000285X) (Mathew & Mezić, 2011), and training agents using the MA-GPS framework.

## Repository Structure

- `MAGPS/` — Core library (policy implementations, environments, training infrastructure)
  - `MARL_gym_envs/` — Custom Gym environments for general-sum dynamic games
  - `policy/gym_marl_policy/` — IPPO, MAPPO, MADDPG, and MA-GPS policy implementations
- `experiment_script/` — Training scripts, evaluation notebooks, and trained checkpoints
  - `run_magps.py`, `run_mappo.py`, `run_maddpg.py` — Training entry points
  - `eval_magps_three_unicycle.ipynb` — Evaluation notebook for unicycle platooning
  - `eval_magps_six_basketball.ipynb` — Evaluation notebook for basketball formation
  - `log/` — Trained model checkpoints
- `notebooks/` — Ergodic search tutorials
  - `ergodic_metric.ipynb` — Introduction to the ergodic metric and Fourier basis functions
  - `smc_ergodic_control.ipynb` — Spectral Multiscale Coverage (SMC) closed-form ergodic control
  - `kernel_ergodic_control.ipynb` — Fast ergodic search with kernel functions (JAX)

## Installation

Our implementation builds upon the deep RL infrastructure of [Tianshou](https://github.com/thu-ml/tianshou) (version 0.5.1).

### Standard Setup (Ubuntu/Linux with CUDA)

```bash
git clone https://github.com/codingwithshawnyt/ma-gps-ergodic-search.git
cd ma-gps-ergodic-search
conda create -n magps python=3.10 -y
conda activate magps
pip install -e .
conda install -c conda-forge ffmpeg
```

### macOS Setup

On macOS, PyTorch 2.4.0 is not available. Before running `pip install -e .`, edit `setup.py` line 22:

```
# Change this:
"torch==2.4.0"
# To this:
"torch==2.2.2"
```

Then proceed with the standard install steps above.

### Older NVIDIA Drivers (CUDA < 12.x)

If your GPU machine has an older NVIDIA driver (e.g., driver 470.x / CUDA 11.4), PyTorch 2.4.0 will silently fall back to CPU. In this case, install PyTorch separately before the editable install:

```bash
conda create -n magps python=3.10 -y
conda activate magps
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113
```

Then loosen the version pins in `setup.py`:
- `"torch==2.4.0"` → `"torch>=1.12.1"`
- `"numpy==1.26.4"` → `"numpy>=1.22.0,<1.25.0"`
- `"gymnasium==0.28.1"` → `"gymnasium>=0.26.0,<0.29.0"`

And run `pip install -e .`.

## Training

From the `experiment_script/` directory:

```bash
# Three unicycle platooning (MA-GPS)
python run_magps.py --task Three_Unicycle_Game-v0 \
    --critic-net 512 512 512 --actor-net 512 512 512 \
    --epoch 15 --total-episodes 160 --gamma 0.99 \
    --behavior-loss-weight 0.1 --batch-size 2048

# Six-player basketball (MA-GPS)
python run_magps.py --task basketball-v0 \
    --critic-net 512 512 512 --actor-net 512 512 512 \
    --epoch 15 --total-episodes 160 --gamma 0.99 \
    --behavior-loss-weight 0.1 --batch-size 2048
```

Replace `run_magps.py` with `run_mappo.py` or `run_maddpg.py` to train with those algorithms.

To specify a GPU: add `--device cuda:0` (or `cuda:1`, etc.).

## Evaluation

Open the evaluation notebooks in `experiment_script/`:

```bash
cd experiment_script
jupyter notebook
```

- `eval_magps_three_unicycle.ipynb` — loads a trained policy and visualizes car platooning trajectories
- `eval_magps_six_basketball.ipynb` — loads a trained policy and visualizes basketball player formations

Set `EPOCH_TO_LOAD` at the top of each notebook to select which checkpoint to evaluate.

## References

- J. Li, G. Qu, J. Choi, S. Sojoudi, C. Tomlin. [Multi-Agent Guided Policy Search for Non-Cooperative Dynamic Games](https://arxiv.org/pdf/2509.24226), 2025.
- G. Mathew, I. Mezić. [Metrics for Ergodicity and Design of Ergodic Dynamics for Multi-Agent Systems](https://www.sciencedirect.com/science/article/pii/S016727891000285X), Physica D, 2011.

## Acknowledgments

This work is part of research conducted at the CMU Biorobotics Lab under Prof. Howie Choset, in collaboration with Darwin Mick.
