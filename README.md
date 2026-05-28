# MADGNN: Multi-Agent Task Offloading for Vehicular Edge Computing in High-Mobility Scenarios

This repository contains the implementation used in the undergraduate thesis **“面向高速移动场景的车联网边缘计算多智能体任务卸载方法研究——基于动态图神经网络与 MAD4PG 的联合任务卸载与资源分配”**.

The project extends the original MAD4PG-based VEC task offloading codebase with:

- a **mobility-aware dynamic environment** that models vehicle boundary crossing risk,
- a **Dynamic Graph Neural Network (DynGNN)** module for spatiotemporal topology encoding,
- a **MADGNN** framework that fuses graph embeddings with local observations for multi-agent task offloading,
- a **MAStaGNN** variant that keeps graph-enhanced decision making but disables temporal graph-state updating.

The repository is mainly used to run comparative experiments among **MAD4PG**, **MAStaGNN**, and **MADGNN** under both the basic environment and the mobility-aware dynamic environment.

---

## 1. Project Overview

The task setting is a vehicular edge computing (VEC) scenario with:

- **9 edge nodes** (agents),
- **27 vehicles**,
- **300 discrete time slots** in one episode,
- task offloading as the main RL decision variable,
- transmission power allocation and computation resource allocation solved inside the environment after the offloading action is given.

Two environment settings are used in the main experiments:

- **BasENV**: basic environment without explicit vehicle boundary-crossing penalty.
- **DynENV**: mobility-aware dynamic environment with handover risk and boundary-related service instability.

Main methods:

- **MAD4PG**: original DRL baseline without any GNN module.
- **MAStaGNN**: static graph variant. It uses graph snapshots and fused graph-enhanced state input, but does **not** use previous graph states for temporal updating.
- **MADGNN**: dynamic graph variant. It uses graph snapshots together with temporal memory updating to model spatiotemporal topology evolution.

---

## 2. Recommended Environment Setup

The recommended Conda environment file is:

- `gtdrl-tf28gpu.yml`

Create and activate the environment:

```bash
conda env create -f gtdrl-tf28gpu.yml
conda activate gtdrl-tf28gpu
```

If your runtime reports a library-path issue, you may temporarily fix it with:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
```

In some older deployments, the project may still contain hard-coded library paths. In that case, check the environment file or runtime shell settings before launching experiments.

---

## 3. Dataset and Environment Files

The trajectory-driven VEC environments are built from vehicular trajectory data derived from the **Didi GAIA Open Data Set** after preprocessing.

In normal use, you do **not** need to rebuild the environment. You can directly use an existing serialized environment file (`.pkl`) by editing the `environment_file_name` field inside the corresponding run script.

Typical workflow:

1. prepare or reuse an existing environment `.pkl` file,
2. open the corresponding run script,
3. replace `environment_file_name` with the correct absolute path on the current server,
4. run the experiment.

Example:

```python
environment_file_name = "/root/projects/Game-Theoretic-Deep-Reinforcement-Learning-main/saved_env/.../convex_environment_xxx.pkl"
```

If you really need to regenerate the environment, you may also need to:

- modify the trajectory CSV path in `environmentConfig.py`,
- modify the save path in `Utilities/FileOperator.py`,
- then rebuild and save the new environment.

---

## 4. Main Experiment Entry Scripts

The main experiment scripts currently used in this repository are:

### 4.1 MAD4PG in BasENV

```text
Experiment/run_mad4pg.py
```

This script runs the original MAD4PG baseline in the basic environment. The uploaded version uses `Environment.environment_old` and sets the environment path manually inside the script.

### 4.2 MAD4PG in DynENV

```text
Experiment/run_mad4pg_dynenv.py
```

This script runs the original MAD4PG baseline in the mobility-aware dynamic environment and uses the dynamic-environment action/observation specification.

### 4.3 GNN-based experiment in BasENV

```text
Experiment/run_mad4pg_gnn.py
```

This script is used for graph-enhanced experiments in the basic environment.

In the uploaded version, it is currently configured with:

- `gnn_is_dynmic=False`

Therefore, this script already corresponds to the **MAStaGNN-style** setting in BasENV.

### 4.4 GNN-based experiment in DynENV

```text
Experiment/run_mad4pg_gnn_dynenv.py
```

This script is used for graph-enhanced experiments in the mobility-aware dynamic environment.

In the uploaded version, it is currently configured with:

- `gnn_is_dynmic=True`

So this script corresponds to the **MADGNN** setting in DynENV.

If you want to run **MAStaGNN in DynENV**, simply change:

```python
gnn_is_dynmic=True
```

to:

```python
gnn_is_dynmic=False
```

inside `run_mad4pg_gnn_dynenv.py`.

---

## 5. How to Switch Between MADGNN and MAStaGNN

The difference between **MADGNN** and **MAStaGNN** is controlled mainly by one flag in the GNN run scripts:

```python
gnn_is_dynmic=True
```

- `True`  -> dynamic graph updating enabled -> **MADGNN**
- `False` -> temporal updating disabled -> **MAStaGNN**

So, for GNN-based experiments:

- use `gnn_is_dynmic=True` for **MADGNN**,
- use `gnn_is_dynmic=False` for **MAStaGNN**.

No separate MAStaGNN-specific run script is strictly required if the rest of the configuration remains the same.

---

## 6. Main Hyperparameters Used in Thesis Experiments

The thesis experiments mainly use the following shared settings (unless otherwise stated in ablation experiments):

- `agent_number = 9`
- `num_actors = 10`
- `batch_size = 256`
- `prefetch_size = 4`
- `min_replay_size = 1000`
- `max_replay_size = 1000000`
- `samples_per_insert = 8.0`
- `n_step = 1`
- `discount = 0.996`
- `target_update_period = 50`
- `variable_update_period = 300`
- `max_actor_steps = 20000 * 300`
- `log_every = 5.0`

Typical GNN-related settings in the final experiments:

- `gnn_updater_type = 'gru'`
- standard **MADGNN**: lightweight dynamic graph setting
- standard **MAStaGNN**: same graph-enhanced pipeline but temporal graph-state updating disabled

Please note that some uploaded run scripts may still contain older intermediate experimental values. When reproducing the final thesis experiments, you should check the exact parameter values before launching.

---

## 7. How to Launch Experiments on a Remote Server

The project is typically run on cloud servers such as Alibaba Cloud or Vast.ai.

### 7.1 Foreground run

```bash
python -m Experiment.experiment 2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).txt
```

### 7.2 Background run

```bash
mkdir -p logs
LOGFILE=logs/train_mad4pg_$(date +%Y%m%d_%H%M%S).txt
nohup python -m Experiment.experiment > "$LOGFILE" 2>&1 &
echo $! > logs/latest_train.pid
echo "PID: $!"
echo "LOG: $LOGFILE"
```

### 7.3 Check running process

```bash
ps -ef | grep "python -m Experiment.experiment" | grep -v grep
ps -ef | grep python
```

### 7.4 Kill a process

```bash
kill <PID>
```

If you use a different entry script instead of `Experiment.experiment`, replace the module path accordingly.

---

## 8. Repository Structure

A simplified view of the repository is as follows:

```text
Agents/
  MAD4PG/
  MAD4PG_DynENV/
  MAD4PG_GNN/
  MAD4PG_GNN_DynENV/
Environment/
Experiment/
Utilities/
saved_env/
```

- `Agents/` contains the agent, learner, actor, and network implementations.
- `Environment/` contains the basic and dynamic VEC environments.
- `Experiment/` contains the main run scripts.
- `Utilities/` contains helper modules such as file loading/saving.
- `saved_env/` stores serialized environment files used by experiments.

---

## 9. Practical Notes

1. **Always check the environment path** before launching a run.
   The run scripts use manually assigned absolute paths for `environment_file_name`.

2. **Do not assume the uploaded scripts already use the final thesis configuration.**
   Some scripts may still keep intermediate debugging or ablation settings.

3. **For MAStaGNN experiments, the quickest way is to modify `gnn_is_dynmic`.**
   In most cases, you do not need to rewrite the training pipeline.

4. **DynENV requires the dynamic-environment-compatible action and observation specification.**
   So make sure you use the correct environment module and corresponding run script.

---

## 10. Suggested Reproduction Order

If you want to reproduce the core thesis experiments, the recommended order is:

1. Run **MAD4PG in BasENV**
2. Run **MAD4PG in DynENV**
3. Run **MAStaGNN in BasENV**
4. Run **MAStaGNN in DynENV**
5. Run **MADGNN in DynENV**
6. Run additional ablation experiments by changing GNN structure parameters

This order is convenient for verifying:

- the effect of the dynamic environment itself,
- the benefit of graph modeling,
- the additional value of temporal graph-state updating.

---

## 11. Notes on This Repository Version

This repository is no longer a pure baseline MAD4PG code release. It has been adapted into a thesis-oriented experimental codebase centered on **MADGNN**, and the README is written according to the final thesis setting rather than the original paper-only version.


