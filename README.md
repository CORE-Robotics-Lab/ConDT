# ConDT
Public implementation of the Contrastive Decision Transformers (ConDT), CoRL 2022

# *** COMING SOON ***

# Atari

We build our Atari implementation on top of [minGPT](https://github.com/karpathy/minGPT) and the original [decision 
transformer](https://github.com/kzl/decision-transformer) and benchmark our results on the [DQN-replay](https://github.com/google-research/batch_rl) dataset.

## Installation

Dependencies can be installed with the following command:

```
conda env create -f conda_env_gpu.yml
```

## Downloading datasets

Create a directory for the dataset and load the dataset using [gsutil](https://cloud.google.com/storage/docs/gsutil_install#install). Replace `[DIRECTORY_NAME]` and `[GAME_NAME]` accordingly (e.g., `./dqn_replay` for `[DIRECTORY_NAME]` and `Breakout` for `[GAME_NAME]`)
```
mkdir [DIRECTORY_NAME]
gsutil -m cp -R gs://atari-replay-datasets/dqn/[GAME_NAME] [DIRECTORY_NAME]
```

## Example usage

Script to reproduce ConDT results can be run as (the seed argument defines the seed used for the environment during evaluation):

```
python run_dt_atari.py --seed 123 --block_size 90 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game [GAME_NAME] --batch_size 128 --data_dir_prefix [DIRECTORY_NAME] --model [MODEL_TYPE]
```

## Additional:

We also provide a script, ```tune_simclr.py``` that uses hyper-parameter search to find the optimal settings for the SimRCRL Loss.

============= ================= ================= ====================== ================ ================= ================== ==================


# OpenAI Gym

We build our gym implementation on top of the original [decision transformer](https://github.com/kzl/decision-transformer) and benchmark our results on the following OpenAI Gym environments:
1. Hopper
2. HalfCheetah
3. Walker2D
4. Pen (Adroit Handgrip)
5. Hammer (Adroit Handgrip)
6. Relocate (Adroit Handgrip)

## Installation

Experiments require MuJoCo.
Follow the instructions in the [mujoco-py repo](https://github.com/openai/mujoco-py) to install.
Then, dependencies can be installed with the following command:

```
conda env create -f conda_env_gpu.yml
```

## Downloading datasets

Datasets are stored in the `data` directory.
Install the [D4RL repo](https://github.com/rail-berkeley/d4rl), following the instructions there.
Then, run the following script in order to download the datasets a``nd save them in our format:

```
python download_d4rl_datasets.py
python download_adroid_datasets.py
```

## Example usage

Experiments for ConTran can be reproduced with the following:

```
python experiment.py --env [ENV_NAME] --dataset [DATASET_TYPE] --model_type [MODEL_TYPE]
```

Adding `-w True` will log results to Weights and Biases.

## Additional:

We also provide a script, ```tune_simclr.py``` that uses hyper-parameter search to find the optimal settings for the SimRCRL Loss.

Note: The OpenAI Gym experiments rely upon using Mujoco-2.1.0, Gym==0.18.3, and d4rl==1.1. Using later versions of Gym may result in errors when downloading the datasets because the environment names have changed.





