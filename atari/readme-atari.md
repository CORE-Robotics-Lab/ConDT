
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
