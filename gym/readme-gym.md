
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