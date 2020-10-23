# Cartpole Q Learning
Q Learning implementation for the "CartPole-v0" environment of the OpenAIGym using python, complete with arguments functionality

<hr>

## Required packages:
1. gym
2. tqdm
3. matplotlib
4. numpy

<hr>

## Installation:
Its best to have a empty virtual environment using virtualenv or conda. Python 3.7 is tested to work, Python 3.8 should ideally work.

requirements.txt has been provided to install the neccessary packages for functionality.

```pip install -r requirements.txt```

<hr>

## Usage:
The help section of main.py is kind of misleading as argparse treats all the arguments which start with - or -- as optional, so kindly refer to functionality here.

The project has 3 modes: Train, Run and Demo

### 1. Demo: -d or --demo
Runs the environment by choosing actions through random sampling for demonstration purposes.

### 2. Train: -t or --train
Starts training the environment from scratch with an empty Q table and updates values by replaying. Dumps it as a .npy file with timestamp and avg rewards at the end of the training.

### 3. Run: -r or --run
Loads a given .npy file into memory and then uses it to make greedy actions.

#### Required: One filename saved under store/qtable (along with .npy extension). -f file_name or --file file_name

#### Optional: Store the video of the agent under a given weight. -s or --save. Stored under store/recording