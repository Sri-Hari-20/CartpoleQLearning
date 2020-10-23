import argparse
import os

from src.cartpole import Cartpole


# Paths to respective storing locations
BASE_PATH = os.path.join(os.getcwd(), 'store')
NPY_PATH = os.path.join(BASE_PATH, 'qtable')
VID_PATH = os.path.join(BASE_PATH, 'recording')

# Verify the above folders exist and if not create them
if not os.path.isdir(NPY_PATH):
    os.makedirs(NPY_PATH)

if not os.path.isdir(VID_PATH):
    os.makedirs(VID_PATH)

# For parsing arguments
parser = argparse.ArgumentParser(
    description='Program for Q Learning demonstration using CartPole environment.')
required_named = parser.add_argument_group('Required arguments')

# Add the arguments
required_named.add_argument(
    '-s', '--save', help='Store the video footage of execution.', action='store_true')
required_named.add_argument(
    '-f', '--file', type=str, help='npy file name for running')

# Choose train or run subgroup
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-t', '--train', action='store_true')
group.add_argument('-r', '--run', action='store_true')

# Parsing arguments
args = parser.parse_args()

# Check for conditions
# If run mode, need file
if args.run:
    if args.file is None:
        parser.error(
            'npy file is needed for run mode, if no files are present, train first!')
    else:
        # npy file must exist
        if not os.path.isfile(os.path.join(NPY_PATH, args.file)):
            parser.error('Specified weights file does not exist!')
elif args.train:
    if args.file is not None:
        print(
            "Warning: File is required for run mode not training. Ignoring.")


# If all these checks go through main starts
if __name__ == "__main__":
    agent = Cartpole(args)
    print("Info: Agent created")
    if args.train:
        print("Info: Training initiated")
        agent.train()
    elif args.run:
        print("Info: Running initiated")
        agent.run()
