import argparse
import os
import sys
import torch
import numpy as np
from invertible_deep_prior import InvertibleDeepPrior
from utils import git_root
np.random.seed(19)
torch.manual_seed(19)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--max_itr', dest='max_itr', type=int, default=501)
parser.add_argument('--sigma', dest='sigma', type=float, default=0.01)
parser.add_argument('--lr', dest='lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', dest='weight_decay', type=float,
                    default=1e-6)
parser.add_argument('--experiment', dest='experiment',
                    default='invertible-deep-prior')
parser.add_argument('--cuda', dest='cuda', type=int, default=0)

args = parser.parse_args()

def main():

    args.save_path = os.path.join(git_root(os.getcwd()), 'plots',
                                  args.experiment)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.cuda and not torch.cuda.is_available():
        print("ERROR: Cuda is not available, try running on CPU")
        sys.exit(1)

    model = InvertibleDeepPrior(args)
    model.train(args)

if __name__ == '__main__':
    main()
