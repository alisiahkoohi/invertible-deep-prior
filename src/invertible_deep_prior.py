import os
import torch
import time
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from hint import HINT
from utils import get_model
sns.set_style("whitegrid")
font = {'family': 'serif',
        'style': 'normal',
        'size': 10}
matplotlib.rc('font', **font)
sfmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
sfmt.set_powerlimits((0, 0))
matplotlib.use("Agg")


class InvertibleDeepPrior(object):
    def __init__(self, args):

        if torch.cuda.is_available() and args.cuda:
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

    def train(self, args):

        x, spacing, shape = get_model()
        self.x = x.to(self.device)
        self.extent = np.array([0., self.x.shape[2]*spacing[0],
                                self.x.shape[3]*spacing[1], 0.])/1.0e3

        z = torch.randn([1, 1] + shape).to(self.device)
        self.y = self.x + args.sigma*torch.randn(self.x.shape).to(self.device)

        G = HINT(shape, n_blocks=8, hidden_layer_sizes=64,
                 scaling=0.01).to(self.device)

        optim = torch.optim.Adam(G.weights, lr=args.lr,
                                 weight_decay=args.weight_decay)

        self.obj_log = []
        self.err_log = []

        with tqdm(range(args.max_itr), unit=" itr's", colour='#B5F2A9') as pb:
            for itr in pb:

                self.xhat = G(z)

                obj = torch.norm(self.xhat - self.y)**2
                obj.backward()

                optim.step()
                optim.zero_grad()

                if itr%100 == 0 or itr == args.max_itr - 1:
                    self.test(args, itr)

                self.obj_log.append(obj.item())
                self.err_log.append((torch.norm(self.x - self.xhat)**2).item())
                pb.set_postfix(obj="{:.2e}".format(self.obj_log[-1]),
                               error="{:.2e}".format(self.err_log[-1]))


    def test(self, args, itr):

        fig = plt.figure("Objective", figsize=(7, 2.5))
        plt.semilogy(self.obj_log, label="objective")
        plt.semilogy(self.err_log, label="error")
        plt.legend()
        plt.title("Training objecjtive and prediction error")
        plt.xlabel("Iterations")
        plt.ylabel("Loss vs model error")
        plt.grid(True)
        plt.savefig(os.path.join(args.save_path, "training_obj.png"),
                    format="png", bbox_inches="tight", dpi=200,
                    pad_inches=.05)
        plt.close(fig)

        fig = plt.figure("predeiction", figsize=(7, 2.5))
        plt.imshow(self.xhat.cpu().detach().squeeze().numpy().T,
                   vmin=-.04, vmax=.04, aspect=1,
                   extent=self.extent, cmap='seismic', alpha=1.0,
                   resample=True, interpolation="lanczos", filterrad=1)
        plt.colorbar(fraction=0.085, pad=0.01, format=sfmt)
        plt.xlabel("Horizontal distance (km)")
        plt.ylabel("Depth (km)")
        plt.grid(False)
        plt.title("Prediction after " + str(itr) + " updates");
        plt.savefig(os.path.join(args.save_path, "xhat_" +
                    str(itr) + ".png"), format="png",
                    bbox_inches="tight", dpi=200,
                    pad_inches=.05)
        plt.close(fig)

        if itr == 0:
            fig = plt.figure("true model", figsize=(7, 2.5))
            plt.imshow(self.x.cpu().detach().squeeze().numpy().T,
                       vmin=-.04, vmax=.04, aspect=1,
                       extent=self.extent, cmap='seismic', alpha=1.0,
                       resample=True, interpolation="lanczos", filterrad=1)
            plt.colorbar(fraction=0.085, pad=0.01, format=sfmt)
            plt.xlabel("Horizontal distance (km)")
            plt.ylabel("Depth (km)")
            plt.title("True model")
            plt.grid(False)
            plt.savefig(os.path.join(args.save_path, "x.png"),format="png",
                        bbox_inches="tight", dpi=200,
                        pad_inches=.05)
            plt.close(fig)

            fig = plt.figure("observed data", figsize=(7, 2.5))
            plt.imshow(self.y.cpu().detach().squeeze().numpy().T,
                       vmin=-.04, vmax=.04, aspect=1,
                       extent=self.extent, cmap='seismic', alpha=1.0,
                       resample=True, interpolation="lanczos", filterrad=1)
            plt.colorbar(fraction=0.085, pad=0.01, format=sfmt)
            plt.xlabel("Horizontal distance (km)")
            plt.ylabel("Depth (km)")
            plt.title("Observed data")
            plt.grid(False)
            plt.savefig(os.path.join(args.save_path, "y.png"),format="png",
                        bbox_inches="tight", dpi=200,
                        pad_inches=.05)
            plt.close(fig)
