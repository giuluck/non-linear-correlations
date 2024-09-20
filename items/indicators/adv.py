from dataclasses import dataclass, field
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import trange

from items.indicators.indicator import KernelsHGR

DEVICE: torch.device = torch.device(str("cuda:0") if torch.cuda.is_available() else "cpu")
"""The torch device on which to run the adversarial networks."""

EPSILON: float = 0.000000001
"""The tolerance used to standardize the results."""


@dataclass(frozen=True, eq=False)
class AdversarialHGR(KernelsHGR):
    """Torch-based implementation of the HGR-NN indicator."""

    epochs: int = field(init=True, default=1000)
    """The number of epochs used to run the adversarial networks."""

    pretrained_epochs: int = field(init=True, default=50)
    """The number of epochs used to run the adversarial networks when they are pretrained."""

    @property
    def name(self) -> str:
        return 'nn'

    @property
    def configuration(self) -> Dict[str, Any]:
        return dict(name=self.name, epochs=self.epochs)

    @staticmethod
    def compute_with_networks(a: torch.Tensor,
                              b: torch.Tensor,
                              net_1: nn.Module,
                              net_2: nn.Module,
                              epochs: int) -> Tuple[torch.Tensor, nn.Module, nn.Module]:
        """Computes the HGR coefficient and returns it along with the adversarial kernel networks."""
        # model_F is linked to yhat and model_G is linked to s_var
        # in order to retrieve compatible kernel functions, we pass net2 on model_F and net1 on modelG
        model = HGR_NN(model_F=net_2, model_G=net_1, device=DEVICE, display=False)
        correlation = model(yhat=b, s_var=a, nb=epochs)
        return correlation, net_1, net_2

    def _kernels(self, a: np.ndarray, b: np.ndarray, experiment: Any) -> Tuple[np.ndarray, np.ndarray]:
        a = torch.tensor(a, dtype=torch.float32).reshape((-1, 1))
        b = torch.tensor(b, dtype=torch.float32).reshape((-1, 1))
        fa = experiment['f'](a).numpy(force=True).flatten()
        gb = experiment['g'](b).numpy(force=True).flatten()
        return fa, gb

    def correlation(self, a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
        # use the default adversarial networks when computing correlations for correlation experiments
        correlation, net_1, net_2 = AdversarialHGR.compute_with_networks(
            a=torch.tensor(a, dtype=torch.float),
            b=torch.tensor(b, dtype=torch.float),
            net_1=Net_HGR(),
            net_2=Net2_HGR(),
            epochs=self.epochs
        )
        correlation = correlation.numpy(force=True).item()
        return dict(correlation=float(correlation), f=net_1, g=net_2)

    def __call__(self, a: torch.Tensor, b: torch.Tensor, kwargs: Dict[str, Any]) -> torch.Tensor:
        def standardize(t: torch.Tensor) -> torch.Tensor:
            t_std, t_mean = torch.std_mean(t, correction=0)
            return (t - t_mean) / (t_std + EPSILON)

        # set default kwargs in case they are not set (and overwrite them for next steps)
        kwargs['net_1'] = kwargs.get('net_1', Net_HGR())
        kwargs['net_2'] = kwargs.get('net_2', Net2_HGR())
        kwargs['epochs'] = kwargs.get('epochs', self.epochs)
        _, net_1, net_2 = self.compute_with_networks(a=a.detach(), b=b.detach(), **kwargs)
        # eventually, replace the number of epochs to the pretrained epochs for the next training step
        kwargs['epochs'] = self.pretrained_epochs
        f = net_1(a.reshape(-1, 1))
        g = net_2(b.reshape(-1, 1))
        return torch.mean(standardize(f) * standardize(g))


# The following code is obtained from the official repository of
# "Fairness-Aware Neural Renyi Minimization for Continuous Features"
# (https://github.com/fairml-research/HGR_NN/)

H = 16
H2 = 8


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, H)
        self.fc2 = nn.Linear(H, H)
        self.fc3 = nn.Linear(H, H2)
        self.fc4 = nn.Linear(H2, 1)
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self, x):
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        h3 = torch.tanh(self.fc3(h2))
        h4 = self.fc4(h3)
        return h4


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(1, H)
        self.fc2 = nn.Linear(H, H)
        self.fc3 = nn.Linear(H, H2)
        self.fc4 = nn.Linear(H2, 1)
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self, x):
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        h3 = torch.tanh(self.fc3(h2))
        h4 = self.fc4(h3)
        return h4


class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(82, 64)
        self.fc2 = nn.Linear(64, 8)
        self.fc3 = nn.Linear(8, 4)
        self.fc4 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        # x = F.dropout(x, p=0.8)
        x = self.fc2(x)
        x = F.relu(x)
        # x = F.dropout(x, p=0.8)
        x = self.fc3(x)
        x = F.relu(x)
        # x = F.dropout(x, p=0.8)
        x = self.fc4(x)
        return x


# noinspection PyRedeclaration
H = 15
# noinspection PyRedeclaration
H2 = 15


# noinspection PyPep8Naming,DuplicatedCode
class Net_HGR(nn.Module):
    def __init__(self):
        super(Net_HGR, self).__init__()
        self.fc1 = nn.Linear(1, H)
        self.fc2 = nn.Linear(H, H)
        self.fc3 = nn.Linear(H, H2)
        self.fc4 = nn.Linear(H2, 1)
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self, x):
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        h3 = torch.tanh(self.fc3(h2))
        h4 = torch.tanh(self.fc4(h3))
        return h4


# noinspection PyPep8Naming,DuplicatedCode
class Net2_HGR(nn.Module):
    def __init__(self):
        super(Net2_HGR, self).__init__()
        self.fc1 = nn.Linear(1, H)
        self.fc2 = nn.Linear(H, H)
        self.fc3 = nn.Linear(H, H2)
        self.fc4 = nn.Linear(H2, 1)
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self, x):
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        h3 = torch.tanh(self.fc3(h2))
        h4 = torch.tanh(self.fc4(h3))
        return h4


# model_Net_F = Net_HGR()
# model_Net_G = Net2_HGR()


# noinspection PyPep8Naming,PyUnusedLocal,DuplicatedCode
class HGR_NN(nn.Module):

    def __init__(self, model_F, model_G, device, display):
        super(HGR_NN, self).__init__()
        # self.mF = model_Net_F
        # self.mG = model_Net_G
        self.mF = model_F
        self.mG = model_G
        self.device = device
        self.optimizer_F = torch.optim.Adam(self.mF.parameters(), lr=0.0005)
        self.optimizer_G = torch.optim.Adam(self.mG.parameters(), lr=0.0005)
        self.display = display

    def forward(self, yhat, s_var, nb):

        # svar = Variable(torch.FloatTensor(np.expand_dims(s_var, axis=1))).to(self.device)
        # yhatvar = Variable(torch.FloatTensor(np.expand_dims(yhat, axis=1))).to(self.device)
        svar = s_var.reshape((-1, 1)).to(self.device)
        yhatvar = yhat.reshape((-1, 1)).to(self.device)

        self.mF.to(self.device)
        self.mG.to(self.device)

        for j in range(nb):

            pred_F = self.mF(yhatvar)
            pred_G = self.mG(svar)

            epsilon = 0.000000001

            pred_F_norm = (pred_F - torch.mean(pred_F)) / torch.sqrt((torch.std(pred_F).pow(2) + epsilon))
            pred_G_norm = (pred_G - torch.mean(pred_G)) / torch.sqrt((torch.std(pred_G).pow(2) + epsilon))

            ret = torch.mean(pred_F_norm * pred_G_norm)
            loss = - ret  # maximize
            self.optimizer_F.zero_grad()
            self.optimizer_G.zero_grad()
            loss.backward()

            if (j % 100 == 0) and (self.display is True):
                print(j, ' ', loss)

            self.optimizer_F.step()
            self.optimizer_G.step()

        # noinspection PyUnboundLocalVariable
        return ret


# noinspection PyPep8Naming,PyUnusedLocal
def FairQuant(s_test, y_test, y_predt_np):
    d = {'sensitivet': s_test, 'y_testt': y_test, 'y_pred3t': y_predt_np}
    df = pd.DataFrame(data=d)
    vec = []
    for i in np.arange(0.02, 1.02, 0.02):
        tableq = df[df.sensitivet <= df.quantile(i)['sensitivet']]
        av_BIN = tableq.y_pred3t.mean()
        av_Glob = df.y_pred3t.mean()
        vec = np.append(vec, (av_BIN - av_Glob))
    FairQuantabs50 = np.mean(np.abs(vec))
    FairQuantsquare50 = np.mean(vec ** 2)
    # print(FairQuantabs50)
    return FairQuantabs50


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


# noinspection PyPep8Naming,PyUnboundLocalVariable,PyAttributeOutsideInit,PyUnusedLocal,DuplicatedCode
class FAIR_HGR_NN(torch.nn.Module):

    def __init__(self, regressor, mod_h, lr, p_device, nbepoch, lambdaHGR, nbepochHGR, start_epochHGR, mod_HGR_F,
                 mod_HGR_G, init_HGR):
        super().__init__()
        self.lr = lr
        self.device = torch.device(p_device)
        self.nbepoch = int(nbepoch)
        self.nbepochHGR = int(nbepochHGR)
        self.model_h = mod_h()
        self.lambdaHGR = lambdaHGR
        self.nbepochHGR = int(nbepochHGR)
        self.start_epochHGR = int(start_epochHGR)
        self.mF = mod_HGR_F()
        self.mG = mod_HGR_G()
        self.init_HGR = init_HGR

        if regressor == 'mse':
            self.criterion = torch.nn.MSELoss(reduction='mean')
        elif regressor == 'rmse':
            self.criterion = RMSELoss()

    def predict(self, X_train):
        x_var = Variable(torch.FloatTensor(X_train.values)).to(self.device)
        yhat = self.model_h(x_var)
        return yhat

    def fit(self, X_train, y_train, s_train):

        self.optimizer_h = torch.optim.Adam(self.model_h.parameters(), lr=self.lr)
        self.model_h.to(self.device)

        self.optimizer_F = torch.optim.Adam(self.mF.parameters(), lr=self.lr)
        self.mF.to(self.device)

        self.optimizer_G = torch.optim.Adam(self.mG.parameters(), lr=self.lr)
        self.mG.to(self.device)

        epsilon = 0.000000001

        loss = 0
        ypred_var = 0
        t = trange(self.nbepoch + 1, desc='Bar desc', leave=True)

        s_var = Variable(torch.FloatTensor(np.expand_dims(s_train, axis=1))).to(self.device)
        x_var = Variable(torch.FloatTensor(X_train.values)).to(self.device)
        y_var = Variable(torch.FloatTensor(np.expand_dims(y_train, axis=1))).to(self.device)

        for epoch in t:  # tqdm(range(1, self.nbepoch + 1), 'Epoch: ', leave=False):

            # Mini batch learning
            t.set_description("Bar desc (file %i)" % epoch)
            t.refresh()  # to show immediately the update
            ret = 0

            # Forward + Backward + Optimize
            if epoch < self.start_epochHGR:  # Initial predictor model that makes sense before mitigating
                self.optimizer_h.zero_grad()
                ypred_var = self.model_h(x_var)
                loss = self.criterion(ypred_var, y_var)
                loss.backward()
                self.optimizer_h.step()
                y_pred_np = ypred_var.cpu().detach().numpy().squeeze(1)
                # print(y_pred_np[:5])
                # if epoch%5==0:
                #    print(rdc(y_pred_np, s_train))
            if epoch >= self.start_epochHGR:

                ypred_var0 = ypred_var.detach()

                if self.init_HGR is True:
                    if epoch == self.start_epochHGR:
                        nbepHGR = 1000
                    else:
                        nbepHGR = self.nbepochHGR
                else:
                    nbepHGR = self.nbepochHGR

                for j in range(nbepHGR):
                    self.optimizer_F.zero_grad()
                    self.optimizer_G.zero_grad()

                    pred_F = self.mF(ypred_var0)
                    pred_G = self.mG(s_var)

                    pred_F_norm = (pred_F - torch.mean(pred_F)) / torch.sqrt((torch.std(pred_F).pow(2) + epsilon))
                    pred_G_norm = (pred_G - torch.mean(pred_G)) / torch.sqrt((torch.std(pred_G).pow(2) + epsilon))
                    # pred_F_norm[torch.isnan(pred_F_norm )] = 0
                    # pred_G_norm[torch.isnan(pred_G_norm )] = 0

                    ret = torch.mean(pred_F_norm * pred_G_norm)
                    lossHGR = - ret  # maximize

                    lossHGR.backward()

                    self.optimizer_F.step()
                    self.optimizer_G.step()

                self.optimizer_h.zero_grad()
                ypred_var = self.model_h(x_var)
                pred_F = self.mF(ypred_var)
                pred_G = self.mG(s_var)

                pred_F_norm = (pred_F - torch.mean(pred_F)) / torch.sqrt((torch.std(pred_F).pow(2) + epsilon))
                pred_G_norm = (pred_G - torch.mean(pred_G)) / torch.sqrt((torch.std(pred_G).pow(2) + epsilon))
                ret = torch.mean(pred_F_norm * pred_G_norm)
                # print('self.lambdaHGR*ret :',self.lambdaHGR*ret)
                # print('self.criterion(ypred_var, y_var)',self.criterion(ypred_var, y_var).cpu().detach().numpy() )
                loss = self.criterion(ypred_var, y_var) + self.lambdaHGR * ret  # **2
                loss.backward()
                self.optimizer_h.step()

                y_pred_np = ypred_var.cpu().detach().numpy().squeeze(1)
                # if epoch%5==0:
                #    print(rdc(y_pred_np, s_train))

        return y_pred_np  # print('DONE')
