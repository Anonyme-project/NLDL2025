import os
from pathlib import Path
import ast
from contextlib import nullcontext

from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import optuna
from optuna.samplers import TPESampler

from Helpers.loss import PhyLoss
from Helpers.models import getModel
from Helpers.utils import Utils

def printStats(x: torch.Tensor):
    mean, std = x.mean().item(), x.std().item()
    min, max = x.min().item(), x.max().item()

    print(f"shape: {x.shape}")
    print(f"mean, std: {(mean, std)}")
    print(f"min, max: {(min, max)}")

class Runner:
    def __init__(self, lambda_phy, lr, use_mode, model_params,
                 data_helper, utils=None, n_epochs=1000, output_dir=None):
        self.lambda_phy = lambda_phy
        self.lr = lr

        self.dh = data_helper
        self.u = Utils() if utils is None else utils
        self.n_epochs = n_epochs
        self.epoch_start = 0

        self.output_dir = output_dir
        if self.output_dir is not None:
            self.ckpt_dir = self.output_dir / "Checkpoints"
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.ckpt_dir = None

        self.use_mode = use_mode
        self.m_params = model_params
        self.model_fn = getModel
        self._model = None
        self.optim_fn = optim.Adam
        self.optim = None
        self.loss_fn = None

        self.initModel()

    def initModel(self):
        device = self.u.device()
        self._model = self.model_fn(self.use_mode, self.m_params).to(device)
        self.m_params["beta"] = nn.Parameter(torch.tensor(0.5, device=device))
        self.m_params["gamma"] = nn.Parameter(torch.tensor(0.1, device=device))
        self.optim = self.optim_fn(
            list(self._model.parameters())
            + [self.m_params["beta"], self.m_params["gamma"]], lr=self.lr
        )
        self.loss_fn = PhyLoss(self.m_params, use_mode=self.use_mode)

    def train(self, no_print=False):
        """
        Train the PINN-SIR model using both data loss and physics-informed loss.
        """
        n_epochs = self.n_epochs
        loss_history = []
        for epoch in range(self.epoch_start + 1, n_epochs + 1):
            has_ckpt, ckpt = self.hasCkpt(epoch)
            if has_ckpt:
                self.loadCkpt(ckpt)
                epoch_loss, epoch_loss_data, epoch_loss_phy = (
                    ckpt["loss"], ckpt["loss_data"], ckpt["loss_phy"]
                )
                # epoch_loss, epoch_loss_data, epoch_loss_phy = self.predictEpoch_(no_print=no_print)
            else:
                epoch_loss, epoch_loss_data, epoch_loss_phy = self.trainEpoch_(no_print=no_print)
            loss_history.append(epoch_loss)
            if epoch == 1 or epoch % 10 == 0:
                epoch_msg = (
                    f"Epoch {epoch}, Total Loss: {epoch_loss:.6f}"
                    + f", Data Loss: {epoch_loss_data:.6f}"
                    + f", Physics Loss: {epoch_loss_phy:.6f}"
                )
                print(epoch_msg)
            if (not has_ckpt) and (epoch == 1 or epoch % 100 == 0):
                self.save_(epoch, dict(loss=epoch_loss, loss_data=epoch_loss_data,
                                       loss_phy=epoch_loss_phy))
        return loss_history

    def trainEpoch_(self, no_print=False):
        device = self.u.device()
        model = self._model
        model.train()
        optim = self.optim
        loss_fn = self.loss_fn
        loader = self.dh.dataloader("train")
        n_batches = len(loader)
        epoch_loss, epoch_loss_data, epoch_loss_phy = 0.0, 0.0, 0.0
        with tqdm(loader, unit="batch") if not no_print else nullcontext() as tepoch:
            if tepoch is None: tepoch = loader
            else: tepoch.set_description("Training")
            for batch_idx, xy in enumerate(tepoch):
                x, y = xy
                x, y = x.to(device), y.to(device)
                optim.zero_grad()

                # Data loss: match the predicted cumulative cases (I + R) to the actual data
                use_cudnn = self.use_mode["lstm"] is None
                x.requires_grad = True
                preds_I, preds_R = model(x, use_cudnn=use_cudnn)
                preds = preds_I + preds_R

                cum_cases = y[:, 0:1]
                n_pop = y[:, 1:2]
                # print(f"{preds_I.shape=}")
                # print(f"{preds_R.shape=}")
                # self.u.printHeader("Stats for preds_I")
                # printStats(preds_I)
                # self.u.printHeader("Stats for preds_R")
                # printStats(preds_R)
                # self.u.printHeader("Stats for preds")
                # printStats(preds)
                # self.u.printHeader("Stats for cum_cases")
                # printStats(cum_cases)
                # self.u.printHeader("Stats for n_pop")
                # printStats(n_pop)
                # input()

                data_loss, physics_loss = loss_fn(preds, cum_cases, preds_I, preds_R, x, n_pop)
                loss = data_loss + self.lambda_phy * physics_loss
                loss.backward()
                optim.step()

                batch_loss, data_loss, physics_loss = loss.item(), data_loss.item(), physics_loss.item()
                epoch_loss += batch_loss
                epoch_loss_data += data_loss
                epoch_loss_phy += physics_loss

                # self.u.printHeader(f"Batch #{batch_idx}")
                # self.u.printHeaderText("batch_loss, data_loss, physics_loss", (batch_loss, data_loss, physics_loss))
                # input()

        epoch_loss = epoch_loss/n_batches
        epoch_loss_data = epoch_loss_data/n_batches
        epoch_loss_phy = epoch_loss_phy/n_batches

        return epoch_loss, epoch_loss_data, epoch_loss_phy

    def test(self, with_outputs=False, with_grad=False):
        model = self._model
        model.eval()
        loss_fn = self.loss_fn
        loader = self.dh.dataloader("test")
        epoch_loss, epoch_loss_data, epoch_loss_phy = 0.0, 0.0, 0.0
        no_grad_fn = torch.no_grad if not with_grad else nullcontext
        with no_grad_fn():
            with tqdm(loader, unit="batch") as tepoch:
                tepoch.set_description("Testing")
                for batch_idx, xy in enumerate(tepoch):
                    x, y = xy
                    # print(f"{x.shape}")
                    # print(f"{y.shape}")
                    # Data loss: match the predicted cumulative cases (I + R) to the actual data
                    use_cudnn = self.use_mode["lstm"] is None
                    x.requires_grad = True
                    preds_I, preds_R = model(x, use_cudnn=use_cudnn)
                    preds = preds_I + preds_R

                    data_loss, physics_loss = loss_fn(preds, y[:, 0:1], preds_I, preds_R, x, y[:, 1:2], with_phy=with_grad)
                    loss = data_loss + self.lambda_phy * physics_loss

                    batch_loss, data_loss, physics_loss = loss.item(), data_loss.item(), physics_loss.item()
                    epoch_loss += batch_loss
                    epoch_loss_data += data_loss
                    epoch_loss_phy += physics_loss

        if with_outputs:
            return epoch_loss, epoch_loss_data, epoch_loss_phy, preds_I, preds_R
        else:
            return epoch_loss, epoch_loss_data, epoch_loss_phy

    def predict(self, x):
        model = self._model
        model.eval()
        self.u.printHeader("Testing")
        with torch.no_grad():
            preds_I, preds_R = model(x)

        return preds_I, preds_R

    def predictEpoch_(self, no_print=False):
        device = self.u.device()
        model = self._model
        model.eval()
        loss_fn = self.loss_fn
        loader = self.dh.dataloader("train")
        n_batches = len(loader)
        epoch_loss, epoch_loss_data, epoch_loss_phy = 0.0, 0.0, 0.0
        with tqdm(loader, unit="batch") if not no_print else nullcontext() as tepoch:
            if tepoch is None: tepoch = loader
            else: tepoch.set_description("Predict Training")
            for batch_idx, xy in enumerate(tepoch):
                x, y = xy
                x, y = x.to(device), y.to(device)

                # Data loss: match the predicted cumulative cases (I + R) to the actual data
                use_cudnn = self.use_mode["lstm"] is None
                x.requires_grad = True
                preds_I, preds_R = model(x, use_cudnn=use_cudnn)
                preds = preds_I + preds_R

                cum_cases = y[:, 0:1]
                n_pop = y[:, 1:2]

                data_loss, physics_loss = loss_fn(preds, cum_cases, preds_I, preds_R, x, n_pop)
                loss = data_loss + self.lambda_phy * physics_loss

                batch_loss, data_loss, physics_loss = loss.item(), data_loss.item(), physics_loss.item()
                epoch_loss += batch_loss
                epoch_loss_data += data_loss
                epoch_loss_phy += physics_loss

        epoch_loss = epoch_loss/n_batches
        epoch_loss_data = epoch_loss_data/n_batches
        epoch_loss_phy = epoch_loss_phy/n_batches

        return epoch_loss, epoch_loss_data, epoch_loss_phy

    def save_(self, epoch, loss):
        if self.ckpt_dir is None: return

        model = self._model
        optim = self.optim
        epoch_loss, epoch_loss_data, epoch_loss_phy = list(loss.values())
        ckpt_path = self.ckpt_dir / f'ckpt_dict_epoch-{epoch}_loss-{epoch_loss:.4f}.pt'
        state_info = dict(epoch=epoch,
                          loss=epoch_loss,
                          loss_data=epoch_loss_data,
                          loss_phy=epoch_loss_phy,
                          model=model.state_dict(),
                          optimizer=optim.state_dict())
        torch.save(state_info, ckpt_path)

    def loadLatest(self):
        if self.ckpt_dir is None: return

        ckpts = self.ckpts()
        if not ckpts: return
        # print(ckpts)
        # exit()

        device = self.u.device()
        ckpt = ckpts[-1]
        ckpt = torch.load(ckpt, map_location=device)
        self._model.load_state_dict(ckpt['model'])
        self.optim.load_state_dict(ckpt['optimizer'])
        self.epoch_start = ckpt['epoch']

    def loadCkptEpoch(self, epoch):
        if self.ckpt_dir is None: return

        ckpts = self.ckpts()
        if not ckpts: return
        # print(ckpts)
        # exit()

        found = None
        for ckpt in ckpts:
            epoch_ckpt = self.getValEpoch_(ckpt)
            if epoch_ckpt == epoch:
                found = ckpt
                break

        if found is None: return
        device = self.u.device()
        ckpt = torch.load(found, map_location=device)
        self.epoch_start = ckpt['epoch']
        self._model.load_state_dict(ckpt['model'])
        self.optim.load_state_dict(ckpt['optimizer'])

    def loadCkpt(self, ckpt):
        self.epoch_start = ckpt['epoch']
        self._model.load_state_dict(ckpt['model'])
        self.optim.load_state_dict(ckpt['optimizer'])

    def hasCkpt(self, epoch):
        if self.ckpt_dir is None: return False, None

        ckpts = self.ckpts()
        if not ckpts: return False, None
        # print(ckpts)
        # exit()

        found = None
        for ckpt in ckpts:
            epoch_ckpt = self.getValEpoch_(ckpt)
            if epoch_ckpt == epoch:
                found = ckpt
                break

        ckpt = None
        if found is not None:
            device = self.u.device()
            ckpt = torch.load(found, map_location=device)

        return (found is not None), ckpt

    def getValEpoch_(self, name):
        val = os.path.basename(name)
        val = val.split("_")[2].split("epoch-")[1]
        val = int(val)
        return val

    def ckpts(self):
        files = []
        if os.path.isdir(self.ckpt_dir) and any(Path(self.ckpt_dir).iterdir()):
            files = [os.path.join(self.ckpt_dir, fn)
                     for fn in next(os.walk(self.ckpt_dir))[2]
                     if fn.endswith(".pt")]
            files = sorted(files,
                           key=lambda x: self.getValEpoch_(x))
        return files

    def model(self):
        return self._model

    def lossFn(self):
        return self.loss_fn

    def params(self, name=None):
        if name is not None: return self.m_params[name]

        return self.m_params

    def loadParams(self, params_file):
        device = self.u.device()
        with open(params_file, 'r') as rfile:
            content = rfile.read()

        start_marker = "Learned β:"
        start_index = content.find(start_marker)
        if start_index == -1:
            raise ValueError("Could not find the best hyperparameters section in the file.")
        start_index = start_index + len(start_marker)
        end_index = content.find(",", start_index)

        beta = content[start_index:end_index].strip()
        beta = float(beta)
        self.m_params["beta"] = nn.Parameter(torch.tensor(beta, device=device))

        start_marker = "Learned γ:"
        start_index = content.find(start_marker)
        if start_index == -1:
            raise ValueError("Could not find the best hyperparameters section in the file.")
        start_index = start_index + len(start_marker)

        gamma = content[start_index:].strip()
        gamma = gamma.splitlines()[0].strip()
        gamma = float(gamma)
        self.m_params["gamma"] = nn.Parameter(torch.tensor(gamma, device=device))

class Objective:
    def __init__(self, in_dim, out_dim, data_helper, use_mode, utils=None):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dh = data_helper
        self.use_mode = use_mode
        self.u = Utils() if utils is None else utils

    def __call__(self, trial):
        # Suggest hyperparameters
        n_layers = trial.suggest_int('n_layers', 2, 4)
        h_dim = trial.suggest_int('h_dim', 10, 50)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        lambda_phy = trial.suggest_float('lambda_phy', 1e-2, 1e2, log=True)
        n_epochs = 500  # Fewer epochs for tuning speed

        # Define network architecture: input dimension is 1, output dimension is 2
        model_params = {
            "in_dim" : self.in_dim,
            "out_dim" : self.out_dim,
            "h_dim" : h_dim,
            "n_layers" : n_layers,
        }

        # Train the model
        self.u.printHeader(f"Finding the optimal parameters with {n_epochs} epochs.")
        r = Runner(lambda_phy, lr, use_mode=self.use_mode, model_params=model_params,
                   data_helper=self.dh, utils=self.u, n_epochs=n_epochs)
        r.train(no_print=True)

        val_loss, _, _ = r.test(with_grad=True)
        return val_loss

class ParamsTune:
    def __init__(self, in_dim, out_dim, dh, use_mode, utils, seed=None, skip=False):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dh = dh
        self.use_mode = use_mode
        self.u = utils
        self.seed = seed
        self.skip = skip
        self.best_params = None

    def init(self):
        if self.skip: self.initDefault()
        else: self.initOptuna()

    def initDefault(self):
        self.best_params = {
            "n_layers" : 3,
            "h_dim" : 128,
            "lr" : 1e-4,
            "lambda_phy" : 3e-2,
        }

    def initOptuna(self):
        seed = self.seed
        in_dim = self.in_dim
        out_dim = self.out_dim
        dh = self.dh
        use_mode = self.use_mode
        u = self.u
        study_sampler = TPESampler(seed=seed)
        study = optuna.create_study(direction="minimize", sampler=study_sampler)
        objective = Objective(in_dim, out_dim, dh, use_mode=use_mode, utils=u)
        study.optimize(objective, n_trials=20)
        self.best_params = study.best_params

    def bestParams(self):
        return self.best_params

    def loadParams(self, params_file):
        with open(params_file, 'r') as rfile:
            content = rfile.read()

        start_marker = "Best hyperparameters"
        end_marker = "----------------------------------------"

        start_index = content.find(start_marker)
        end_index = content.find(end_marker, start_index + len(start_marker))

        if start_index == -1 or end_index == -1:
            raise ValueError("Could not find the best hyperparameters section in the file.")

        start_index = end_index + len(end_marker)
        hyperparams_section = content[start_index:].strip()
        hyperparams_section = hyperparams_section.splitlines()[0]

        # Find the dictionary string within the section
        dict_start = hyperparams_section.find('{')
        dict_end = hyperparams_section.find('}') + 1

        if dict_start == -1 or dict_end == -1:
            raise ValueError("Could not find the dictionary in the hyperparameters section.")

        dict_str = hyperparams_section[dict_start:dict_end]

        # Convert the dictionary string to a Python dictionary
        best_params = ast.literal_eval(dict_str)

        self.best_params = best_params

