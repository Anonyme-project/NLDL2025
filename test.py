import sys

import torch

from Helpers.runner import Runner, Objective, ParamsTune
from Helpers.data import DataHandler
from Helpers.plot import Plotter
from Helpers.utils import Utils

# =============================================================================
# Main Script
# =============================================================================
if __name__ == "__main__":

    u = Utils()
    if len(sys.argv) == 1:
        output_dir_run = u.mkdir(["Outputs", "Test"], cwd=True, reset=False)
        params_file = "params.txt"
    else:
        output_dir_run = f"{sys.argv[1]}"
        output_dir_run = u.mkdir([output_dir_run], cwd=True, reset=False)
        params_file = output_dir_run / "params.txt"
    output_dir = u.mkdir(["Outputs", "Test"], cwd=True, reset=False)
    params_file_plot = output_dir / "params.txt"
    seed, rng, loader_params = u.loadSeed(params_file)
    device = u.device()

    u.printHeaderText("Using random seed", seed)

    data_name = "ECDC"
    dh = DataHandler(data_name=data_name, utils=u)
    df_full = dh.loadData()

    u.printHeaderText("countriesAndTerritories", df_full["countriesAndTerritories"].unique())
    u.printHeaderText("continentExp", df_full["continentExp"].unique())

    u.printHeader(f"Data summary for {data_name}")
    dh.summarize(df_full)

    df = dh.extendData(df=df_full)

    in_cols = ['t']
    out_cols = ['cum_cases', 'popData2019']

    ratio = 0.8
    use_mode = {
        "lstm" : None,
        "local" : False
    }
    filters_global = {
        'continentExp' : 'Europe'
    }
    filters_local = {
        'countriesAndTerritories' : 'Italy',
        'continentExp' : 'Europe'
    }

    x_train, x_test, y_train, y_test = dh.mkTrainTest(use_mode, in_cols, out_cols,
                                                      filters_global, filters_local,
                                                      df, ratio)
    in_dim, out_dim = x_train.shape[-1], 2

    # =============================================================================
    # Hyperparameter Tuning with Optuna
    # =============================================================================
    skip_tune = True
    params_tune = ParamsTune(in_dim, out_dim, dh, use_mode, u, seed, skip_tune)
    params_tune.loadParams(params_file)
    best_params = params_tune.bestParams()
    u.printHeaderText("Best hyperparameters", best_params)

    n_layers = best_params['n_layers']
    h_dim = best_params['h_dim']
    lr = best_params['lr']
    lambda_phy = best_params['lambda_phy']
    n_epochs = 1000  # Increase epochs for final training
    model_params = {
        "in_dim" : in_dim,
        "out_dim" : out_dim,
        "h_dim" : h_dim,
        "n_layers" : n_layers,
    }
    runner = Runner(lambda_phy, lr, use_mode=use_mode, model_params=model_params,
                    data_helper=dh, utils=u, n_epochs=n_epochs, output_dir=output_dir_run)
    runner.loadLatest()
    runner.loadParams(params_file)

    # Print the learned epidemiological parameters
    beta, gamma = runner.params("beta").item(), runner.params("gamma").item()
    u.printHeader(f"Learned β: {beta:.4f}, Learned γ: {gamma:.4f}")

    plotter = Plotter(runner.predict, use_mode, u, output_dir, params_file_plot)
    plotter.plotData(x_test, y_test, loss_history=None)
