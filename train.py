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
    generate = True
    seed, rng, loader_params = u.initSeed(generate=generate)
    device = u.device()

    output_dir = u.mkdir(["Outputs", "Current"], cwd=True, reset=True)
    params_file = output_dir / "params.txt"
    u.printHeaderText("Using random seed", seed)
    u.printHeaderText("Using random seed", seed, file=params_file)

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
        "lstm" : 8,
        "local" : True
    }
    filters_global = {
        'continentExp' : 'Europe'
    }
    filters_local = {
        'countriesAndTerritories' : 'Norway',
        'continentExp' : 'Europe'
    }

    x_train, x_test, y_train, y_test = dh.mkTrainTest(use_mode, in_cols, out_cols,
                                                      filters_global, filters_local,
                                                      df, ratio)
    in_dim, out_dim = x_train.shape[-1], 2
    batch_size = x_train.shape[0] if use_mode["local"] else 128

    dh.mkDataloader(x_train, y_train, cache=True, batch_size=batch_size,
                    name="train", use_mode=use_mode, loader_params=loader_params)

    dh.mkDataloader(x_test, y_test, shuffle=False, cache=True,
                    name="test", use_mode=use_mode)
    # n_pop = torch.tensor(60e6, dtype=torch.float32, device=device)

    # =============================================================================
    # Hyperparameter Tuning with Optuna
    # =============================================================================
    skip_tune = True if use_mode["lstm"] is not None else False
    params_tune = ParamsTune(in_dim, out_dim, dh, use_mode, u, seed, skip_tune)
    params_tune.init()
    best_params = params_tune.bestParams()
    u.printHeaderText("Best hyperparameters", best_params)
    u.printHeaderText("Best hyperparameters", best_params, file=params_file)

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
                    data_helper=dh, utils=u, n_epochs=n_epochs, output_dir=output_dir)

    loss_history = runner.train(no_print=False)

    # Print the learned epidemiological parameters
    beta, gamma = runner.params("beta").item(), runner.params("gamma").item()
    u.printHeader(f"Learned β: {beta:.4f}, Learned γ: {gamma:.4f}")
    u.printHeader(f"Learned β: {beta:.4f}, Learned γ: {gamma:.4f}", file=params_file)

    plotter = Plotter(runner.predict, use_mode, u, output_dir, params_file)
    plotter.plotData(x_test, y_test, loss_history)
