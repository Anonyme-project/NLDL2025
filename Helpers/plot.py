import numpy as np
import matplotlib.pyplot as plt

def computeMetrics(pred, true):
    """Compute MAE, RMSE, and R-squared metrics."""
    mae = np.mean(np.abs(pred - true))
    rmse = np.sqrt(np.mean((pred - true)**2))
    ss_res = np.sum((true - pred)**2)
    ss_tot = np.sum((true - np.mean(true))**2)
    r2 = 1 - ss_res/ss_tot
    return mae, rmse, r2

class Plotter:
    def __init__(self, predicter, use_mode, utils, output_dir=None, params_file=None):
        self.predict_fn = predicter
        self.use_mode = use_mode
        self.u = utils
        self.output_dir = output_dir
        self.params_file = params_file

    def plotData(self, x_all, y_all, loss_history, metrics_only=False):
        if loss_history is not None:
            self.plotLoss(loss_history)
        for k in x_all:
            x, y = x_all[k], y_all[k]
            preds_I, preds_R = self.predict_fn(x)
            self.printMetrics(y, preds_I, preds_R, name=k)
            if not metrics_only:
                self.plotCurves(x, y, preds_I, preds_R, name=k)

    def plotCurves(self, x, targets, preds_I, preds_R, name="somename"):
        output_dir = self.output_dir
        use_mode = self.use_mode
        use_local = use_mode["local"]
        if output_dir is not None:
            output_dir = output_dir / "Plots" / ("Local" if use_local else "Global")
            output_dir = output_dir / name
            output_dir.mkdir(parents=True, exist_ok=True)

        if use_mode["lstm"] is not None:
            x = x[:, -1, 0]

        # Convert predictions and ground truth to numpy arrays for metric computation
        preds = preds_I + preds_R
        x = x.cpu().numpy()
        preds_I = preds_I.cpu().numpy()[:, 0]
        preds_R = preds_R.cpu().numpy()[:, 0]
        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()

        cum_cases, n_pop = targets[:, 0], targets[:, 1]

        # Plot: Data vs. PINN-SIR predictions over time
        fig, axis = plt.subplots(figsize=(17,10))
        axis.plot(x, cum_cases, 'bo', label="Observed Data")
        # axis.plot(x, preds, 'r-', label="PINN-SIR Prediction")
        axis.plot(x, preds, 'r*', label="PINN-SIR Prediction")
        axis.set_xlabel("Normalized Time")
        axis.set_ylabel("Normalized Cumulative No. of Cases")
        axis.set_title(f"COVID-19 Cumulative Cases in {name}")
        axis.legend()
        if use_local:
            plt.show()
        if output_dir is not None:
            save_filename = str(output_dir / "predictions")
            fig.savefig(save_filename + ".png", dpi=200, bbox_inches='tight')
        plt.close(fig)

        # Plot: Phase Portrait (I vs. R)
        fig, axis = plt.subplots(figsize=(17,10))
        # plt.plot(preds_I, preds_R, 'g-', linewidth=2)
        axis.plot(preds_I, preds_R, 'g*', linewidth=2)
        axis.set_xlabel("Infected (I)")
        axis.set_ylabel("Removed (R)")
        axis.set_title(f"Phase Portrait: I vs R in {name}")
        fig.tight_layout()
        if use_local:
            plt.show()
        if output_dir is not None:
            save_filename = str(output_dir / "I_R_Curve")
            fig.savefig(save_filename + ".png", dpi=200, bbox_inches='tight')
        plt.close(fig)

        # Plot: S, I, R Curve
        fig, axis = plt.subplots(figsize=(17,10))
        S = n_pop - preds_I - preds_R
        self.u.printHeader(f"mean (S, I, R): ({np.mean(S):.3f}, {np.mean(preds_I):.3f}, {np.mean(preds_R):.3f})")
        axis.plot(x, S, marker='*', linewidth=0, label="Susceptible")
        axis.plot(x, preds_I, '+', label="Infected")
        axis.plot(x, preds_R, 'x', label="Removed")
        axis.set_xlabel("Normalized Time")
        axis.set_ylabel("Normalized No. of Cases")
        axis.set_title(f"S, I, R Curve for {name}")
        axis.legend()
        fig.tight_layout()
        if use_local:
            plt.show()
        if output_dir is not None:
            save_filename = str(output_dir / "S_I_R_Curve")
            fig.savefig(save_filename + ".png", dpi=200, bbox_inches='tight')
        plt.close(fig)

    def plotLoss(self, loss_history):
        output_dir = self.output_dir
        use_mode = self.use_mode
        use_local = use_mode["local"]
        if output_dir is not None:
            output_dir = output_dir / "Plots" / ("Local" if use_local else "Global")
            output_dir.mkdir(parents=True, exist_ok=True)

        # Plot: Training loss curve
        fig, axis = plt.subplots(figsize=(17,10))
        axis.plot(loss_history, 'k-')
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.set_title("Training Loss Curve")
        fig.tight_layout()
        plt.show()
        if output_dir is not None:
            save_filename = str(output_dir / "preds_loss")
            fig.savefig(save_filename + ".png", dpi=200, bbox_inches='tight')
        plt.close(fig)

    def printMetrics(self, targets, preds_I, preds_R, name="somename"):
        # Convert predictions and ground truth to numpy arrays for metric computation
        preds = preds_I + preds_R
        preds = preds.cpu().numpy()[:, 0]
        targets = targets.cpu().numpy()
        cum_cases, n_pop = targets[:, 0], targets[:, 1]
        # Compute evaluation metrics on the validation set
        print(f"{preds.shape=}\n{cum_cases.shape=}")
        mae, rmse, r2 = computeMetrics(preds, cum_cases)
        self.u.printHeader(f"{name} Validation MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")
        if self.params_file is not None:
            self.u.printHeader(f"{name} Validation MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}", file=self.params_file)
