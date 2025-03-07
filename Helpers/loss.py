import torch

class PhyLoss(torch.nn.Module):
    def __init__(self, m_params, use_mode=None):
        super(PhyLoss, self).__init__()
        self.m_params = m_params
        self.use_mode = use_mode

    def forward(self, preds, targets, preds_I, preds_R, t, n_pop, with_phy=True):
        data_loss = self.dataLoss_(preds, targets)
        phy_loss = self.phyLoss_(preds_I, preds_R, t, n_pop) if with_phy else torch.tensor([0.0])

        return data_loss, phy_loss

    def dataLoss_(self, preds, targets):
        loss = torch.mean((preds - targets)**2)

        return loss

    def phyLoss_(self, preds_I, preds_R, t, n_pop):
        # n_pop = torch.tensor(60e6, dtype=torch.float32, device=preds_I.device)
        # t.requires_grad = True
        S = n_pop - preds_I - preds_R
        if torch.any(S <= 0):
            raise Exception(f"[Error] (PhyLoss::phyLoss_): S less than 0.0\n{S}")

        # Compute time derivatives dI/dt and dR/dt via autograd
        dI_dt = torch.autograd.grad(preds_I, t, grad_outputs=torch.ones_like(preds_I),
                                    create_graph=True, retain_graph=True)[0]
        dR_dt = torch.autograd.grad(preds_R, t, grad_outputs=torch.ones_like(preds_R),
                                    create_graph=True, retain_graph=True)[0]

        if self.use_mode["lstm"] is not None:
            dI_dt = dI_dt[:, -1, 0:1]
            dR_dt = dR_dt[:, -1, 0:1]

        # SIR model equations residuals:
        # dI/dt = beta * S * I / N - gamma * I
        # dR/dt = gamma * I
        beta, gamma = self.m_params["beta"], self.m_params["gamma"]
        # print(f"Learned β: {beta.item():.4f}, Learned γ: {gamma.item():.4f}")
        res_I = dI_dt - (beta * S * preds_I / n_pop - gamma * preds_I)
        res_R = dR_dt - (gamma * preds_I)

        loss = torch.mean(res_I**2) + torch.mean(res_R**2)

        return loss
