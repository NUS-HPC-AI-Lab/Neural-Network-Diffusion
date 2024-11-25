import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def extract(v, i, shape):
    out = torch.gather(v, index=i, dim=0)
    out = out.to(device=i.device, dtype=torch.float32)
    # reshape to (batch_size, 1, 1, 1, 1, ...) for broadcasting purposes.
    out = out.view([i.shape[0]] + [1] * (len(shape) - 1))
    return out


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model: nn.Module, beta: tuple[int, int], T: int):
        super().__init__()
        self.model = model
        self.T = T
        # generate T steps of beta
        self.register_buffer("beta_t", torch.linspace(*beta, T, dtype=torch.float32))
        # calculate the cumulative product of $\alpha$ , named $\bar{\alpha_t}$ in paper
        alpha_t = 1.0 - self.beta_t
        alpha_t_bar = torch.cumprod(alpha_t, dim=0)
        # calculate and store two coefficient of $q(x_t | x_0)$
        self.register_buffer("signal_rate", torch.sqrt(alpha_t_bar))
        self.register_buffer("noise_rate", torch.sqrt(1.0 - alpha_t_bar))

    def forward(self, x_0, z, **kwargs):
        # preprocess nan to zero
        mask = torch.isnan(x_0)
        x_0 = torch.nan_to_num(x_0, 0.)
        # get a random training step $t \sim Uniform({1, ..., T})$
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)
        # generate $\epsilon \sim N(0, 1)$
        epsilon = torch.randn_like(x_0)
        # predict the noise added from $x_{t-1}$ to $x_t$
        x_t = (extract(self.signal_rate, t, x_0.shape) * x_0 +
               extract(self.noise_rate, t, x_0.shape) * epsilon)
        epsilon_theta = self.model(x_t, t, z)
        # get the gradient
        loss = F.mse_loss(epsilon_theta, epsilon, reduction="none")
        loss[mask] = torch.nan
        return loss.nanmean()


class DDPMSampler(nn.Module):
    def __init__(self, model: nn.Module, beta: tuple[int, int], T: int):
        super().__init__()
        self.model = model
        self.T = T
        # generate T steps of beta
        self.register_buffer("beta_t", torch.linspace(*beta, T, dtype=torch.float32))
        # calculate the cumulative product of $\alpha$ , named $\bar{\alpha_t}$ in paper
        alpha_t = 1.0 - self.beta_t
        alpha_t_bar = torch.cumprod(alpha_t, dim=0)
        alpha_t_bar_prev = F.pad(alpha_t_bar[:-1], (1, 0), value=1.0)
        self.register_buffer("coeff_1", torch.sqrt(1.0 / alpha_t))
        self.register_buffer("coeff_2", self.coeff_1 * (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_t_bar))
        self.register_buffer("posterior_variance", self.beta_t * (1.0 - alpha_t_bar_prev) / (1.0 - alpha_t_bar))

    @torch.no_grad()
    def cal_mean_variance(self, x_t, t, c):
        # """ Calculate the mean and variance for $q(x_{t-1} | x_t, x_0)$ """
        epsilon_theta = self.model(x_t, t, c)
        mean = extract(self.coeff_1, t, x_t.shape) * x_t - extract(self.coeff_2, t, x_t.shape) * epsilon_theta
        # var is a constant
        var = extract(self.posterior_variance, t, x_t.shape)
        return mean, var

    @torch.no_grad()
    def sample_one_step(self, x_t, time_step, c):
        # """ Calculate $x_{t-1}$ according to $x_t$ """
        t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)
        mean, var = self.cal_mean_variance(x_t, t, c)
        z = torch.randn_like(x_t) if time_step > 0 else 0
        x_t_minus_one = mean + torch.sqrt(var) * z
        if torch.isnan(x_t_minus_one).int().sum() != 0:
            raise ValueError("nan in tensor!")
        return x_t_minus_one

    @torch.no_grad()
    def forward(self, x_t, c, only_return_x_0=True, interval=1):
        x = [x_t]
        for time_step in reversed(range(self.T)):
            x_t = self.sample_one_step(x_t, time_step, c)
            if not only_return_x_0 and ((self.T - time_step) % interval == 0 or time_step == 0):
                x.append(x_t)
        if only_return_x_0:
            return x_t  # [batch_size, channels, height, width]
        return torch.stack(x, dim=1)  # [batch_size, sample, channels, height, width]


class DDIMSampler(nn.Module):
    def __init__(self, model: nn.Module, beta: tuple[int, int], T: int):
        super().__init__()
        self.model = model
        self.T = T
        # generate T steps of beta
        beta_t = torch.linspace(*beta, T, dtype=torch.float32)
        # calculate the cumulative product of $\alpha$ , named $\bar{\alpha_t}$ in paper
        alpha_t = 1.0 - beta_t
        self.register_buffer("alpha_t_bar", torch.cumprod(alpha_t, dim=0))

    @torch.no_grad()
    def sample_one_step(self, x_t, time_step, c, prev_time_step, eta):
        t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)
        prev_t = torch.full((x_t.shape[0],), prev_time_step, device=x_t.device, dtype=torch.long)
        # get current and previous alpha_cumprod
        alpha_t = extract(self.alpha_t_bar, t, x_t.shape)
        alpha_t_prev = extract(self.alpha_t_bar, prev_t, x_t.shape)
        # predict noise using model
        epsilon_theta_t = self.model(x_t, t, c)
        # calculate x_{t-1}
        sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
        epsilon_t = torch.randn_like(x_t)
        x_t_minus_one = (torch.sqrt(alpha_t_prev / alpha_t) * x_t +
                         (torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) - torch.sqrt(
                             (alpha_t_prev * (1 - alpha_t)) / alpha_t)) * epsilon_theta_t +
                         sigma_t * epsilon_t)
        return x_t_minus_one

    @torch.no_grad()
    def forward(self, x_t, c, steps=60, method="linear", eta=0.05, only_return_x_0=True, interval=1):
        if method == "linear":
            a = self.T // steps
            time_steps = np.asarray(list(range(0, self.T, a)))
        elif method == "quadratic":
            time_steps = (np.linspace(0, np.sqrt(self.T * 0.8), steps) ** 2).astype(np.int)
        else:  # NotImplementedError
            raise NotImplementedError(f"sampling method {method} is not implemented!")
        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        time_steps = time_steps + 1
        # previous sequence
        time_steps_prev = np.concatenate([[0], time_steps[:-1]])
        x = [x_t]
        for i in reversed(range(0, steps)):
            x_t = self.sample_one_step(x_t, time_steps[i], c, time_steps_prev[i], eta)
            if not only_return_x_0 and ((steps - i) % interval == 0 or i == 0):
                x.append(x_t)
        if only_return_x_0:
            return x_t  # [batch_size, channels, height, width]
        return torch.stack(x, dim=1)  # [batch_size, sample, channels, height, width]
