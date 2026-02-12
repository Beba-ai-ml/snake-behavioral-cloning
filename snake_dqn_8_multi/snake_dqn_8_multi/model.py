import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """Noisy layer (Fortunato et al.) for exploration without epsilon."""

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("weight_eps", torch.empty(out_features, in_features))
        self.register_buffer("bias_eps", torch.empty(out_features))
        self.reset_parameters(sigma_init)
        self.reset_noise()

    def reset_parameters(self, sigma_init: float) -> None:
        mu_range = 1 / self.in_features**0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(sigma_init / (self.in_features**0.5))
        self.bias_sigma.data.fill_(sigma_init / (self.out_features**0.5))

    def reset_noise(self) -> None:
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_eps.copy_(eps_out.outer(eps_in))
        self.bias_eps.copy_(eps_out)

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_eps
            bias = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class DQN(nn.Module):
    """CNN + Dueling head with NoisyLinear layers."""

    def __init__(self, output_size: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),  # 8x8 -> 4x4
            nn.Flatten(),
        )
        # 128 * 4 * 4 = 2048
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
        )
        self.advantage = nn.Sequential(
            NoisyLinear(512, 256),
            nn.ReLU(),
            NoisyLinear(256, output_size),
        )
        self.value = nn.Sequential(
            NoisyLinear(512, 256),
            nn.ReLU(),
            NoisyLinear(256, 1),
        )

    def reset_noise(self) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.dim() == 3:
            x = x.unsqueeze(0)
        feats = self.conv(x)
        feats = self.fc(feats)
        adv = self.advantage(feats)
        val = self.value(feats)
        q = val + adv - adv.mean(dim=1, keepdim=True)
        return q
