import torch
import torch.nn as nn
import torch.nn.functional as F

POLICY_DIM = 4672

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity
        return F.relu(out)


class AlphaZeroNet(nn.Module):
    def __init__(self, in_channels=18, channels=64, num_blocks=6, policy_dim=4672):
        super().__init__()

        self.conv_in = nn.Conv2d(in_channels, channels, 3, padding=1)
        self.bn_in = nn.BatchNorm2d(channels)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(num_blocks)
        ])

        self.policy_conv = nn.Conv2d(channels, 16, 1)
        self.policy_fc = nn.Linear(16 * 8 * 8, policy_dim)

        self.value_conv = nn.Conv2d(channels, 4, 1)
        self.value_fc1 = nn.Linear(4 * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            x = block(x)

        p = F.relu(self.policy_conv(x))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        v = F.relu(self.value_conv(x))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        return p, v
