import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.gamma = 0.99
        self.eps = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.995
        self.batch_size = 64
        self.memory = deque(maxlen=10000)

        self.model = QNet(state_dim, action_dim)
        self.target = QNet(state_dim, action_dim)
        self.target.load_state_dict(self.model.state_dict())

        self.opt = optim.Adam(self.model.parameters(), lr=1e-3)

    def act(self, state):
        if random.random() < self.eps:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            return int(torch.argmax(self.model(s), dim=1).item())

    def store(self, s, a, r, ns, d):
        self.memory.append((s, a, r, ns, d))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        s, a, r, ns, d = map(np.array, zip(*batch))

        s = torch.tensor(s, dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.int64).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
        ns = torch.tensor(ns, dtype=torch.float32)
        d = torch.tensor(d, dtype=torch.float32).unsqueeze(1)

        q = self.model(s).gather(1, a)

        with torch.no_grad():
            nq = self.target(ns).max(dim=1, keepdim=True)[0]
            y = r + self.gamma * (1 - d) * nq

        loss = nn.MSELoss()(q, y)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.eps = max(self.eps_min, self.eps * self.eps_decay)