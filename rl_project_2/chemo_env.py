import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class ChemoEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None, horizon=30, dt=1.0, patient=None):
        super().__init__()

        self.render_mode = render_mode
        self.horizon = horizon
        self.dt = dt

        self.patient = patient or {
            "r_t": 1.2,
            "k_t": 1.0,
            "kill_t": 0.9,
            "r_h": 0.7,
            "k_h": 1.0,
            "kill_h": 0.45,
            "drug_decay": 1.0,
        }

        self.doses = np.linspace(0.0, 1.0, 11)

        self.action_space = spaces.Discrete(len(self.doses))
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([2.0, 1.5, 2.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.T = None
        self.H = None
        self.C = None
        self.day = None
        self.last_dose = 0.0

        self.days = []
        self.tumor_history = []
        self.healthy_history = []
        self.drug_history = []
        self.dose_history = []

        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None
        self.ax4 = None

    def dynamics(self, t, y, u):
        T, H, C = y
        p = self.patient

        dT = p["r_t"] * T * (1 - T / p["k_t"]) - p["kill_t"] * C * T
        dH = p["r_h"] * H * (1 - H / p["k_h"]) - p["kill_h"] * C * H
        dC = -p["drug_decay"] * C + u

        return [dT, dH, dC]

    def _get_obs(self):
        return np.array(
            [self.T, self.H, self.C, self.day / self.horizon],
            dtype=np.float32
        )

    def _get_info(self):
        return {"dose": self.last_dose}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.day = 0
        self.T = 0.7
        self.H = 1.0
        self.C = 0.0
        self.last_dose = 0.0

        self.days = [0]
        self.tumor_history = [self.T]
        self.healthy_history = [self.H]
        self.drug_history = [self.C]
        self.dose_history = [self.last_dose]

        if self.render_mode == "human":
            self._init_plot()
            self.render()

        return self._get_obs(), self._get_info()

    def step(self, action):
        u = float(self.doses[action])
        self.last_dose = u

        sol = solve_ivp(
            lambda t, y: self.dynamics(t, y, u),
            [0, self.dt],
            [self.T, self.H, self.C],
            t_eval=[self.dt]
        )

        self.T, self.H, self.C = sol.y[:, -1]

        self.T = float(np.clip(self.T, 0.0, 2.0))
        self.H = float(np.clip(self.H, 0.0, 1.5))
        self.C = float(np.clip(self.C, 0.0, 2.0))

        self.day += 1

        self.days.append(self.day)
        self.tumor_history.append(self.T)
        self.healthy_history.append(self.H)
        self.drug_history.append(self.C)
        self.dose_history.append(self.last_dose)

        reward = -2.0 * self.T + 1.0 * self.H - 0.3 * (u ** 2)
        if self.H < 0.3:
            reward -= 2.0

        terminated = self.T < 0.05 or self.H < 0.05
        truncated = self.day >= self.horizon

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _init_plot(self):
        plt.ion()
        self.fig, (self.ax1, self.ax2, self.ax3, self.ax4) = plt.subplots(
            4, 1, figsize=(10, 12), sharex=True
        )
        self.fig.suptitle("Chemotherapy Environment Live Rendering")

    def render(self):
        print(
            f"Day: {self.day:02d} | "
            f"Tumor: {self.T:.3f} | "
            f"Healthy: {self.H:.3f} | "
            f"Drug: {self.C:.3f} | "
            f"Last dose: {self.last_dose:.2f}"
        )

        if self.fig is None:
            return

        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()

        self.ax1.plot(self.days, self.tumor_history, color="red")
        self.ax1.set_ylabel("Tumor")
        self.ax1.set_title("Tumor Burden")
        self.ax1.grid(True)

        self.ax2.plot(self.days, self.healthy_history, color="green")
        self.ax2.set_ylabel("Healthy")
        self.ax2.set_title("Healthy Tissue")
        self.ax2.grid(True)

        self.ax3.plot(self.days, self.drug_history, color="purple")
        self.ax3.set_ylabel("Drug")
        self.ax3.set_title("Drug Concentration")
        self.ax3.grid(True)

        self.ax4.step(self.days, self.dose_history, where="post", color="blue")
        self.ax4.set_ylabel("Dose")
        self.ax4.set_xlabel("Day")
        self.ax4.set_title("Dose Schedule")
        self.ax4.grid(True)

        plt.tight_layout()
        plt.pause(0.3)

    def close(self):
        if self.fig is not None:
            plt.ioff()
            plt.close(self.fig)