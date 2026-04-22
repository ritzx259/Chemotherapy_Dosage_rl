import math

import dose as dose
import numpy as np
import matplotlib.pyplot as plt
import rows as rows
from cattrs import cols

def evaluate_policy(env, agent):
    state, _ = env.reset()
    done = False
    days = [env.day]
    tumors = [env.T]
    healthy = [env.H]
    drug_conc = [env.C]
    doses = [env.last_dose]
    while not done:
        action = agent.act(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        days.append(env.day)
        tumors.append(env.T)
        healthy.append(env.H)
        drug_conc.append(env.C)
        doses.append(info["dose"])
        state = next_state
    return days, tumors, healthy, drug_conc, doses


def plot_training_returns(returns):
    plt.figure(figsize=(10, 5))
    plt.plot(returns, label="Episode Return", color="black")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Training Return Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_returns.png")
    plt.show()

def plot_episode_trajectories(days, tumors, healthy, drug_conc, doses):
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    axes[0].plot(days, tumors, color="red")
    axes[0].set_ylabel("Tumor")
    axes[0].set_title("Tumor Burden Over Time")
    axes[0].grid(True)

    axes[1].plot(days, healthy, color="green")
    axes[1].set_ylabel("Healthy")
    axes[1].set_title("Healthy Tissue Over Time")
    axes[1].grid(True)

    axes[2].plot(days, drug_conc, color="purple")
    axes[2].set_ylabel("Drug Conc.")
    axes[2].set_title("Drug Concentration Over Time")
    axes[2].grid(True)

    axes[3].step(days, doses, where="post", color="blue")
    axes[3].set_ylabel("Dose")
    axes[3].set_xlabel("Day")
    axes[3].set_title("Dose Schedule Over Time")
    axes[3].grid(True)

    plt.tight_layout()
    plt.savefig("final_episode_trajectories.png")
    plt.show()


def plot_dose_grid(days, doses, tumors=None, healthy=None, cols=6):
    n = len(doses)
    rows = math.ceil(n / cols)
    grid = np.full((rows, cols), np.nan)
    labels = [["" for _ in range(cols)] for _ in range(rows)]

    for i, dose in enumerate(doses):
        r, c = i // cols, i % cols
        grid[r, c] = dose
        label = f"D{int(days[i])} {dose:.2f}"
        if i == 0:
            label = "START" + label
        if i == n - 1:
            label = "END" + label
        if healthy is not None and healthy[i] < 0.3:
            label += "TOX"
        if tumors is not None and tumors[i] < 0.05:
            label += "REM"
        labels[r][c] = label

    fig, ax = plt.subplots(figsize=(cols * 1.8, rows * 1.8))
    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color="white")
    im = ax.imshow(grid, cmap=cmap, vmin=0.0, vmax=1.0)

    for r in range(rows):
        for c in range(cols):
            if not np.isnan(grid[r, c]):
                ax.text(c, r, labels[r][c], ha="center", va="center", fontsize=9, color="black", weight="bold")

    ax.set_title("Agent Chemotherapy Dose Grid")
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels([f"C{c+1}" for c in range(cols)])
    ax.set_yticklabels([f"R{r+1}" for r in range(rows)])
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Dose intensity")
    plt.tight_layout()
    plt.savefig("dose_grid.png")
    plt.show()