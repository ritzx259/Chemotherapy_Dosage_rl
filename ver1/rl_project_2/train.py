import math
import numpy as np
import matplotlib.pyplot as plt
import torch

from chemo_env import ChemoEnv
from dqn_agent import DQNAgent
from chemo_gridworld import demo_grid_world
#from chemo_gridworld_live import live_demo_grid_world


def evaluate_policy(env, agent):
    state, _ = env.reset()
    done = False

    days = [env.day]
    tumors = [env.T]
    healthy = [env.H]
    drug_conc = [env.C]
    doses = [env.last_dose]

    while not done:
        with torch.no_grad():
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
        r = i // cols
        c = i % cols
        grid[r, c] = dose

        label = f"D{int(days[i])}\n{dose:.2f}"

        if i == 0:
            label = "START\n" + label
        if i == n - 1:
            label = "END\n" + label

        if healthy is not None and healthy[i] < 0.3:
            label += "\nTOX"

        if tumors is not None and tumors[i] < 0.05:
            label += "\nREM"

        labels[r][c] = label

    fig, ax = plt.subplots(figsize=(cols * 1.8, rows * 1.8))
    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color="white")

    im = ax.imshow(grid, cmap=cmap, vmin=0.0, vmax=1.0)

    for r in range(rows):
        for c in range(cols):
            if not np.isnan(grid[r, c]):
                ax.text(
                    c, r, labels[r][c],
                    ha="center", va="center",
                    fontsize=9, color="black", weight="bold"
                )

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


def train(episodes=300):
    env = ChemoEnv(render_mode=None)
    agent = DQNAgent(state_dim=4, action_dim=env.action_space.n)
    returns = []

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total = 0.0

        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            total += reward

        returns.append(total)

        if ep % 10 == 0:
            agent.target.load_state_dict(agent.model.state_dict())
            print(f"episode={ep}, return={total:.3f}, eps={agent.eps:.3f}")

    return agent, returns


def demo_trained_agent(agent):
    live_env = ChemoEnv(render_mode="human")
    state, _ = live_env.reset()
    done = False

    agent.eps = 0.0

    while not done:
        action = agent.act(state)
        state, reward, terminated, truncated, info = live_env.step(action)
        done = terminated or truncated

    live_env.close()


if __name__ == "__main__":
    agent, returns = train()

    plot_training_returns(returns)

    demo_trained_agent(agent)

    eval_env = ChemoEnv(render_mode=None)
    days, tumors, healthy, drug_conc, doses = evaluate_policy(eval_env, agent)
    plot_episode_trajectories(days, tumors, healthy, drug_conc, doses)
    plot_dose_grid(days, doses, tumors=tumors, healthy=healthy, cols=6)
    demo_grid_world(days, doses, tumors=tumors, healthy=healthy)
    #live_demo_grid_world(days, doses, tumors=tumors, healthy=healthy)