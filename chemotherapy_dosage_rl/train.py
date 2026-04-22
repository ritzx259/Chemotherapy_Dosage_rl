from chemo_env import ChemoEnv
from dataset_loader import DATASET_PATHS, DatasetPatientBuilder
from dqn_agent import DQNAgent
from utils import evaluate_policy, plot_training_returns, plot_episode_trajectories, plot_dose_grid


def train(episodes=300, dataset_key="crc_folfox"):
    csv_path = DATASET_PATHS[dataset_key]
    builder = DatasetPatientBuilder(csv_path)
    env0 = ChemoEnv(render_mode=None)
    agent = DQNAgent(state_dim=4, action_dim=env0.action_space.n)
    returns = []

    for ep in range(episodes):
        patient, row = builder.sample_patient()
        env = ChemoEnv(render_mode=None, patient=patient)
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
            print(f"episode={ep}, return={total:.3f}, eps={agent.eps:.3f}, dataset={dataset_key}, path={csv_path}")

    return agent, returns, builder


def demo_trained_agent(agent, builder):
    patient, row = builder.sample_patient()
    live_env = ChemoEnv(render_mode="human", patient=patient)
    state, _ = live_env.reset()
    done = False
    agent.eps = 0.0
    while not done:
        action = agent.act(state)
        state, reward, terminated, truncated, info = live_env.step(action)
        done = terminated or truncated
    live_env.close()


if __name__ == "__main__":
    selected_dataset = "crc_folfox"
    agent, returns, builder = train(episodes=300, dataset_key=selected_dataset)
    plot_training_returns(returns)
    demo_trained_agent(agent, builder)

    patient, row = builder.sample_patient()
    eval_env = ChemoEnv(render_mode=None, patient=patient)
    days, tumors, healthy, drug_conc, doses = evaluate_policy(eval_env, agent)
    plot_episode_trajectories(days, tumors, healthy, drug_conc, doses)
    plot_dose_grid(days, doses, tumors=tumors, healthy=healthy, cols=6)