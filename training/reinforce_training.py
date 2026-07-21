import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from environment.custom_env import SafeBirthEnv

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.network(x)

def train_reinforce(hyperparams, run_id=1):
    log_dir = f"logs/reinforce/run_{run_id}"
    model_dir = f"models/pg/reinforce/run_{run_id}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    env = SafeBirthEnv()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    lr = hyperparams.get("learning_rate", 1e-3)
    gamma = hyperparams.get("gamma", 0.99)
    hidden_size = hyperparams.get("hidden_size", 64)
    n_episodes = hyperparams.get("n_episodes", 500)

    policy = PolicyNetwork(obs_dim, action_dim, hidden_size)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    episode_rewards = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        log_probs = []
        rewards = []
        done = False

        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            probs = policy(obs_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            log_probs.append(log_prob)
            rewards.append(reward)

        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns)
        if returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient loss
        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss -= log_prob * G

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_rewards.append(sum(rewards))

        if (episode + 1) % 100 == 0:
            mean_r = np.mean(episode_rewards[-100:])
            print(f"  REINFORCE Run {run_id} | Ep {episode+1}/{n_episodes} | Mean Reward: {mean_r:.2f}")

    torch.save(policy.state_dict(), f"{model_dir}/reinforce_safebirth.pt")
    env.close()

    mean_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
    print(f"REINFORCE Run {run_id} | Final Mean Reward: {mean_reward:.2f}")
    return policy, mean_reward, episode_rewards

def run_all_reinforce_experiments():
    experiments = [
        {"learning_rate": 1e-3, "gamma": 0.99, "hidden_size": 64,  "n_episodes": 500},
        {"learning_rate": 5e-4, "gamma": 0.99, "hidden_size": 64,  "n_episodes": 500},
        {"learning_rate": 1e-4, "gamma": 0.99, "hidden_size": 64,  "n_episodes": 500},
        {"learning_rate": 1e-3, "gamma": 0.95, "hidden_size": 64,  "n_episodes": 500},
        {"learning_rate": 1e-3, "gamma": 0.90, "hidden_size": 64,  "n_episodes": 500},
        {"learning_rate": 1e-3, "gamma": 0.99, "hidden_size": 128, "n_episodes": 500},
        {"learning_rate": 1e-3, "gamma": 0.99, "hidden_size": 32,  "n_episodes": 500},
        {"learning_rate": 1e-3, "gamma": 0.99, "hidden_size": 64,  "n_episodes": 800},
        {"learning_rate": 5e-3, "gamma": 0.99, "hidden_size": 64,  "n_episodes": 500},
        {"learning_rate": 1e-3, "gamma": 0.99, "hidden_size": 256, "n_episodes": 500},
    ]

    results = []
    all_rewards = []
    for i, params in enumerate(experiments):
        print(f"\nRunning REINFORCE experiment {i+1}/10...")
        policy, mean_reward, rewards = train_reinforce(params, run_id=i+1)
        results.append({"run": i+1, "params": params, "mean_reward": mean_reward})
        all_rewards.append(rewards)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    for idx, rewards in enumerate(all_rewards):
        axes[idx].plot(rewards, alpha=0.7, color='purple')
        axes[idx].set_title(f"REINFORCE Run {idx+1}", fontsize=10)
        axes[idx].set_xlabel("Episode")
        axes[idx].set_ylabel("Reward")
        axes[idx].grid(True, alpha=0.3)
    plt.suptitle("REINFORCE Hyperparameter Experiments — SAFEBIRTH", fontsize=14)
    plt.tight_layout()
    plt.savefig("logs/reinforce_all_experiments.png", dpi=150)
    plt.close()
    print("\nREINFORCE plot saved to logs/reinforce_all_experiments.png")
    return results

if __name__ == "__main__":
    run_all_reinforce_experiments()