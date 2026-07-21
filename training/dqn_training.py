import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from environment.custom_env import SafeBirthEnv

def train_dqn(hyperparams: dict, run_id: int = 1):
    log_dir = f"logs/dqn/run_{run_id}"
    model_dir = f"models/dqn/run_{run_id}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    env = Monitor(SafeBirthEnv(), log_dir)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=hyperparams.get("learning_rate", 1e-3),
        gamma=hyperparams.get("gamma", 0.99),
        batch_size=hyperparams.get("batch_size", 32),
        buffer_size=hyperparams.get("buffer_size", 10000),
        exploration_initial_eps=hyperparams.get("exploration_initial_eps", 1.0),
        exploration_final_eps=hyperparams.get("exploration_final_eps", 0.05),
        exploration_fraction=hyperparams.get("exploration_fraction", 0.1),
        target_update_interval=hyperparams.get("target_update_interval", 500),
        learning_starts=hyperparams.get("learning_starts", 1000),
        verbose=0,
        tensorboard_log=f"logs/dqn/tb_{run_id}"
    )

    model.learn(total_timesteps=hyperparams.get("total_timesteps", 50000))
    model.save(f"{model_dir}/dqn_safebirth")
    env.close()

    x, y = ts2xy(load_results(log_dir), "timesteps")
    mean_reward = np.mean(y[-100:]) if len(y) >= 100 else np.mean(y)
    print(f"DQN Run {run_id} | Mean Reward (last 100 eps): {mean_reward:.2f}")
    return model, mean_reward, x, y

def run_all_dqn_experiments():
    experiments = [
        {"learning_rate": 1e-3,  "gamma": 0.99, "batch_size": 32,  "buffer_size": 10000,  "exploration_fraction": 0.1,  "total_timesteps": 50000},
        {"learning_rate": 5e-4,  "gamma": 0.99, "batch_size": 32,  "buffer_size": 10000,  "exploration_fraction": 0.1,  "total_timesteps": 50000},
        {"learning_rate": 1e-4,  "gamma": 0.99, "batch_size": 32,  "buffer_size": 10000,  "exploration_fraction": 0.1,  "total_timesteps": 50000},
        {"learning_rate": 1e-3,  "gamma": 0.95, "batch_size": 32,  "buffer_size": 10000,  "exploration_fraction": 0.1,  "total_timesteps": 50000},
        {"learning_rate": 1e-3,  "gamma": 0.90, "batch_size": 32,  "buffer_size": 10000,  "exploration_fraction": 0.1,  "total_timesteps": 50000},
        {"learning_rate": 1e-3,  "gamma": 0.99, "batch_size": 64,  "buffer_size": 10000,  "exploration_fraction": 0.1,  "total_timesteps": 50000},
        {"learning_rate": 1e-3,  "gamma": 0.99, "batch_size": 128, "buffer_size": 10000,  "exploration_fraction": 0.1,  "total_timesteps": 50000},
        {"learning_rate": 1e-3,  "gamma": 0.99, "batch_size": 32,  "buffer_size": 50000,  "exploration_fraction": 0.1,  "total_timesteps": 50000},
        {"learning_rate": 1e-3,  "gamma": 0.99, "batch_size": 32,  "buffer_size": 10000,  "exploration_fraction": 0.2,  "total_timesteps": 50000},
        {"learning_rate": 1e-3,  "gamma": 0.99, "batch_size": 32,  "buffer_size": 10000,  "exploration_fraction": 0.05, "total_timesteps": 50000},
    ]

    results = []
    all_rewards = []
    for i, params in enumerate(experiments):
        print(f"\nRunning DQN experiment {i+1}/10...")
        model, mean_reward, x, y = train_dqn(params, run_id=i+1)
        results.append({"run": i+1, "params": params, "mean_reward": mean_reward})
        all_rewards.append((x, y, f"Run {i+1}"))

    # Plot all reward curves
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    for idx, (x, y, label) in enumerate(all_rewards):
        axes[idx].plot(x, y, alpha=0.7, color='steelblue')
        axes[idx].set_title(f"DQN {label}", fontsize=10)
        axes[idx].set_xlabel("Timesteps")
        axes[idx].set_ylabel("Reward")
        axes[idx].grid(True, alpha=0.3)
    plt.suptitle("DQN Hyperparameter Experiments — SAFEBIRTH", fontsize=14)
    plt.tight_layout()
    plt.savefig("logs/dqn_all_experiments.png", dpi=150)
    plt.close()
    print("\nDQN plot saved to logs/dqn_all_experiments.png")
    return results

if __name__ == "__main__":
    run_all_dqn_experiments()