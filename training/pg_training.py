import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from environment.custom_env import SafeBirthEnv

def train_pg(algorithm, hyperparams, run_id, algo_name):
    log_dir = f"logs/{algo_name}/run_{run_id}"
    model_dir = f"models/pg/{algo_name}/run_{run_id}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    env = Monitor(SafeBirthEnv(), log_dir)

    if algo_name == "ppo":
        model = PPO(
            "MlpPolicy", env,
            learning_rate=hyperparams.get("learning_rate", 3e-4),
            gamma=hyperparams.get("gamma", 0.99),
            n_steps=hyperparams.get("n_steps", 2048),
            batch_size=hyperparams.get("batch_size", 64),
            n_epochs=hyperparams.get("n_epochs", 10),
            ent_coef=hyperparams.get("ent_coef", 0.0),
            clip_range=hyperparams.get("clip_range", 0.2),
            verbose=0,
            tensorboard_log=f"logs/{algo_name}/tb_{run_id}"
        )
    else:
        model = A2C(
            "MlpPolicy", env,
            learning_rate=hyperparams.get("learning_rate", 7e-4),
            gamma=hyperparams.get("gamma", 0.99),
            n_steps=hyperparams.get("n_steps", 5),
            ent_coef=hyperparams.get("ent_coef", 0.0),
            vf_coef=hyperparams.get("vf_coef", 0.5),
            verbose=0,
            tensorboard_log=f"logs/{algo_name}/tb_{run_id}"
        )

    model.learn(total_timesteps=hyperparams.get("total_timesteps", 50000))
    model.save(f"{model_dir}/{algo_name}_safebirth")
    env.close()

    x, y = ts2xy(load_results(log_dir), "timesteps")
    mean_reward = np.mean(y[-100:]) if len(y) >= 100 else np.mean(y)
    print(f"{algo_name.upper()} Run {run_id} | Mean Reward: {mean_reward:.2f}")
    return model, mean_reward, x, y


def run_all_ppo_experiments():
    experiments = [
        {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 2048, "batch_size": 64,  "ent_coef": 0.0,  "clip_range": 0.2, "total_timesteps": 50000},
        {"learning_rate": 1e-4, "gamma": 0.99, "n_steps": 2048, "batch_size": 64,  "ent_coef": 0.0,  "clip_range": 0.2, "total_timesteps": 50000},
        {"learning_rate": 1e-3, "gamma": 0.99, "n_steps": 2048, "batch_size": 64,  "ent_coef": 0.0,  "clip_range": 0.2, "total_timesteps": 50000},
        {"learning_rate": 3e-4, "gamma": 0.95, "n_steps": 2048, "batch_size": 64,  "ent_coef": 0.0,  "clip_range": 0.2, "total_timesteps": 50000},
        {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 1024, "batch_size": 64,  "ent_coef": 0.0,  "clip_range": 0.2, "total_timesteps": 50000},
        {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 2048, "batch_size": 128, "ent_coef": 0.0,  "clip_range": 0.2, "total_timesteps": 50000},
        {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 2048, "batch_size": 64,  "ent_coef": 0.01, "clip_range": 0.2, "total_timesteps": 50000},
        {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 2048, "batch_size": 64,  "ent_coef": 0.0,  "clip_range": 0.3, "total_timesteps": 50000},
        {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 2048, "batch_size": 64,  "ent_coef": 0.0,  "clip_range": 0.1, "total_timesteps": 50000},
        {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 4096, "batch_size": 64,  "ent_coef": 0.01, "clip_range": 0.2, "total_timesteps": 50000},
    ]

    results = []
    all_rewards = []
    for i, params in enumerate(experiments):
        print(f"\nRunning PPO experiment {i+1}/10...")
        model, mean_reward, x, y = train_pg(None, params, i+1, "ppo")
        results.append({"run": i+1, "params": params, "mean_reward": mean_reward})
        all_rewards.append((x, y, f"Run {i+1}"))

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    for idx, (x, y, label) in enumerate(all_rewards):
        axes[idx].plot(x, y, alpha=0.7, color='green')
        axes[idx].set_title(f"PPO {label}", fontsize=10)
        axes[idx].set_xlabel("Timesteps")
        axes[idx].set_ylabel("Reward")
        axes[idx].grid(True, alpha=0.3)
    plt.suptitle("PPO Hyperparameter Experiments — SAFEBIRTH", fontsize=14)
    plt.tight_layout()
    plt.savefig("logs/ppo_all_experiments.png", dpi=150)
    plt.close()
    print("\nPPO plot saved to logs/ppo_all_experiments.png")
    return results


def run_all_a2c_experiments():
    experiments = [
        {"learning_rate": 7e-4, "gamma": 0.99, "n_steps": 5,  "ent_coef": 0.0,  "vf_coef": 0.5, "total_timesteps": 50000},
        {"learning_rate": 1e-3, "gamma": 0.99, "n_steps": 5,  "ent_coef": 0.0,  "vf_coef": 0.5, "total_timesteps": 50000},
        {"learning_rate": 1e-4, "gamma": 0.99, "n_steps": 5,  "ent_coef": 0.0,  "vf_coef": 0.5, "total_timesteps": 50000},
        {"learning_rate": 7e-4, "gamma": 0.95, "n_steps": 5,  "ent_coef": 0.0,  "vf_coef": 0.5, "total_timesteps": 50000},
        {"learning_rate": 7e-4, "gamma": 0.99, "n_steps": 10, "ent_coef": 0.0,  "vf_coef": 0.5, "total_timesteps": 50000},
        {"learning_rate": 7e-4, "gamma": 0.99, "n_steps": 20, "ent_coef": 0.0,  "vf_coef": 0.5, "total_timesteps": 50000},
        {"learning_rate": 7e-4, "gamma": 0.99, "n_steps": 5,  "ent_coef": 0.01, "vf_coef": 0.5, "total_timesteps": 50000},
        {"learning_rate": 7e-4, "gamma": 0.99, "n_steps": 5,  "ent_coef": 0.0,  "vf_coef": 0.25,"total_timesteps": 50000},
        {"learning_rate": 7e-4, "gamma": 0.99, "n_steps": 5,  "ent_coef": 0.0,  "vf_coef": 0.75,"total_timesteps": 50000},
        {"learning_rate": 5e-4, "gamma": 0.99, "n_steps": 10, "ent_coef": 0.01, "vf_coef": 0.5, "total_timesteps": 50000},
    ]

    results = []
    all_rewards = []
    for i, params in enumerate(experiments):
        print(f"\nRunning A2C experiment {i+1}/10...")
        model, mean_reward, x, y = train_pg(None, params, i+1, "a2c")
        results.append({"run": i+1, "params": params, "mean_reward": mean_reward})
        all_rewards.append((x, y, f"Run {i+1}"))

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    for idx, (x, y, label) in enumerate(all_rewards):
        axes[idx].plot(x, y, alpha=0.7, color='orange')
        axes[idx].set_title(f"A2C {label}", fontsize=10)
        axes[idx].set_xlabel("Timesteps")
        axes[idx].set_ylabel("Reward")
        axes[idx].grid(True, alpha=0.3)
    plt.suptitle("A2C Hyperparameter Experiments — SAFEBIRTH", fontsize=14)
    plt.tight_layout()
    plt.savefig("logs/a2c_all_experiments.png", dpi=150)
    plt.close()
    print("\nA2C plot saved to logs/a2c_all_experiments.png")
    return results


if __name__ == "__main__":
    run_all_ppo_experiments()
    run_all_a2c_experiments()