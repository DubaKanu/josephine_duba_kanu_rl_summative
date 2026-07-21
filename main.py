import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.monitor import Monitor
from environment.custom_env import SafeBirthEnv

def play(algo="dqn", run_id=1, episodes=5):
    """Load best model and play with visualization."""
    env = SafeBirthEnv(render_mode="human")

    if algo == "dqn":
        path = f"models/dqn/run_{run_id}/dqn_safebirth"
        model = DQN.load(path)
    elif algo == "ppo":
        path = f"models/pg/ppo/run_{run_id}/ppo_safebirth"
        model = PPO.load(path)
    elif algo == "a2c":
        path = f"models/pg/a2c/run_{run_id}/a2c_safebirth"
        model = A2C.load(path)
    else:
        print(f"Unknown algorithm: {algo}")
        return

    print(f"\nPlaying with {algo.upper()} model — {episodes} episodes")
    print("=" * 50)

    total_rewards = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        total_rewards.append(ep_reward)
        print(f"Episode {ep+1}: Reward = {ep_reward:.1f}")

    print(f"\nMean Reward: {np.mean(total_rewards):.2f}")
    env.close()

def train_all():
    """Run all training experiments."""
    print("=" * 60)
    print("SAFEBIRTH RL TRAINING — All Algorithms")
    print("=" * 60)

    print("\n[1/4] Training DQN...")
    from training.dqn_training import run_all_dqn_experiments
    dqn_results = run_all_dqn_experiments()

    print("\n[2/4] Training PPO...")
    from training.pg_training import run_all_ppo_experiments
    ppo_results = run_all_ppo_experiments()

    print("\n[3/4] Training A2C...")
    from training.pg_training import run_all_a2c_experiments
    a2c_results = run_all_a2c_experiments()

    print("\n[4/4] Training REINFORCE...")
    from training.reinforce_training import run_all_reinforce_experiments
    reinforce_results = run_all_reinforce_experiments()

    print("\n" + "=" * 60)
    print("ALL TRAINING COMPLETE!")
    print("=" * 60)

    best_dqn = max(dqn_results, key=lambda x: x["mean_reward"])
    best_ppo = max(ppo_results, key=lambda x: x["mean_reward"])
    best_a2c = max(a2c_results, key=lambda x: x["mean_reward"])
    best_reinforce = max(reinforce_results, key=lambda x: x["mean_reward"])

    print(f"\nBest DQN: Run {best_dqn['run']} | Reward: {best_dqn['mean_reward']:.2f}")
    print(f"Best PPO: Run {best_ppo['run']} | Reward: {best_ppo['mean_reward']:.2f}")
    print(f"Best A2C: Run {best_a2c['run']} | Reward: {best_a2c['mean_reward']:.2f}")
    print(f"Best REINFORCE: Run {best_reinforce['run']} | Reward: {best_reinforce['mean_reward']:.2f}")

    return {
        "dqn": best_dqn,
        "ppo": best_ppo,
        "a2c": best_a2c,
        "reinforce": best_reinforce
    }

def test_env():
    """Quick test of the environment."""
    print("Testing SAFEBIRTH environment...")
    env = SafeBirthEnv(render_mode="human")
    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Initial obs: {obs}")

    import pygame
    import time
    running = True
    step = 0
    while running and step < 20:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step+1} | Action: {info['action_taken']} | Reward: {reward:.1f} | Severity: {info['true_severity']}")
        time.sleep(0.5)
        step += 1
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
    print("Environment test complete!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  uv run main.py test      — test the environment visually")
        print("  uv run main.py train     — train all 4 algorithms")
        print("  uv run main.py play dqn  — play with best DQN model")
        print("  uv run main.py play ppo  — play with best PPO model")
        print("  uv run main.py play a2c  — play with best A2C model")
    elif sys.argv[1] == "test":
        test_env()
    elif sys.argv[1] == "train":
        train_all()
    elif sys.argv[1] == "play":
        algo = sys.argv[2] if len(sys.argv) > 2 else "dqn"
        run_id = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        play(algo=algo, run_id=run_id)