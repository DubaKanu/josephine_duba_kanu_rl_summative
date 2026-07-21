# SAFEBIRTH — Maternal Health Triage RL Agent
**Josephine Duba Kanu | African Leadership University | ML Techniques 2**

## Project Overview
SAFEBIRTH is a reinforcement learning environment simulating a Community Health Worker triaging pregnant women showing danger signs in Sierra Leone. The RL agent learns to make optimal referral decisions — from monitoring at home to calling an ambulance — based on patient vitals and danger signs.

## Mission Connection
This environment is directly tied to the SAFEBIRTH project — an AI-powered maternal health early warning system for Community Health Workers in Sierra Leone, where maternal mortality remains critically high.

## Environment Description
| Component | Description |
|---|---|
| Observation Space | 11 continuous values: BP, HR, temperature, respiratory rate, bleeding, convulsions, loss of consciousness, gestational age, distance to facility |
| Action Space | 5 discrete actions: Monitor at Home, Refer to Health Post, Refer to Clinic, Emergency Hospital Referral, Call Ambulance |
| Reward | +20 for correct triage, -3 to -50 for over/under triage depending on severity |
| Terminal Condition | After 10 triage decisions per episode |
| Start State | Random patient profile from 5 severity levels |

## How to Run
```bash
# Install dependencies
uv sync

# Test the environment visually
uv run main.py test

# Train all 4 algorithms (40 experiments total)
uv run main.py train

# Play with best trained model
uv run main.py play dqn 5
uv run main.py play ppo 7
uv run main.py play a2c 9
```

## Results Summary
| Algorithm | Best Run | Best Mean Reward |
|---|---|---|
| DQN | Run 5 | 185.62 |
| PPO | Run 7 | 120.78 |
| A2C | Run 9 | 98.09 |
| REINFORCE | Run 6 | -19.07 |

**Best performing agent: DQN Run 5 — achieved perfect score of 200/200 during evaluation**

## DQN Hyperparameter Experiments
| Run | Learning Rate | Gamma | Batch Size | Buffer Size | Exploration Fraction | Mean Reward |
|---|---|---|---|---|---|---|
| 1 | 0.001 | 0.99 | 32 | 10000 | 0.10 | 174.88 |
| 2 | 0.0005 | 0.99 | 32 | 10000 | 0.10 | 143.24 |
| 3 | 0.0001 | 0.99 | 32 | 10000 | 0.10 | - |
| 4 | 0.001 | 0.95 | 32 | 10000 | 0.10 | - |
| 5 | 0.001 | 0.90 | 32 | 10000 | 0.10 | **185.62** |
| 6 | 0.001 | 0.99 | 64 | 10000 | 0.10 | - |
| 7 | 0.001 | 0.99 | 128 | 10000 | 0.10 | - |
| 8 | 0.001 | 0.99 | 32 | 50000 | 0.10 | - |
| 9 | 0.001 | 0.99 | 32 | 10000 | 0.20 | - |
| 10 | 0.001 | 0.99 | 32 | 10000 | 0.05 | - |

## PPO Hyperparameter Experiments
| Run | Learning Rate | Gamma | N Steps | Batch Size | Entropy Coef | Clip Range | Mean Reward |
|---|---|---|---|---|---|---|---|
| 1 | 0.0003 | 0.99 | 2048 | 64 | 0.0 | 0.2 | - |
| 2 | 0.0001 | 0.99 | 2048 | 64 | 0.0 | 0.2 | - |
| 3 | 0.001 | 0.99 | 2048 | 64 | 0.0 | 0.2 | - |
| 4 | 0.0003 | 0.95 | 2048 | 64 | 0.0 | 0.2 | - |
| 5 | 0.0003 | 0.99 | 1024 | 64 | 0.0 | 0.2 | - |
| 6 | 0.0003 | 0.99 | 2048 | 128 | 0.0 | 0.2 | - |
| 7 | 0.0003 | 0.99 | 2048 | 64 | 0.01 | 0.2 | **120.78** |
| 8 | 0.0003 | 0.99 | 2048 | 64 | 0.0 | 0.3 | - |
| 9 | 0.0003 | 0.99 | 2048 | 64 | 0.0 | 0.1 | - |
| 10 | 0.0003 | 0.99 | 4096 | 64 | 0.01 | 0.2 | - |

## A2C Hyperparameter Experiments
| Run | Learning Rate | Gamma | N Steps | Entropy Coef | VF Coef | Mean Reward |
|---|---|---|---|---|---|---|
| 1 | 0.0007 | 0.99 | 5 | 0.0 | 0.5 | - |
| 2 | 0.001 | 0.99 | 5 | 0.0 | 0.5 | - |
| 3 | 0.0001 | 0.99 | 5 | 0.0 | 0.5 | - |
| 4 | 0.0007 | 0.95 | 5 | 0.0 | 0.5 | - |
| 5 | 0.0007 | 0.99 | 10 | 0.0 | 0.5 | - |
| 6 | 0.0007 | 0.99 | 20 | 0.0 | 0.5 | - |
| 7 | 0.0007 | 0.99 | 5 | 0.01 | 0.5 | - |
| 8 | 0.0007 | 0.99 | 5 | 0.0 | 0.25 | - |
| 9 | 0.0007 | 0.99 | 5 | 0.0 | 0.75 | **98.09** |
| 10 | 0.0005 | 0.99 | 10 | 0.01 | 0.5 | - |

## REINFORCE Hyperparameter Experiments
| Run | Learning Rate | Gamma | Hidden Size | Episodes | Mean Reward |
|---|---|---|---|---|---|
| 1 | 0.001 | 0.99 | 64 | 500 | - |
| 2 | 0.0005 | 0.99 | 64 | 500 | - |
| 3 | 0.0001 | 0.99 | 64 | 500 | - |
| 4 | 0.001 | 0.95 | 64 | 500 | - |
| 5 | 0.001 | 0.90 | 64 | 500 | - |
| 6 | 0.001 | 0.99 | 128 | 500 | **-19.07** |
| 7 | 0.001 | 0.99 | 32 | 500 | - |
| 8 | 0.001 | 0.99 | 64 | 800 | -23.22 |
| 9 | 0.005 | 0.99 | 64 | 500 | -223.05 |
| 10 | 0.001 | 0.99 | 256 | 500 | -55.20 |

## Video Demonstration
[Add link after recording]

## Project Structure
```
josephine_duba_kanu_rl_summative/
├── pyproject.toml
├── uv.lock
├── README.md
├── main.py
├── environment/
│   ├── __init__.py
│   └── custom_env.py
├── training/
│   ├── __init__.py
│   ├── dqn_training.py
│   ├── pg_training.py
│   └── reinforce_training.py
├── models/
│   ├── dqn/
│   └── pg/
└── logs/
```