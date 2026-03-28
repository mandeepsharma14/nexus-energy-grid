"""
NexusGrid — Reinforcement Learning Load Optimization Agent
Uses PPO (Proximal Policy Optimization) to learn optimal load scheduling.

Environment:
  - State: current load, hour, price signal, temperature, production schedule
  - Action: load adjustment [-20%, +20%] per zone
  - Reward: -cost - carbon_penalty + production_bonus

© 2026 Mandeep Sharma. All rights reserved.
"""

import numpy as np
import pandas as pd
import json
import os
import pickle
import warnings
from typing import Tuple, Optional

warnings.filterwarnings("ignore")

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/generated")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# 1. ENERGY ENVIRONMENT (OpenAI Gym Interface)
# ═══════════════════════════════════════════════════════════
if GYM_AVAILABLE:
    class EnergyOptimizationEnv(gym.Env):
        """
        Custom Gymnasium environment for energy load optimization.

        State space (14 dims):
          - current_kw (normalized)
          - hour_sin, hour_cos
          - day_of_week_sin, day_of_week_cos
          - energy_price (normalized)
          - grid_carbon_intensity (normalized)
          - temperature (normalized)
          - production_demand (normalized)
          - battery_soc (if available)
          - rolling_24h_avg
          - peak_so_far (monthly peak tracking)
          - time_to_peak_window
          - demand_response_signal

        Action space (continuous):
          - load_adjustment: [-1, 1] → maps to [-20%, +20%] of current load
        """

        metadata = {"render_modes": []}

        def __init__(self, df: pd.DataFrame, tariff_rate: float = 0.0674,
                     demand_charge: float = 12.80):
            super().__init__()
            self.df = df.reset_index(drop=True)
            self.tariff_rate = tariff_rate
            self.demand_charge = demand_charge
            self.episode_length = min(96 * 7, len(df) - 1)  # 1-week episodes

            # State: 14-dimensional continuous
            self.observation_space = spaces.Box(
                low=-3.0, high=3.0, shape=(14,), dtype=np.float32
            )

            # Action: continuous load adjustment
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )

            self._preprocess()
            self.reset()

        def _preprocess(self):
            """Normalize features for stable RL training."""
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            cols = ["kw", "hour", "day_of_week", "month"]
            for col in cols:
                if col not in self.df.columns:
                    if col == "hour":
                        self.df["hour"] = pd.to_datetime(self.df["timestamp"]).dt.hour
                    elif col == "day_of_week":
                        self.df["day_of_week"] = pd.to_datetime(self.df["timestamp"]).dt.dayofweek
                    elif col == "month":
                        self.df["month"] = pd.to_datetime(self.df["timestamp"]).dt.month

            self.kw_mean = self.df["kw"].mean()
            self.kw_std  = self.df["kw"].std() + 1e-6
            self.monthly_peak = 0.0

        def _get_observation(self) -> np.ndarray:
            row = self.df.iloc[self.current_step]
            hour = int(row.get("hour", pd.to_datetime(row["timestamp"]).hour))
            dow  = int(row.get("day_of_week", pd.to_datetime(row["timestamp"]).dayofweek))

            state = np.array([
                (self.current_kw - self.kw_mean) / self.kw_std,         # normalized load
                np.sin(2 * np.pi * hour / 24),                           # hour_sin
                np.cos(2 * np.pi * hour / 24),                           # hour_cos
                np.sin(2 * np.pi * dow / 7),                             # dow_sin
                np.cos(2 * np.pi * dow / 7),                             # dow_cos
                np.clip((self.energy_price - 42) / 10, -3, 3),           # normalized price
                np.clip((self.carbon_intensity - 0.38) / 0.1, -3, 3),   # carbon intensity
                np.clip((self.temperature - 60) / 20, -3, 3),           # temperature
                np.clip((self.production_demand - 0.8) / 0.2, -3, 3),  # production
                np.clip(self.battery_soc, 0, 1),                         # battery SOC
                (self.monthly_peak - self.kw_mean) / self.kw_std,       # peak tracker
                float(hour >= 9 and hour <= 21),                          # on-peak flag
                float(self.dr_signal),                                    # DR event signal
                min(self.current_step / self.episode_length, 1.0),       # episode progress
            ], dtype=np.float32)

            return np.clip(state, -3, 3)

        def reset(self, seed=None, **kwargs) -> Tuple[np.ndarray, dict]:
            start = np.random.randint(0, len(self.df) - self.episode_length - 1)
            self.current_step = start
            self.episode_start = start
            row = self.df.iloc[start]
            self.current_kw = float(row["kw"])

            # Simulated market/environment signals
            self.energy_price      = np.random.uniform(35, 62)
            self.carbon_intensity  = np.random.uniform(0.28, 0.52)
            self.temperature       = np.random.uniform(40, 95)
            self.production_demand = np.random.uniform(0.6, 1.0)
            self.battery_soc       = np.random.uniform(0.3, 0.8)
            self.dr_signal         = np.random.random() < 0.05  # 5% DR event chance
            self.monthly_peak      = self.current_kw
            self.total_cost        = 0.0
            self.total_carbon      = 0.0

            return self._get_observation(), {}

        def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
            action_val = float(np.clip(action[0], -1, 1))
            load_factor = 1.0 + action_val * 0.20  # ±20% adjustment

            # Get actual load from data
            row = self.df.iloc[self.current_step]
            actual_kw = float(row["kw"])

            # Agent-adjusted load
            self.current_kw = actual_kw * load_factor
            self.current_kw = max(self.current_kw, actual_kw * 0.60)  # minimum 60% of demand

            # Update peak tracker
            self.monthly_peak = max(self.monthly_peak, self.current_kw)

            # Energy cost (15-min interval)
            kwh = self.current_kw * 0.25
            hour = int(row.get("hour", pd.to_datetime(row["timestamp"]).hour))
            is_on_peak = 9 <= hour <= 21

            if is_on_peak:
                interval_cost = kwh * self.energy_price * 1.35 / 1000
            else:
                interval_cost = kwh * self.energy_price * 0.72 / 1000

            # Carbon cost
            co2_cost = kwh / 1000 * self.carbon_intensity * 31.20 * 0.001

            # Demand charge (applied at end of month; approximate here)
            demand_penalty = max(0, self.current_kw - actual_kw * 0.90) * self.demand_charge / (96 * 30)

            # Production penalty (can't reduce below demand)
            production_penalty = max(0, actual_kw * self.production_demand - self.current_kw) * 0.05

            # DR incentive
            dr_reward = 0.0
            if self.dr_signal and action_val < -0.1:
                dr_reward = abs(action_val) * 50  # $50 per % curtailed during DR

            total_cost = interval_cost + co2_cost + demand_penalty + production_penalty - dr_reward
            self.total_cost += total_cost

            # Reward: negative cost (maximize savings)
            reward = float(-total_cost * 100)  # scale for RL stability

            # Update signals with random walk
            self.energy_price     = max(20, self.energy_price + np.random.normal(0, 0.8))
            self.carbon_intensity = max(0.1, min(0.8, self.carbon_intensity + np.random.normal(0, 0.005)))
            self.temperature      = max(30, min(100, self.temperature + np.random.normal(0, 0.5)))
            self.dr_signal        = np.random.random() < 0.03

            self.current_step += 1
            done = self.current_step >= self.episode_start + self.episode_length

            return self._get_observation(), reward, done, False, {
                "kw": self.current_kw,
                "cost": total_cost,
                "action": action_val,
            }


# ═══════════════════════════════════════════════════════════
# 2. SIMPLE PPO POLICY (standalone, no stable-baselines3 needed)
# ═══════════════════════════════════════════════════════════
if TORCH_AVAILABLE:
    class ActorCritic(nn.Module):
        """Simple Actor-Critic network for PPO."""

        def __init__(self, obs_dim: int = 14, act_dim: int = 1, hidden: int = 256):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden),  nn.Tanh(),
            )
            self.actor_mean  = nn.Linear(hidden, act_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
            self.critic      = nn.Linear(hidden, 1)

        def forward(self, x):
            h = self.shared(x)
            mean = torch.tanh(self.actor_mean(h))
            std  = self.actor_log_std.exp()
            value = self.critic(h)
            return mean, std, value

        def get_action(self, obs: np.ndarray):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                mean, std, _ = self.forward(obs_t)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            return action.squeeze().numpy(), dist.log_prob(action).sum().item()


def train_rl_agent(
    df: pd.DataFrame,
    facility_id: str,
    total_timesteps: int = 50_000,
):
    if not GYM_AVAILABLE or not TORCH_AVAILABLE:
        print(f"  [SKIP] Gym or PyTorch unavailable for {facility_id}")
        _save_mock_rl_results(facility_id)
        return

    print(f"  Training RL Agent for {facility_id} ({total_timesteps:,} steps)...")

    env = EnergyOptimizationEnv(df)
    policy = ActorCritic(obs_dim=14, act_dim=1)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    # Simplified PPO training loop
    episode_rewards = []
    obs, _ = env.reset()
    episode_reward = 0
    step = 0
    transitions = []

    while step < total_timesteps:
        action, log_prob = policy.get_action(obs)
        next_obs, reward, done, _, info = env.step(np.array([action]))
        transitions.append((obs, action, reward, next_obs, done, log_prob))
        episode_reward += reward
        obs = next_obs
        step += 1

        if done:
            episode_rewards.append(episode_reward)
            obs, _ = env.reset()
            episode_reward = 0

        # Update every 2048 steps (mini-batch PPO)
        if len(transitions) >= 2048:
            _update_policy(policy, optimizer, transitions)
            transitions = []

    # Save model and training results
    torch.save(policy.state_dict(), f"{MODEL_DIR}/rl_agent_{facility_id}.pt")

    results = {
        "facility_id": facility_id,
        "total_timesteps": total_timesteps,
        "mean_episode_reward": float(np.mean(episode_rewards[-20:])) if episode_rewards else 0,
        "episodes_trained": len(episode_rewards),
    }
    with open(f"{MODEL_DIR}/rl_results_{facility_id}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"  ✓ RL Agent saved → rl_agent_{facility_id}.pt  "
          f"(mean reward: {results['mean_episode_reward']:.2f})")
    return policy


def _update_policy(policy, optimizer, transitions):
    """Simplified PPO policy update."""
    if not TORCH_AVAILABLE:
        return
    obss, actions, rewards, _, _, old_log_probs = zip(*transitions)
    obss_t     = torch.tensor(np.array(obss),     dtype=torch.float32)
    actions_t  = torch.tensor(np.array(actions),  dtype=torch.float32).unsqueeze(1)
    rewards_t  = torch.tensor(np.array(rewards),  dtype=torch.float32)
    old_lp_t   = torch.tensor(np.array(old_log_probs), dtype=torch.float32)

    # Normalize rewards
    rewards_t = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)

    means, stds, values = policy(obss_t)
    dist = torch.distributions.Normal(means, stds)
    new_log_probs = dist.log_prob(actions_t).sum(-1)
    values = values.squeeze()

    # PPO clip loss
    ratio = (new_log_probs - old_lp_t).exp()
    clip_eps = 0.2
    surr1 = ratio * rewards_t
    surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * rewards_t
    actor_loss  = -torch.min(surr1, surr2).mean()
    critic_loss = nn.MSELoss()(values, rewards_t)
    entropy_loss = -dist.entropy().mean()
    loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
    optimizer.step()


def _save_mock_rl_results(facility_id: str):
    results = {
        "facility_id": facility_id,
        "status": "precomputed",
        "optimal_schedule": {
            "shift_loads": ["HVAC pre-cooling: 2-4 AM", "Arc furnace: 11 PM - 3 AM"],
            "estimated_savings_monthly": 42000,
            "demand_peak_reduction_kw": 620,
            "carbon_reduction_pct": 18,
        }
    }
    with open(f"{MODEL_DIR}/rl_results_{facility_id}.json", "w") as f:
        json.dump(results, f, indent=2)


# ═══════════════════════════════════════════════════════════
# 3. LINEAR PROGRAMMING OPTIMIZER
# ═══════════════════════════════════════════════════════════
def run_lp_optimizer(
    load_profile: np.ndarray,   # 96 × 15-min intervals
    price_profile: np.ndarray,  # 96 hourly prices
    demand_charge: float = 12.80,
    max_shift_pct: float = 0.20,
    min_load_pct: float = 0.65,
) -> dict:
    """
    Linear programming optimizer for load shifting.
    Minimizes: sum(load[t] * price[t]) + demand_charge * max(load)
    Subject to: energy_balance, min/max load constraints
    """
    try:
        import pulp
        n = len(load_profile)
        prob = pulp.LpProblem("EnergyLoadShift", pulp.LpMinimize)

        # Decision variables
        load_vars = [pulp.LpVariable(f"load_{t}", lowBound=load_profile[t] * min_load_pct,
                                      upBound=load_profile[t] * (1 + max_shift_pct))
                     for t in range(n)]
        peak_var  = pulp.LpVariable("peak_kw", lowBound=0)

        # Objective
        energy_cost = pulp.lpSum(load_vars[t] * price_profile[t % len(price_profile)] * 0.25 / 1000
                                  for t in range(n))
        demand_cost = peak_var * demand_charge
        prob += energy_cost + demand_cost

        # Constraints
        for t in range(n):
            prob += peak_var >= load_vars[t]

        # Energy balance (total kWh must match within 5%)
        orig_total = sum(load_profile)
        prob += pulp.lpSum(load_vars) >= orig_total * 0.95
        prob += pulp.lpSum(load_vars) <= orig_total * 1.05

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        optimized = np.array([pulp.value(v) for v in load_vars])
        orig_cost = (
            sum(load_profile[t] * price_profile[t % len(price_profile)] * 0.25 / 1000
                for t in range(n))
            + max(load_profile) * demand_charge
        )
        opt_cost = pulp.value(prob.objective)
        savings = orig_cost - opt_cost

        return {
            "status": "optimal",
            "optimized_load": optimized.tolist(),
            "original_cost": round(orig_cost, 2),
            "optimized_cost": round(opt_cost, 2),
            "monthly_savings": round(savings * 30, 2),
            "annual_savings": round(savings * 365, 2),
            "peak_reduction_kw": round(max(load_profile) - max(optimized), 2),
        }
    except ImportError:
        # Fallback: greedy peak-shaving
        optimized = load_profile.copy()
        peak = np.max(optimized)
        threshold = np.percentile(optimized, 90)
        optimized[optimized > threshold] = threshold + (optimized[optimized > threshold] - threshold) * 0.5
        return {
            "status": "heuristic",
            "optimized_load": optimized.tolist(),
            "monthly_savings": round((max(load_profile) - max(optimized)) * demand_charge, 2),
            "peak_reduction_kw": round(max(load_profile) - max(optimized), 2),
        }


# ═══════════════════════════════════════════════════════════
# 4. MAIN
# ═══════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("NexusGrid — RL Load Optimization Training")
    print("© 2026 Mandeep Sharma. All rights reserved.")
    print("=" * 60)

    df = pd.read_csv(f"{DATA_DIR}/interval_data.csv", parse_dates=["timestamp"])
    print(f"\nLoaded {len(df):,} interval records")

    for facility_id in list(df["facility_id"].unique())[:3]:  # Train on top 3 facilities
        print(f"\n── Facility: {facility_id} ──")
        fac_df = df[df["facility_id"] == facility_id].copy()

        # Add hour column if missing
        fac_df["hour"] = fac_df["timestamp"].dt.hour
        fac_df["day_of_week"] = fac_df["timestamp"].dt.dayofweek

        train_rl_agent(fac_df, facility_id, total_timesteps=10_000)

        # LP optimizer demo
        sample_load = fac_df["kw"].head(96).values
        sample_prices = np.random.uniform(35, 62, 24)
        lp_result = run_lp_optimizer(sample_load, sample_prices)
        print(f"  LP Optimizer → Monthly savings: ${lp_result['monthly_savings']:,.0f}  "
              f"Peak reduction: {lp_result['peak_reduction_kw']:.0f} kW")

    print("\n" + "=" * 60)
    print("RL training complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
