"""Experiment runner for multi-armed bandits.

This module orchestrates the end-to-end experiment for two bandit algorithms:
    * EpsilonGreedy  (ε(t) = 1 / t)
    * ThompsonSampling (Gaussian rewards with known precision)

It:
    1) Instantiates both algorithms with TRUE_MEANS.
    2) Runs the experiment loops (.experiment()) for each algorithm.
    3) Collects per-trial results via .report() and writes a combined CSV.
    4) Renders the required visualizations:
        - Learning curves (running average reward)
        - Cumulative rewards
        - Cumulative regrets
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger

from algorithms import EpsilonGreedy, ThompsonSampling

# Ground-truth arm means and CSV destination for per-trial results
TRUE_MEANS = np.array([1, 2, 3, 4])
CSV_PATH = "bandit_results.csv"


def plot_learning_curves(run_avg_dict: dict[str, np.ndarray], optimal_mean: float) -> None:
    """Plot the learning process (running average reward) for each algorithm.

    Args:
        run_avg_dict (dict[str, np.ndarray]): Mapping from algorithm name to the
            per-trial running average reward array (shape: [n_trials]).
        optimal_mean (float): Best possible expected reward (max(TRUE_MEANS)),
            plotted as a dashed reference line.
    """
    plt.figure()
    for name, arr in run_avg_dict.items():
        plt.plot(arr, label=name)
    plt.axhline(optimal_mean, linestyle='--', label='Optimal mean')
    plt.title("Learning Curves")
    plt.xlabel("Trial")
    plt.ylabel("Running Average Reward")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_cumulative_rewards(cum_rewards_dict: dict[str, np.ndarray]) -> None:
    """Plot cumulative reward curves for all algorithms.

    Args:
        cum_rewards_dict (dict[str, np.ndarray]): Mapping from algorithm name to
            the cumulative reward array (shape: [n_trials]).
    """
    plt.figure()
    for name, arr in cum_rewards_dict.items():
        plt.plot(arr, label=name)
    plt.title("Cumulative Rewards")
    plt.xlabel("Trial")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_cumulative_regrets(cum_regrets_dict: dict[str, np.ndarray]) -> None:
    """Plot cumulative (sample) regret curves for all algorithms.

    Args:
        cum_regrets_dict (dict[str, np.ndarray]): Mapping from algorithm name to
            the cumulative regret array (shape: [n_trials]). This is *sample*
            regret as computed by the base Bandit: sum_t (optimal_mean - r_t),
            which may dip below zero for Gaussian rewards when r_t > optimal_mean.
    """
    plt.figure()
    for name, arr in cum_regrets_dict.items():
        plt.plot(arr, label=name)
    plt.title("Cumulative Regrets")
    plt.xlabel("Trial")
    plt.ylabel("Cumulative Regret")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Run EpsilonGreedy and ThompsonSampling, save CSV, and plot figures.

    Side effects:
        * Writes a CSV file at CSV_PATH with columns:
              {Bandit, Reward, Algorithm}
          concatenating per-trial rows from both algorithms.
        * Displays three matplotlib figures:
              - Learning Curves
              - Cumulative Rewards
              - Cumulative Regrets
        * Logs cumulative reward and regret for each algorithm via loguru.
    """
    eg = EpsilonGreedy(TRUE_MEANS)
    ts = ThompsonSampling(TRUE_MEANS)

    eg.experiment()
    ts.experiment()

    df_eg = eg.report()
    df_ts = ts.report()

    out = pd.concat([df_eg, df_ts], ignore_index=True)
    out.to_csv(CSV_PATH, index=False)
    logger.info(f"Saved combined results to {CSV_PATH}")

    optimal_mean = np.max(TRUE_MEANS)
    plot_learning_curves(
        {
            "EpsilonGreedy": eg.history["run_avg_reward"],
            "ThompsonSampling": ts.history["run_avg_reward"],
        },
        optimal_mean,
    )

    plot_cumulative_rewards(
        {
            "EpsilonGreedy": eg.history["cum_rewards"],
            "ThompsonSampling": ts.history["cum_rewards"],
        }
    )

    plot_cumulative_regrets(
        {
            "EpsilonGreedy": eg.history["cum_regret"],
            "ThompsonSampling": ts.history["cum_regret"],
        }
    )


if __name__ == "__main__":
    main()
