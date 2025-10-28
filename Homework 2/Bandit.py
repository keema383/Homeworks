"""Bandit base module.

This module defines the abstract :class:`Bandit` interface and provides a
common, fully functional experiment loop and reporting logic suitable for
Gaussian multi-armed bandits with known observation precision.

Usage pattern (subclasses must implement policy-specific behavior):
    1) Subclass :class:`Bandit` and implement:
        - __repr__(self)
        - pull(self)            # choose an arm, sample reward, call update(...)
        - update(self, arm, reward)
    2) Instantiate the subclass with true means (e.g., [1, 2, 3, 4]).
    3) Call .experiment() to run the loop for ``n_trials`` steps.
    4) Call .report() to obtain a per-trial DataFrame and log aggregates.

Notes:
    * Reward model: Gaussian with known precision ``beta_env`` (variance = 1 / beta_env).
    * ``experiment()`` computes **sample cumulative regret** as
      sum_t (optimal_mean − observed_reward_t). This may become negative due to
      noise exceeding the optimal mean on some trials. If you need *pseudo-regret*
      (based on arm means, always ≥ 0), compute and store it in a subclass
      or a custom runner as: sum_t (optimal_mean − true_means[arm_t]).

"""

from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import pandas as pd
import math


class Bandit(ABC):
    """Abstract base class for bandit algorithms.

    Subclasses must not remove any methods from this interface. You may add
    additional attributes or helper methods in subclasses as needed.

    The base class implements:
        * Environment configuration (Gaussian with known precision).
        * A generic experiment loop over ``n_trials`` that calls the subclass
          policy via :meth:`pull` and :meth:`update`, and records history.
        * A reporting helper that returns a tidy per-trial DataFrame and logs
          aggregate cumulative reward/regret.

    Attributes:
        true_means (np.ndarray): True mean reward of each arm, shape (k,).
        k (int): Number of arms.
        n_trials (int): Number of interaction steps to run in :meth:`experiment`.
        optimal_mean (float): Maximum of ``true_means``; used for regret.
        beta_env (float): Known observation precision (variance = 1 / beta_env).
        history (dict[str, np.ndarray]): Filled by :meth:`experiment` with
            - 'arms': chosen arm index per trial, shape (n_trials,)
            - 'rewards': observed reward per trial, shape (n_trials,)
            - 'cum_rewards': cumulative reward, shape (n_trials,)
            - 'cum_regret': sample cumulative regret, shape (n_trials,)
            - 'run_avg_reward': running average reward, shape (n_trials,)
        _rng_env (np.random.Generator | None): Environment RNG (set in experiment()).
        _t (int): Current trial index (1-based) set during :meth:`experiment`.
    """

    @abstractmethod
    def __init__(self, p):
        """Initialize the bandit environment and bookkeeping.

        Args:
            p (array-like): True mean reward of each arm (e.g., ``[1, 2, 3, 4]``).

        Notes:
            This initializer also sets default values for:
                - ``n_trials`` = 20_000
                - ``beta_env`` = 1.0 (known precision ⇒ variance = 1 / 1.0 = 1.0)
        """
        self.true_means = np.array(p, dtype=float)
        self.k = len(self.true_means)
        self.n_trials = 20000
        self.optimal_mean = float(np.max(self.true_means))
        self.beta_env = 1.0  # known precision (variance = 1/beta_env)
        self.history = {}
        self._rng_env = None
        self._t = 0

    @abstractmethod
    def __repr__(self):
        """Return a concise, informative name used in logging/debugging."""
        pass

    @abstractmethod
    def pull(self):
        """Execute one trial: select arm, draw reward, update algorithm state.

        The typical policy flow in a subclass is:
            1) Choose arm according to the policy/state (e.g., ε-greedy or TS).
            2) Sample reward from the Gaussian environment:
                   reward ~ Normal(true_means[arm], variance = 1 / beta_env)
            3) Call :meth:`update(arm, reward)` to update the policy state.
            4) Return ``(arm, reward)``.

        Returns:
            tuple[int, float]: The selected arm index and the observed reward.
        """
        pass

    @abstractmethod
    def update(self, arm, reward):
        """Update the policy’s internal state based on the latest observation.

        Args:
            arm (int): Index of the chosen arm.
            reward (float): Observed reward sampled from the environment.
        """
        pass

    @abstractmethod
    def experiment(self):
        """Run the full interaction loop and record rewards/regret.

        This method:
            * Initializes the environment RNG.
            * Iterates for ``n_trials`` steps, calling :meth:`pull` at each step.
            * Tracks:
                - arms: chosen arm index per trial
                - rewards: observed reward per trial
                - cum_rewards: cumulative sum of rewards
                - cum_regret: sample cumulative regret (optimal_mean − observed reward)
                - run_avg_reward: cumulative average reward per trial
            * Stores all arrays in :attr:`history`.

        Side Effects:
            Populates :attr:`history`.

        Logs:
            - A start and completion message including the subclass's ``__repr__``.
        """
        logger.info(f"Starting experiment: {self}")
        self._rng_env = np.random.default_rng(42)

        arms = np.zeros(self.n_trials, dtype=int)
        rewards = np.zeros(self.n_trials, dtype=float)
        cum_rewards = np.zeros(self.n_trials, dtype=float)
        cum_regret = np.zeros(self.n_trials, dtype=float)
        run_avg = np.zeros(self.n_trials, dtype=float)

        total = 0.0
        total_regret = 0.0
        for t in range(1, self.n_trials + 1):
            self._t = t
            arm, r = self.pull()
            arms[t - 1] = arm
            rewards[t - 1] = r

            total += r
            cum_rewards[t - 1] = total

            # Sample (realized) regret: can be negative when r > optimal_mean
            regret = self.optimal_mean - r
            total_regret += regret
            cum_regret[t - 1] = total_regret

            run_avg[t - 1] = total / t

        self.history = dict(
            arms=arms,
            rewards=rewards,
            cum_rewards=cum_rewards,
            cum_regret=cum_regret,
            run_avg_reward=run_avg,
        )
        logger.info(f"Finished experiment: {self}")

    @abstractmethod
    def report(self):
        """Return per-trial results and log cumulative metrics.

        Returns:
            pandas.DataFrame: A tidy per-trial table with columns
                ``{Bandit, Reward, Algorithm}``, where:
                    - Bandit: chosen arm index (int)
                    - Reward: observed reward (float)
                    - Algorithm: subclass name (str)

        Notes:
            This method logs cumulative reward and *sample* cumulative regret.
            If you also need pseudo-regret, compute it in a subclass or caller
            using ``true_means`` and the chosen ``arms`` sequence.
        """
        if not self.history:
            logger.error("Run experiment() first.")
            return
        df = pd.DataFrame({
            "Bandit": self.history["arms"],
            "Reward": self.history["rewards"],
            "Algorithm": self.__class__.__name__,
        })
        cum_reward = self.history["cum_rewards"][-1]
        cum_regret = self.history["cum_regret"][-1]
        logger.info(f"[{self.__class__.__name__}] Cumulative reward: {cum_reward:.3f}")
        logger.info(f"[{self.__class__.__name__}] Cumulative regret: {cum_regret:.3f}")
        return df
