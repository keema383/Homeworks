"""Concrete bandit algorithms.

This module implements two policies that inherit from the abstract :class:`Bandit`
interface:

* :class:`EpsilonGreedy` — ε-greedy with time-decay ε(t)=1/t
* :class:`ThompsonSampling` — Gaussian Thompson Sampling with known precision

Both classes rely on the base class to:
    - set up the Gaussian environment (known precision: variance = 1 / beta_env),
    - run the experiment loop (:meth:`Bandit.experiment`),
    - and produce a per-trial DataFrame + logs in :meth:`Bandit.report`.

Only policy-specific behavior is implemented here:
    - how an arm is selected in :meth:`pull`,
    - how internal state is updated in :meth:`update`.
"""

from typing import Tuple
import math
import numpy as np
from loguru import logger

# If your file is named `bandit.py` (lowercase), use: from bandit import Bandit
from Bandit import Bandit


class EpsilonGreedy(Bandit):
    """Epsilon-Greedy policy with time-decay epsilon: ε(t) = 1 / t.

    The policy explores with probability ε(t) and exploits (chooses any arm
    that maximizes the current value estimate) with probability 1−ε(t).
    Value estimates are maintained as running means per arm.

    Notes:
        * Rewards are assumed Gaussian with known precision (set in base class).
        * The environment RNG and trial counter `_t` are initialized/managed
          by :meth:`Bandit.experiment`.
    """

    def __init__(self, p):
        """Initialize ε-greedy state.

        Args:
            p (array-like): True mean reward for each arm (e.g., [1, 2, 3, 4]).
        """
        super().__init__(p)
        self.counts = np.zeros(self.k, dtype=int)   # pulls per arm
        self.values = np.zeros(self.k, dtype=float) # running mean reward per arm
        self._rng_policy = np.random.default_rng(123)

    def __repr__(self) -> str:
        """Return a concise representation used in logs."""
        return f"EpsilonGreedy(k={self.k}, trials={self.n_trials})"

    def pull(self) -> Tuple[int, float]:
        """Select an arm using ε-greedy, sample reward, and update state.

        Policy:
            ε(t) = 1 / t. With prob ε(t), pick a random arm (explore).
            Otherwise, pick any arm with maximal current estimate (exploit).

        Returns:
            tuple[int, float]: (chosen arm index, observed Gaussian reward).
        """
        eps = 1.0 / max(1, self._t)

        # Explore vs. exploit
        if self._rng_policy.random() < eps:
            arm = int(self._rng_policy.integers(0, self.k))
        else:
            m = np.max(self.values)
            ties = np.flatnonzero(np.isclose(self.values, m))
            arm = int(self._rng_policy.choice(ties))

        # Sample Gaussian reward with known precision (variance = 1 / beta_env)
        reward = float(
            self._rng_env.normal(self.true_means[arm], math.sqrt(1.0 / self.beta_env))
        )

        # Update internal estimates
        self.update(arm, reward)
        return arm, reward

    def update(self, arm: int, reward: float) -> None:
        """Incrementally update the running-mean estimate for a chosen arm.

        Args:
            arm (int): Index of the chosen arm.
            reward (float): Observed reward from the environment.
        """
        self.counts[arm] += 1
        n = self.counts[arm]
        # incremental mean update
        self.values[arm] += (reward - self.values[arm]) / n

    # Abstracts re-exposed to satisfy ABC, delegating to the base implementation.
    def experiment(self):
        """Run the base experiment loop and record history (delegates to base)."""
        return super().experiment()

    def report(self):
        """Return per-trial DataFrame and log aggregates (delegates to base)."""
        return super().report()


class ThompsonSampling(Bandit):
    """Thompson Sampling for Gaussian rewards with known precision (beta).

    Conjugate Normal-Normal model per arm:

        Prior:     μ ~ Normal(mu0, 1 / tau0)
        Likelihood with known precision beta (var = 1/beta)
        Posterior: μ | data ~ Normal(mu_n, 1 / tau_n)
            where tau_n = tau0 + n * beta
                  mu_n  = (tau0 * mu0 + beta * sum_x) / tau_n

    At each step, draw one sample θ_j from each arm's posterior and pick the arm
    with the largest sample.

    Notes:
        * This implementation maintains sufficient statistics (n, sum_x) per arm.
        * The environment RNG and trial counter `_t` are initialized/managed
          by :meth:`Bandit.experiment`.
    """

    def __init__(self, p):
        """Initialize TS hyperparameters and sufficient statistics.

        Args:
            p (array-like): True mean reward for each arm.
        """
        super().__init__(p)
        # Known observation precision (likelihood)
        self.beta = 1.0
        # Weak Normal prior
        self.mu0 = 0.0
        self.tau0 = 1e-6

        # Sufficient stats per arm
        self.n = np.zeros(self.k, dtype=int)
        self.sum_x = np.zeros(self.k, dtype=float)

        self._rng_policy = np.random.default_rng(456)

    def __repr__(self) -> str:
        """Return a concise representation used in logs."""
        return f"ThompsonSampling(k={self.k}, trials={self.n_trials})"

    def _posterior_params(self, a: int) -> tuple[float, float]:
        """Compute Normal posterior parameters for arm `a`.

        Args:
            a (int): Arm index.

        Returns:
            tuple[float, float]: (mu_n, var_n) for the posterior Normal.
        """
        tau_n = self.tau0 + self.n[a] * self.beta
        mu_n = (self.tau0 * self.mu0 + self.beta * self.sum_x[a]) / tau_n
        var_n = 1.0 / tau_n
        return mu_n, var_n

    def pull(self) -> Tuple[int, float]:
        """Sample a draw from each arm's posterior and select the best arm.

        Returns:
            tuple[int, float]: (chosen arm index, observed Gaussian reward).
        """
        samples = np.zeros(self.k)
        for a in range(self.k):
            mu_n, var_n = self._posterior_params(a)
            samples[a] = self._rng_policy.normal(mu_n, math.sqrt(var_n))

        arm = int(np.argmax(samples))

        reward = float(
            self._rng_env.normal(self.true_means[arm], math.sqrt(1.0 / self.beta_env))
        )
        self.update(arm, reward)
        return arm, reward

    def update(self, arm: int, reward: float) -> None:
        """Update sufficient statistics for the chosen arm.

        Args:
            arm (int): Index of the chosen arm.
            reward (float): Observed reward from the environment.
        """
        self.n[arm] += 1
        self.sum_x[arm] += reward

    # Abstracts re-exposed to satisfy ABC, delegating to the base implementation.
    def experiment(self):
        """Run the base experiment loop and record history (delegates to base)."""
        return super().experiment()

    def report(self):
        """Return per-trial DataFrame and log aggregates (delegates to base)."""
        return super().report()
