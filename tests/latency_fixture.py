"""Create the isolated synthetic Optuna run used by the browser latency probe."""

from typing import Optional, Sequence, Tuple

import argparse
import pickle
from pathlib import Path

import optuna


def _objective(trial: optuna.Trial) -> Tuple[float, float]:
    """Return deterministic-shape objectives over mixed parameter types."""
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1.0, log=True)
    depth = trial.suggest_int("depth", 1, 12)
    booster = trial.suggest_categorical("booster", ["gbtree", "dart", "linear"])
    penalty = 0.0 if booster == "gbtree" else 0.1
    loss = (learning_rate - 0.1) ** 2 + (depth - 6) ** 2 / 100 + penalty
    score = 1.0 / (1.0 + (depth - 6) ** 2)
    return loss, score


def create_fixture(output: Path, trials: int) -> Path:
    """Write a reproducible pickled study and return its run directory."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        directions=["minimize", "maximize"],
        sampler=optuna.samplers.TPESampler(seed=7),
    )
    study.optimize(_objective, n_trials=trials)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "study.pkl").open("wb") as file_handle:
        pickle.dump(study, file_handle)
    return output


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Create the command-line-selected fixture."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/tmp/deepcave-latency/run"))
    parser.add_argument("--trials", type=int, default=120)
    arguments = parser.parse_args(argv)
    print(create_fixture(arguments.output, arguments.trials))


if __name__ == "__main__":
    main()
