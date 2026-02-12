#!/usr/bin/env python3
"""
Benchmark script demonstrating ALBL with CoreSet, BALD, and InformationDensity.

This script compares the performance of ALBL using the query strategies:
- CoreSet: Diversity via k-Center Greedy (farthest from labeled set)
- BALD: Epistemic uncertainty via ensemble disagreement
- InformationDensity: Density-weighted uncertainty (avoids querying outliers)

Usage:
    python albl_new_strategies_benchmark.py

Requirements:
    - scikit-learn
    - numpy
    - matplotlib (for plotting)
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# libact imports
from libact.base.dataset import Dataset
from libact.models import LogisticRegression
from libact.query_strategies import (
    ActiveLearningByLearning,
    UncertaintySampling,
    RandomSampling,
    CoreSet,
    BALD,
    InformationDensity,
)
from libact.labelers import IdealLabeler


def create_synthetic_dataset(n_samples=500, n_features=20, n_informative=10,
                             n_redundant=5, n_classes=2, random_state=42):
    """Create a synthetic classification dataset."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        random_state=random_state,
        flip_y=0.1  # Add some label noise
    )
    return X, y


def run_experiment(trn_ds, tst_ds, labeler, model, qs, quota):
    """Run active learning experiment and return error curves."""
    E_in, E_out = [], []

    for _ in range(quota):
        try:
            ask_id = qs.make_query()
        except ValueError:
            # No more unlabeled samples or budget exhausted
            break

        label = labeler.label(trn_ds.data[ask_id][0])
        trn_ds.update(ask_id, label)

        model.train(trn_ds)
        E_in.append(1 - model.score(trn_ds))
        E_out.append(1 - model.score(tst_ds))

    return np.array(E_in), np.array(E_out)


def main():
    print("=" * 60)
    print("ALBL Benchmark with CoreSet, BALD & InformationDensity")
    print("=" * 60)

    # Parameters
    n_labeled = 10  # Initial labeled samples
    quota = 100     # Number of queries to make
    n_repeats = 5   # Number of experiment repetitions
    random_state = 42

    # Create dataset
    print("\nCreating synthetic dataset...")
    X, y = create_synthetic_dataset(n_samples=500, random_state=random_state)

    results = {
        'Random': [],
        'UncertaintySampling': [],
        'CoreSet': [],
        'BALD': [],
        'InformationDensity': [],
        'ALBL (All)': [],
    }

    for rep in range(n_repeats):
        print(f"\nRepetition {rep + 1}/{n_repeats}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=random_state + rep
        )

        # Ensure initial labels have both classes
        while len(np.unique(y_train[:n_labeled])) < 2:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=random_state + rep + 100
            )

        # Create datasets
        def make_trn_ds():
            labels = list(y_train[:n_labeled]) + [None] * (len(y_train) - n_labeled)
            return Dataset(X_train, labels)

        tst_ds = Dataset(X_test, y_test)
        fully_labeled = Dataset(X_train, y_train)
        labeler = IdealLabeler(fully_labeled)

        # Model factory
        def make_model():
            return LogisticRegression(solver='liblinear')

        # 1. Random Sampling
        print("  Running Random Sampling...")
        trn_ds = make_trn_ds()
        qs = RandomSampling(trn_ds, random_state=random_state)
        _, E_out = run_experiment(trn_ds, tst_ds, labeler, make_model(), qs, quota)
        results['Random'].append(E_out)

        # 2. Uncertainty Sampling
        print("  Running Uncertainty Sampling...")
        trn_ds = make_trn_ds()
        qs = UncertaintySampling(trn_ds, model=make_model(), method='lc')
        _, E_out = run_experiment(trn_ds, tst_ds, labeler, make_model(), qs, quota)
        results['UncertaintySampling'].append(E_out)

        # 3. CoreSet
        print("  Running CoreSet...")
        trn_ds = make_trn_ds()
        qs = CoreSet(trn_ds, random_state=random_state)
        _, E_out = run_experiment(trn_ds, tst_ds, labeler, make_model(), qs, quota)
        results['CoreSet'].append(E_out)

        # 4. BALD
        print("  Running BALD...")
        trn_ds = make_trn_ds()
        qs = BALD(
            trn_ds,
            models=[
                LogisticRegression(solver='liblinear', C=0.1),
                LogisticRegression(solver='liblinear', C=1.0),
                LogisticRegression(solver='liblinear', C=10.0),
            ],
            random_state=random_state
        )
        _, E_out = run_experiment(trn_ds, tst_ds, labeler, make_model(), qs, quota)
        results['BALD'].append(E_out)

        # 5. InformationDensity
        print("  Running InformationDensity...")
        trn_ds = make_trn_ds()
        qs = InformationDensity(
            trn_ds,
            model=make_model(),
            method='entropy',
            random_state=random_state
        )
        _, E_out = run_experiment(trn_ds, tst_ds, labeler, make_model(), qs, quota)
        results['InformationDensity'].append(E_out)

        # 6. ALBL with all strategies combined
        print("  Running ALBL with all strategies...")
        trn_ds = make_trn_ds()
        qs = ActiveLearningByLearning(
            trn_ds,
            query_strategies=[
                CoreSet(trn_ds, random_state=random_state),
                BALD(
                    trn_ds,
                    models=[
                        LogisticRegression(solver='liblinear', C=c)
                        for c in [0.1, 1.0, 10.0]
                    ],
                    random_state=random_state
                ),
                InformationDensity(
                    trn_ds,
                    model=make_model(),
                    method='entropy',
                    random_state=random_state
                ),
            ],
            T=quota,
            uniform_sampler=True,
            model=make_model(),
            random_state=random_state
        )
        _, E_out = run_experiment(trn_ds, tst_ds, labeler, make_model(), qs, quota)
        results['ALBL (All)'].append(E_out)

    # Compute mean results
    print("\n" + "=" * 60)
    print("Computing mean results...")

    mean_results = {}
    for name, runs in results.items():
        # Pad shorter runs to same length
        max_len = max(len(r) for r in runs)
        padded = [np.pad(r, (0, max_len - len(r)), mode='edge') for r in runs]
        mean_results[name] = np.mean(padded, axis=0)

    # Print final errors
    print("\nFinal Test Error (mean over {} runs):".format(n_repeats))
    for name, errors in mean_results.items():
        print(f"  {name}: {errors[-1]:.4f}")

    # Plot results
    print("\nGenerating plot...")
    plt.figure(figsize=(10, 6))

    colors = {
        'Random': 'gray',
        'UncertaintySampling': 'blue',
        'CoreSet': 'orange',
        'BALD': 'red',
        'InformationDensity': 'green',
        'ALBL (All)': 'purple',
    }

    for name, errors in mean_results.items():
        plt.plot(range(1, len(errors) + 1), errors,
                 label=name, color=colors[name], linewidth=2)

    plt.xlabel('Number of Queries', fontsize=12)
    plt.ylabel('Test Error', fontsize=12)
    plt.title('Active Learning Benchmark: CoreSet, BALD & InformationDensity',
              fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(__file__), 'albl_new_strategies_results.png')
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")
    plt.show()


if __name__ == '__main__':
    main()
