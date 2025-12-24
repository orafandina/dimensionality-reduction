"""
Comparison experiments for dimensionality reduction methods.

This script compares the approximation algorithm with popular embedding methods
including PCA, t-SNE, and Johnson-Lindenstrauss projection.
"""

import logging
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.random_projection import GaussianRandomProjection

# Add parent directory to path for local imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import dimensionality_reduction as dr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def jl_transform(space: np.ndarray, k: int) -> np.ndarray:
    """
    Apply Johnson-Lindenstrauss transform.

    Parameters
    ----------
    space : np.ndarray
        Input vectors (n x d)
    k : int
        Target dimension

    Returns
    -------
    np.ndarray
        Embedded vectors (n x k)
    """
    transformer = GaussianRandomProjection(k)
    return transformer.fit_transform(space)


def pca_transform(space: np.ndarray, k: int) -> np.ndarray:
    """
    Apply PCA dimensionality reduction.

    Parameters
    ----------
    space : np.ndarray
        Input vectors (n x d)
    k : int
        Target dimension

    Returns
    -------
    np.ndarray
        Embedded vectors (n x k)
    """
    transformer = PCA(n_components=k, whiten=False, svd_solver="full")
    return transformer.fit_transform(space)


def tsne_transform(space: np.ndarray, k: int) -> np.ndarray:
    """
    Apply t-SNE dimensionality reduction.

    Parameters
    ----------
    space : np.ndarray
        Input vectors (n x d)
    k : int
        Target dimension

    Returns
    -------
    np.ndarray
        Embedded vectors (n x k)
    """
    transformer = TSNE(
        n_components=k, init="random", learning_rate="auto", method="exact"
    )
    return transformer.fit_transform(space)


def run_dimension_range_experiment(
    distance_matrix: np.ndarray,
    dimension_range: np.ndarray,
    q: float,
    measure_type: str,
    embedding_type: str,
    num_trials: int = 10,
) -> np.ndarray:
    """
    Run embedding experiment across multiple target dimensions.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Input distance matrix (n x n)
    dimension_range : np.ndarray
        Array of target dimensions to test
    q : float
        Order parameter for distortion measure
    measure_type : str
        Distortion measure: 'lq_dist', 'rem_q', 'sigma_q', 'energy', 'stress'
    embedding_type : str
        Embedding method: 'PCA', 'TSNE', 'JL', 'approx_algo'
    num_trials : int, optional
        Number of repetitions for randomized methods (default: 10)

    Returns
    -------
    np.ndarray
        Array of distortion values for each dimension
    """
    results = np.zeros(dimension_range.shape)

    measure_dict = {
        "lq_dist": dr.lq_dist,
        "rem_q": dr.rem_q,
        "sigma_q": dr.sigma_q,
        "energy": dr.energy,
        "stress": dr.stress,
    }

    embedding_dict = {
        "PCA": pca_transform,
        "TSNE": tsne_transform,
        "JL": jl_transform,
        "approx_algo": dr.approx_algo,
    }

    measure_func = measure_dict[measure_type]
    embedding_func = embedding_dict[embedding_type]

    logger.info(f"Experiment: {embedding_type} with {measure_type}")

    if embedding_type in ["PCA", "JL", "TSNE"]:
        input_space = dr.space_from_dists(distance_matrix)

        for i, k in enumerate(dimension_range):
            distortion_sum = 0.0
            trials = 1 if embedding_type == "PCA" else num_trials

            for _ in range(trials):
                embedded_space = embedding_func(input_space, k)
                embedded_dists = dr.space_to_dist(embedded_space)
                distortion_sum += measure_func(distance_matrix, embedded_dists, q)

            results[i] = distortion_sum / trials

    else:  # approx_algo
        for i, k in enumerate(dimension_range):
            embedded_space = embedding_func(distance_matrix, k, q, measure_type)
            embedded_dists = dr.space_to_dist(embedded_space)
            results[i] = measure_func(distance_matrix, embedded_dists, q)

    return results


def plot_results(
    dimension_range: np.ndarray,
    distortions_and_labels: List[Tuple[np.ndarray, str]],
    measure_type: str,
    q: float,
) -> None:
    """
    Plot comparison of embedding methods.

    Parameters
    ----------
    dimension_range : np.ndarray
        Array of dimensions
    distortions_and_labels : List[Tuple[np.ndarray, str]]
        List of (distortion_values, method_name) tuples
    measure_type : str
        Name of distortion measure
    q : float
        Order parameter
    """
    plt.figure(figsize=(10, 6))

    for distortions, label in distortions_and_labels:
        rounded_distortions = np.around(distortions, 2)
        plt.plot(dimension_range, rounded_distortions, marker="o", label=label)

    plt.legend(loc="upper right")
    plt.xlabel("Target Dimension")
    plt.ylabel(f"{measure_type} (q={q})")
    plt.title("Comparison of Embedding Methods")
    plt.grid(True, alpha=0.3)
    plt.show()


def experiment_euclidean_data() -> None:
    """
    Experiment 1: Euclidean synthetic data.

    Compares JL, PCA, and approx_algo on random Euclidean space.
    Expected: approx_algo and JL should perform similarly and best.
    """
    logger.info("=== Experiment 1: Euclidean Data ===")

    q = 2
    measure_type = "energy"
    dimension_range = np.array([10, 15, 20])

    space = dr.get_random_space(100, 100)
    dists = dr.space_to_dist(space)

    jl_distortions = run_dimension_range_experiment(
        dists, dimension_range, q, measure_type, "JL"
    )
    pca_distortions = run_dimension_range_experiment(
        dists, dimension_range, q, measure_type, "PCA"
    )
    approx_distortions = run_dimension_range_experiment(
        dists, dimension_range, q, measure_type, "approx_algo"
    )

    results = [
        (jl_distortions, "JL"),
        (pca_distortions, "PCA"),
        (approx_distortions, "Approx_Algo"),
    ]

    plot_results(dimension_range, results, measure_type, q)


def experiment_non_euclidean_data() -> None:
    """
    Experiment 2: Non-Euclidean synthetic data.

    Compares approx_algo with PCA on non-Euclidean space.
    Expected: approx_algo should significantly outperform PCA.
    """
    logger.info("=== Experiment 2: Non-Euclidean Data ===")

    q = 2
    measure_type = "stress"
    dimension_range = np.array([3, 5, 7])

    dists = dr.get_random_non_euclidean(n=50, epsilon=0.8)

    approx_distortions = run_dimension_range_experiment(
        dists, dimension_range, q, measure_type, "approx_algo"
    )
    pca_distortions = run_dimension_range_experiment(
        dists, dimension_range, q, measure_type, "PCA"
    )

    results = [
        (approx_distortions, "Approx_Algo"),
        (pca_distortions, "PCA"),
    ]

    plot_results(dimension_range, results, measure_type, q)


def experiment_mnist_data() -> None:
    """
    Experiment 3: Real data (MNIST).

    Applies approx_algo to MNIST dataset.
    Note: Requires torchvision to be installed.
    """
    try:
        from torchvision import datasets
    except ImportError:
        logger.error("torchvision not installed. Skipping MNIST experiment.")
        logger.error("Install with: pip install torch torchvision")
        return

    logger.info("=== Experiment 3: MNIST Data ===")

    k, q = 4, 2

    # Download and load MNIST
    train_set = datasets.MNIST("./data", train=True, download=True)
    train_array = train_set.data.numpy()
    space_all = np.reshape(train_array, (60000, 784))

    # Use first 1000 points for computational efficiency
    space = space_all[:1000, :785]
    dists = dr.space_to_dist(space)

    logger.info(f"Embedding {space.shape[0]} MNIST images to {k} dimensions...")
    embedded_space = dr.approx_algo(dists, k, q, "lq_dist")

    logger.info(f"Result shape: {embedded_space.shape}")
    logger.info("MNIST experiment complete")


def main() -> None:
    """Run all experiments."""
    print("=" * 60)
    print("Dimensionality Reduction Comparison Experiments")
    print("=" * 60)
    print()

    # Experiment 1: Euclidean data
    experiment_euclidean_data()

    # Experiment 2: Non-Euclidean data
    experiment_non_euclidean_data()

    # Experiment 3: MNIST (optional)
    # Uncomment to run:
    # experiment_mnist_data()


if __name__ == "__main__":
    main()

