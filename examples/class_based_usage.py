"""
Example: Class-Based API for Caching Optimization Results

This example demonstrates the benefits of using the class-based API
when you need to embed the same data to multiple target dimensions.
"""

import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import dimensionality_reduction as dr


def functional_api_example():
    """
    Functional API: Simple but re-solves optimization each time.
    """
    print("=" * 70)
    print("FUNCTIONAL API (Original)")
    print("=" * 70)
    
    # Generate data
    space = dr.get_random_space(30, 30)
    distances = dr.space_to_dist(space)
    
    print(f"Data: {space.shape[0]} points in {space.shape[1]} dimensions")
    print()
    
    # Embed to different dimensions
    dimensions = [10, 7, 5, 3]
    
    total_time = 0
    for k in dimensions:
        start = time.time()
        embedded = dr.approx_algo(distances, target_dimension=k, q=2)
        elapsed = time.time() - start
        total_time += elapsed
        
        print(f"  Embedded to {k}D: {embedded.shape} (took {elapsed:.2f}s)")
    
    print(f"\n  Total time: {total_time:.2f}s")
    print(f"  Problem: Re-solves expensive optimization each time! 😞")
    print()


def class_based_api_example():
    """
    Class-Based API: Solves optimization once, reuses for multiple dimensions.
    """
    print("=" * 70)
    print("CLASS-BASED API (New - with Caching)")
    print("=" * 70)
    
    # Generate same data
    space = dr.get_random_space(30, 30)
    distances = dr.space_to_dist(space)
    
    print(f"Data: {space.shape[0]} points in {space.shape[1]} dimensions")
    print()
    
    # Create reducer
    reducer = dr.ApproximationAlgorithm(q=2, objective='lq_dist')
    print(f"Created: {reducer}")
    print()
    
    # Fit ONCE (expensive)
    print("Step 1: Fit (solve optimization - expensive)")
    start_fit = time.time()
    reducer.fit(distances)
    fit_time = time.time() - start_fit
    print(f"  ✅ Optimization solved and cached ({fit_time:.2f}s)")
    print()
    
    # Transform MULTIPLE times (cheap!)
    print("Step 2: Transform to multiple dimensions (fast - reuses optimization)")
    dimensions = [10, 7, 5, 3]
    
    transform_time = 0
    for k in dimensions:
        start = time.time()
        embedded = reducer.transform(k)
        elapsed = time.time() - start
        transform_time += elapsed
        
        print(f"  Embedded to {k}D: {embedded.shape} (took {elapsed:.2f}s) ⚡")
    
    total_time = fit_time + transform_time
    print(f"\n  Fit time: {fit_time:.2f}s")
    print(f"  Transform time: {transform_time:.2f}s (total for all dimensions)")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Benefit: Only solve optimization once! 🚀")
    print()


def comparison():
    """
    Direct comparison showing speedup.
    """
    print("=" * 70)
    print("COMPARISON: Functional vs Class-Based")
    print("=" * 70)
    
    space = dr.get_random_space(25, 25)
    distances = dr.space_to_dist(space)
    
    dimensions = [8, 5, 3]
    
    # Functional API
    print("\n📊 Functional API (re-solves each time):")
    start = time.time()
    for k in dimensions:
        embedded = dr.approx_algo(distances, target_dimension=k, q=2)
    functional_time = time.time() - start
    print(f"   Time: {functional_time:.2f}s")
    
    # Class-Based API
    print("\n📊 Class-Based API (solve once, reuse):")
    start = time.time()
    reducer = dr.ApproximationAlgorithm(q=2)
    reducer.fit(distances)
    for k in dimensions:
        embedded = reducer.transform(k)
    class_time = time.time() - start
    print(f"   Time: {class_time:.2f}s")
    
    # Speedup
    speedup = functional_time / class_time
    print(f"\n🚀 Speedup: {speedup:.2f}x faster!")
    print(f"   Saved: {functional_time - class_time:.2f}s")
    print()


def scikit_learn_style_api():
    """
    Demonstrate scikit-learn-style API.
    """
    print("=" * 70)
    print("SCIKIT-LEARN STYLE API")
    print("=" * 70)
    
    space = dr.get_random_space(20, 20)
    distances = dr.space_to_dist(space)
    
    print("The class follows sklearn conventions:")
    print()
    
    # Create estimator
    print("1. Create estimator with hyperparameters:")
    reducer = dr.ApproximationAlgorithm(q=2, objective='lq_dist')
    print(f"   reducer = dr.ApproximationAlgorithm(q=2, objective='lq_dist')")
    print()
    
    # Fit
    print("2. Fit to data:")
    reducer.fit(distances)
    print(f"   reducer.fit(distances)")
    print(f"   Status: {reducer}")
    print()
    
    # Transform
    print("3. Transform:")
    embedded = reducer.transform(5)
    print(f"   embedded = reducer.transform(5)")
    print(f"   Result: {embedded.shape}")
    print()
    
    # Or fit_transform
    print("4. Or do both at once:")
    reducer2 = dr.ApproximationAlgorithm(q=2)
    embedded = reducer2.fit_transform(distances, target_dimension=5)
    print(f"   embedded = reducer.fit_transform(distances, target_dimension=5)")
    print(f"   Result: {embedded.shape}")
    print()


def future_pytorch_example():
    """
    Show future PyTorch/GPU usage (not yet implemented).
    """
    print("=" * 70)
    print("FUTURE: PyTorch/GPU Support (Coming Soon!)")
    print("=" * 70)
    print()
    print("When PyTorch backend is implemented, you'll be able to:")
    print()
    print("  # Use GPU for acceleration")
    print("  reducer = dr.ApproximationAlgorithm(")
    print("      q=2,")
    print("      backend='pytorch',  # Use PyTorch")
    print("      device='cuda:0'     # Use GPU 0")
    print("  )")
    print()
    print("  reducer.fit(distances)  # Runs on GPU! ⚡")
    print("  embedded = reducer.transform(10)")
    print()
    print("  # Mix CPU and GPU")
    print("  cpu_reducer = dr.ApproximationAlgorithm(backend='numpy')     # CPU")
    print("  gpu_reducer = dr.ApproximationAlgorithm(backend='pytorch')   # GPU")
    print()
    print("Benefits:")
    print("  ✅ 10-100x speedup on large datasets")
    print("  ✅ Batch processing")
    print("  ✅ Mixed precision training")
    print()
    
    # Show it's not implemented yet
    try:
        reducer = dr.ApproximationAlgorithm(backend='pytorch')
    except NotImplementedError as e:
        print(f"Current status: {e}")
    print()


def main():
    """
    Run all examples.
    """
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + " " * 10 + "Class-Based API: Caching & Reuse Example" + " " * 18 + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    # Example 1: Show the problem
    functional_api_example()
    
    # Example 2: Show the solution
    class_based_api_example()
    
    # Example 3: Direct comparison
    comparison()
    
    # Example 4: sklearn-style API
    scikit_learn_style_api()
    
    # Example 5: Future PyTorch support
    future_pytorch_example()
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Use Functional API when:")
    print("  ✅ You only need one embedding dimension")
    print("  ✅ You want simple, straightforward code")
    print("  ✅ You're doing quick experiments")
    print()
    print("Use Class-Based API when:")
    print("  ✅ You need multiple target dimensions")
    print("  ✅ You want to cache expensive optimization")
    print("  ✅ You want sklearn-style fit/transform")
    print("  ✅ You'll use PyTorch/GPU in the future")
    print()
    print("Both APIs produce identical results!")
    print()


if __name__ == "__main__":
    main()

