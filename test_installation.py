"""
Quick installation test script.

Run this after installing the package to verify everything works.
"""

import sys


def test_import():
    """Test that package can be imported."""
    print("Testing import...")
    try:
        import dimensionality_reduction as dr
        print(f"✅ Package imported successfully")
        print(f"   Version: {dr.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import package: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality."""
    print("\nTesting basic functionality...")
    try:
        import dimensionality_reduction as dr
        import numpy as np
        
        # Test 1: Generate random space
        space = dr.get_random_space(10, 5)
        assert space.shape == (10, 5), "Wrong shape from get_random_space"
        print("✅ Random space generation works")
        
        # Test 2: Compute distances
        distances = dr.space_to_dist(space)
        assert distances.shape == (10, 10), "Wrong distance matrix shape"
        print("✅ Distance computation works")
        
        # Test 3: Check if Euclidean
        is_euclidean = dr.is_euclidean_space(distances)
        assert is_euclidean, "Random space should be Euclidean"
        print("✅ Euclidean space detection works")
        
        # Test 4: Run main algorithm (small test)
        embedded = dr.approx_algo(distances, target_dimension=3, q=2)
        assert embedded.shape == (10, 3), "Wrong embedded shape"
        print("✅ Main algorithm (functional) works")
        
        # Test 5: Run class-based algorithm
        reducer = dr.ApproximationAlgorithm(q=2)
        reducer.fit(distances)
        embedded_class = reducer.transform(3)
        assert embedded_class.shape == (10, 3), "Wrong embedded shape from class"
        print("✅ Main algorithm (class-based) works")
        
        # Test 6: Measure distortion
        embedded_dists = dr.space_to_dist(embedded)
        distortion = dr.lq_dist(distances, embedded_dists, q=2)
        assert distortion > 0, "Distortion should be positive"
        print(f"✅ Distortion measurement works (distortion={distortion:.3f})")
        
        return True
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_exceptions():
    """Test that exceptions work properly."""
    print("\nTesting exception handling...")
    try:
        import dimensionality_reduction as dr
        
        # Test that proper exceptions are raised
        try:
            dr.expansion(0.0, 1.0)  # Should raise DivisionByZeroError
            print("❌ Exception not raised when expected")
            return False
        except dr.DivisionByZeroError:
            print("✅ Custom exceptions work")
            return True
    except Exception as e:
        print(f"❌ Exception test failed: {e}")
        return False


def test_all_imports():
    """Test that all main functions can be imported."""
    print("\nTesting all exports...")
    try:
        import dimensionality_reduction as dr
        
        # Check main functions exist
        functions_to_check = [
            'approx_algo',
            'lq_dist',
            'wc_distortion',
            'rem_q',
            'sigma_q',
            'energy',
            'stress',
            'is_metric_space',
            'is_euclidean_space',
            'space_from_dists',
            'space_to_dist',
            'get_random_space',
        ]
        
        # Check classes exist
        classes_to_check = [
            'ApproximationAlgorithm',
            'BaseDimensionalityReducer',
        ]
        
        for func_name in functions_to_check:
            if not hasattr(dr, func_name):
                print(f"❌ Missing function: {func_name}")
                return False
        
        for class_name in classes_to_check:
            if not hasattr(dr, class_name):
                print(f"❌ Missing class: {class_name}")
                return False
        
        print(f"✅ All {len(functions_to_check)} functions and {len(classes_to_check)} classes available")
        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Dimensionality Reduction - Installation Test")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Import", test_import()))
    
    if results[0][1]:  # Only continue if import succeeded
        results.append(("All Exports", test_all_imports()))
        results.append(("Basic Functionality", test_basic_functionality()))
        results.append(("Exception Handling", test_exceptions()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:25s}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("=" * 60)
    if all_passed:
        print("🎉 All tests passed! Installation successful!")
        print("\nNext steps:")
        print("  1. Read the documentation: docs/QUICKSTART.md")
        print("  2. Try examples: python examples/comparison_experiments.py")
        print("  3. Run full test suite: pytest")
        return 0
    else:
        print("❌ Some tests failed. Please check the installation.")
        print("\nTroubleshooting:")
        print("  1. Reinstall: pip install -e .")
        print("  2. Check dependencies: pip install -r requirements.txt")
        print("  3. See docs/INSTALLATION.md for help")
        return 1


if __name__ == "__main__":
    sys.exit(main())

