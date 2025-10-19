"""
Comprehensive Test Runner

Runs all test suites and generates a detailed report:
- Unit tests (mathematical correctness)
- Edge case tests (boundary conditions)
- Integration tests (end-to-end)
- Performance tests (speed, memory)

Generates:
- Console output with test results
- JSON report with metrics
- HTML report (optional)
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestResult:
    """Store result of a single test."""

    def __init__(self, name: str, suite: str, passed: bool, duration: float, error: str = None):
        self.name = name
        self.suite = suite
        self.passed = passed
        self.duration = duration
        self.error = error

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'suite': self.suite,
            'passed': self.passed,
            'duration_ms': self.duration * 1000,
            'error': self.error,
        }


class TestSuite:
    """Run a test suite and collect results."""

    def __init__(self, name: str, test_class):
        self.name = name
        self.test_class = test_class
        self.results: List[TestResult] = []
        self.total_duration = 0

    def run(self):
        """Run all tests in the suite."""
        print(f"\n{'='*60}")
        print(f"{self.name}")
        print(f"{'='*60}")

        test_instance = self.test_class()
        test_methods = [m for m in dir(test_instance) if m.startswith('test_')]

        for method_name in test_methods:
            start_time = time.time()
            passed = False
            error = None

            try:
                method = getattr(test_instance, method_name)
                method()
                passed = True
                status = "✓"
            except Exception as e:
                error = str(e)
                status = "✗"

            duration = time.time() - start_time
            self.total_duration += duration

            result = TestResult(
                name=method_name,
                suite=self.name,
                passed=passed,
                duration=duration,
                error=error
            )
            self.results.append(result)

            # Print result
            print(f"  {status} {method_name} ({duration*1000:.1f}ms)")
            if error and len(error) < 100:
                print(f"    Error: {error}")

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed

        return {
            'suite': self.name,
            'total': len(self.results),
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / len(self.results) if self.results else 0,
            'duration_sec': self.total_duration,
        }


class TestRunner:
    """Main test runner."""

    def __init__(self):
        self.suites: List[TestSuite] = []
        self.start_time = None
        self.end_time = None

    def add_suite(self, name: str, test_class):
        """Add a test suite."""
        self.suites.append(TestSuite(name, test_class))

    def run_all(self):
        """Run all test suites."""
        print("=" * 60)
        print("TOROIDAL ATTENTION - COMPREHENSIVE TEST SUITE")
        print("=" * 60)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        self.start_time = time.time()

        for suite in self.suites:
            try:
                suite.run()
            except Exception as e:
                print(f"\n✗ Suite {suite.name} crashed: {e}")

        self.end_time = time.time()

        self.print_summary()

    def print_summary(self):
        """Print overall summary."""
        total_duration = self.end_time - self.start_time

        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        total_tests = 0
        total_passed = 0
        total_failed = 0

        for suite in self.suites:
            summary = suite.get_summary()
            total_tests += summary['total']
            total_passed += summary['passed']
            total_failed += summary['failed']

            status = "✓" if summary['failed'] == 0 else "✗"
            print(f"{status} {summary['suite']}: {summary['passed']}/{summary['total']} passed "
                  f"({summary['pass_rate']*100:.1f}%) in {summary['duration_sec']:.2f}s")

        print("-" * 60)
        print(f"TOTAL: {total_passed}/{total_tests} passed ({total_passed/total_tests*100:.1f}%)")
        print(f"Total duration: {total_duration:.2f}s")
        print("=" * 60)

        # Overall pass/fail
        if total_failed == 0:
            print("\n✓ ALL TESTS PASSED!")
        else:
            print(f"\n✗ {total_failed} TESTS FAILED")

            # List failed tests
            print("\nFailed tests:")
            for suite in self.suites:
                for result in suite.results:
                    if not result.passed:
                        print(f"  - {suite.name}::{result.name}")
                        if result.error:
                            print(f"    {result.error[:100]}")

    def generate_report(self, output_path: Path):
        """Generate JSON report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_duration_sec': self.end_time - self.start_time,
            'suites': []
        }

        for suite in self.suites:
            suite_data = suite.get_summary()
            suite_data['tests'] = [r.to_dict() for r in suite.results]
            report['suites'].append(suite_data)

        # Calculate overall stats
        report['overall'] = {
            'total_tests': sum(s.get_summary()['total'] for s in self.suites),
            'total_passed': sum(s.get_summary()['passed'] for s in self.suites),
            'total_failed': sum(s.get_summary()['failed'] for s in self.suites),
        }
        report['overall']['pass_rate'] = (
            report['overall']['total_passed'] / report['overall']['total_tests']
            if report['overall']['total_tests'] > 0 else 0
        )

        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n📊 Test report saved to: {output_path}")


def main():
    """Main entry point."""
    runner = TestRunner()

    # Add all test suites
    print("Loading test suites...")

    # Unit tests (mathematical correctness)
    try:
        from test_toroidal_attention import (
            TestDepthFusion,
            TestToroidal3DPositionalEncoding,
            TestToroidalAttention,
            TestToroidalDistance,
        )

        runner.add_suite("PE Tests", TestToroidal3DPositionalEncoding)
        runner.add_suite("Distance Tests", TestToroidalDistance)
        runner.add_suite("Fusion Tests", TestDepthFusion)
        runner.add_suite("Attention Tests", TestToroidalAttention)
        print("  ✓ Loaded unit tests")
    except ImportError as e:
        print(f"  ⚠ Could not load unit tests: {e}")

    # Edge case tests
    try:
        from test_edge_cases import (
            TestDepthFusionEdgeCases,
            TestDistanceMetricEdgeCases,
            TestInvalidInputs,
            TestLargeDimensions,
            TestMaskingEdgeCases,
            TestMinimalDimensions,
            TestNumericalStability,
            TestPositionalEncodingEdgeCases,
            TestSequenceLengthVariation,
        )

        runner.add_suite("Minimal Dimensions", TestMinimalDimensions)
        runner.add_suite("Large Dimensions", TestLargeDimensions)
        runner.add_suite("Numerical Stability", TestNumericalStability)
        runner.add_suite("Invalid Inputs", TestInvalidInputs)
        runner.add_suite("Sequence Length Variation", TestSequenceLengthVariation)
        runner.add_suite("Masking Edge Cases", TestMaskingEdgeCases)
        runner.add_suite("Distance Edge Cases", TestDistanceMetricEdgeCases)
        runner.add_suite("PE Edge Cases", TestPositionalEncodingEdgeCases)
        runner.add_suite("Fusion Edge Cases", TestDepthFusionEdgeCases)
        print("  ✓ Loaded edge case tests")
    except ImportError as e:
        print(f"  ⚠ Could not load edge case tests: {e}")

    # Integration tests
    try:
        from test_integration import (
            TestCheckpointSaveLoad,
            TestConfigurationManagement,
            TestCrossComponentInteraction,
            TestDataPipeline,
            TestForwardBackwardPass,
            TestMetricsTracking,
            TestModelIntegration,
        )

        runner.add_suite("Data Pipeline", TestDataPipeline)
        runner.add_suite("Model Integration", TestModelIntegration)
        runner.add_suite("Forward/Backward", TestForwardBackwardPass)
        runner.add_suite("Checkpoint Save/Load", TestCheckpointSaveLoad)
        runner.add_suite("Configuration", TestConfigurationManagement)
        runner.add_suite("Metrics Tracking", TestMetricsTracking)
        runner.add_suite("Cross-Component", TestCrossComponentInteraction)
        print("  ✓ Loaded integration tests")
    except ImportError as e:
        print(f"  ⚠ Could not load integration tests: {e}")

    # Performance tests
    try:
        from test_performance import (
            TestComparativePerformance,
            TestInferenceSpeed,
            TestMemoryUsage,
            TestScalability,
        )

        runner.add_suite("Memory Usage", TestMemoryUsage)
        runner.add_suite("Inference Speed", TestInferenceSpeed)
        runner.add_suite("Comparative Performance", TestComparativePerformance)
        runner.add_suite("Scalability", TestScalability)
        print("  ✓ Loaded performance tests")
    except ImportError as e:
        print(f"  ⚠ Could not load performance tests: {e}")

    if not runner.suites:
        print("\n✗ No test suites loaded!")
        sys.exit(1)

    print(f"\nTotal test suites: {len(runner.suites)}")

    # Run all tests
    runner.run_all()

    # Generate report
    report_path = Path(__file__).parent / "test_report.json"
    runner.generate_report(report_path)

    # Exit with appropriate code
    total_failed = sum(s.get_summary()['failed'] for s in runner.suites)
    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()

