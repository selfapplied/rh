#!/usr/bin/env python3
"""
Computational Validation of Mathematical Theory

This script validates the mathematical theory developed for the Riemann Hypothesis
computational framework, specifically testing:

1. First-moment cancellation on σ=1/2
2. Off-line linear growth away from critical line
3. Connection between computational certificates and zeta zeros
"""

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# Import our computational framework
from riemann.analysis.rh_analyzer import RHIntegerAnalyzer


@dataclass
class ValidationResult:
    """Results from mathematical theory validation"""
    test_name: str
    expected: bool
    actual: bool
    confidence: float
    details: Dict[str, Any]
    timestamp: str


class MathematicalTheoryValidator:
    """
    Validates the mathematical theory of the RH computational framework
    """
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        
    def test_first_moment_cancellation(self, 
                                     known_rh_zeros: List[float],
                                     depth: int = 4,
                                     window_size: float = 0.5) -> ValidationResult:
        """
        Test first-moment cancellation on σ=1/2 for known RH zeros
        """
        print("Testing first-moment cancellation on σ=1/2...")
        
        analyzer = RHIntegerAnalyzer(depth=depth)
        cancellation_results = []
        
        for zero_imag in known_rh_zeros:
            # Test on the critical line σ=1/2
            sigma = 0.5
            t = zero_imag
            
            # Generate certificate
            cert = analyzer.analyze_point(sigma, t, window_size=window_size)
            
            # Check if first-moment cancellation occurs
            # This should manifest as a small smoothed drift
            smoothed_drift = cert.get('smoothed_drift', float('inf'))
            
            # First-moment cancellation should give small drift
            cancellation_occurred = abs(smoothed_drift) < 0.01  # Threshold for cancellation
            
            cancellation_results.append({
                'zero': zero_imag,
                'smoothed_drift': smoothed_drift,
                'cancellation': cancellation_occurred
            })
            
            print(f"  Zero {zero_imag:.6f}: drift={smoothed_drift:.6f}, cancelled={cancellation_occurred}")
        
        # Calculate success rate
        successful_cancellations = sum(1 for r in cancellation_results if r['cancellation'])
        success_rate = successful_cancellations / len(cancellation_results)
        
        result = ValidationResult(
            test_name="first_moment_cancellation",
            expected=True,
            actual=success_rate > 0.8,  # Expect >80% success rate
            confidence=success_rate,
            details={
                'success_rate': success_rate,
                'results': cancellation_results,
                'depth': depth,
                'window_size': window_size
            },
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        
        self.results.append(result)
        return result
    
    def test_off_line_linear_growth(self,
                                  test_points: List[Tuple[float, float]],  # (sigma, t) pairs
                                  depth: int = 4,
                                  window_size: float = 0.5) -> ValidationResult:
        """
        Test linear growth away from the critical line σ=1/2
        """
        print("Testing off-line linear growth...")
        
        analyzer = RHIntegerAnalyzer(depth=depth)
        growth_results = []
        
        for sigma, t in test_points:
            # Generate certificate
            cert = analyzer.analyze_point(sigma, t, window_size=window_size)
            
            # Get smoothed drift
            smoothed_drift = cert.get('smoothed_drift', 0.0)
            
            # Calculate distance from critical line
            distance_from_critical = abs(sigma - 0.5)
            
            # Off-line points should have larger drift
            growth_occurred = abs(smoothed_drift) > 0.1 * distance_from_critical
            
            growth_results.append({
                'sigma': sigma,
                't': t,
                'distance_from_critical': distance_from_critical,
                'smoothed_drift': smoothed_drift,
                'growth': growth_occurred
            })
            
            print(f"  Point ({sigma:.3f}, {t:.3f}): distance={distance_from_critical:.3f}, "
                  f"drift={smoothed_drift:.6f}, growth={growth_occurred}")
        
        # Calculate success rate
        successful_growths = sum(1 for r in growth_results if r['growth'])
        success_rate = successful_growths / len(growth_results)
        
        result = ValidationResult(
            test_name="off_line_linear_growth",
            expected=True,
            actual=success_rate > 0.7,  # Expect >70% success rate
            confidence=success_rate,
            details={
                'success_rate': success_rate,
                'results': growth_results,
                'depth': depth,
                'window_size': window_size
            },
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        
        self.results.append(result)
        return result
    
    def test_certificate_generation(self,
                                  known_rh_zeros: List[float],
                                  off_line_points: List[Tuple[float, float]],
                                  depth: int = 4) -> ValidationResult:
        """
        Test that certificates are generated for RH zeros but not for off-line points
        """
        print("Testing certificate generation...")
        
        analyzer = RHIntegerAnalyzer(depth=depth)
        certificate_results = []
        
        # Test known RH zeros (should generate certificates)
        for zero_imag in known_rh_zeros:
            sigma = 0.5  # Critical line
            t = zero_imag
            
            cert = analyzer.analyze_point(sigma, t)
            certificate_generated = cert.get('certificate_generated', False)
            
            certificate_results.append({
                'point_type': 'rh_zero',
                'sigma': sigma,
                't': t,
                'certificate_generated': certificate_generated,
                'expected': True
            })
            
            print(f"  RH Zero {zero_imag:.6f}: certificate={certificate_generated}")
        
        # Test off-line points (should not generate certificates)
        for sigma, t in off_line_points:
            cert = analyzer.analyze_point(sigma, t)
            certificate_generated = cert.get('certificate_generated', False)
            
            certificate_results.append({
                'point_type': 'off_line',
                'sigma': sigma,
                't': t,
                'certificate_generated': certificate_generated,
                'expected': False
            })
            
            print(f"  Off-line ({sigma:.3f}, {t:.3f}): certificate={certificate_generated}")
        
        # Calculate accuracy
        correct_predictions = sum(1 for r in certificate_results 
                                if r['certificate_generated'] == r['expected'])
        accuracy = correct_predictions / len(certificate_results)
        
        result = ValidationResult(
            test_name="certificate_generation",
            expected=True,
            actual=accuracy > 0.8,  # Expect >80% accuracy
            confidence=accuracy,
            details={
                'accuracy': accuracy,
                'results': certificate_results,
                'depth': depth
            },
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        
        self.results.append(result)
        return result
    
    def test_convergence_properties(self,
                                  test_zero: float,
                                  depths: List[int] = [3, 4, 5, 6]) -> ValidationResult:
        """
        Test that the computational framework converges as depth increases
        """
        print("Testing convergence properties...")
        
        convergence_results = []
        
        for depth in depths:
            analyzer = RHIntegerAnalyzer(depth=depth)
            sigma = 0.5
            t = test_zero
            
            cert = analyzer.analyze_point(sigma, t)
            smoothed_drift = cert.get('smoothed_drift', float('inf'))
            gap = cert.get('dihedral_gap', 0.0)
            
            convergence_results.append({
                'depth': depth,
                'smoothed_drift': smoothed_drift,
                'dihedral_gap': gap,
                'N': 2**depth + 1
            })
            
            print(f"  Depth {depth}: N={2**depth + 1}, drift={smoothed_drift:.6f}, gap={gap:.3f}")
        
        # Check if smoothed drift approaches zero (first-moment cancellation)
        drifts = [r['smoothed_drift'] for r in convergence_results]
        convergence_occurred = all(abs(drifts[i]) <= abs(drifts[i-1]) for i in range(1, len(drifts)))
        
        result = ValidationResult(
            test_name="convergence_properties",
            expected=True,
            actual=convergence_occurred,
            confidence=1.0 - max(abs(d) for d in drifts) if drifts else 0.0,
            details={
                'convergence_occurred': convergence_occurred,
                'results': convergence_results,
                'final_drift': drifts[-1] if drifts else 0.0
            },
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        
        self.results.append(result)
        return result
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive validation report
        """
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.actual == r.expected)
        overall_success_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        
        report = {
            'validation_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'overall_success_rate': overall_success_rate,
                'validation_timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ")
            },
            'individual_results': [
                {
                    'test_name': r.test_name,
                    'passed': r.actual == r.expected,
                    'confidence': r.confidence,
                    'details': r.details,
                    'timestamp': r.timestamp
                }
                for r in self.results
            ]
        }
        
        return report
    
    def save_report(self, filename: str = None):
        """
        Save validation report to file
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"mathematical_theory_validation_{timestamp}.json"
        
        report = self.generate_validation_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Validation report saved to: {filename}")
        return filename


def main():
    """
    Main validation function
    """
    print("🔬 Mathematical Theory Validation")
    print("=" * 50)
    
    # Known RH zeros (first few)
    known_rh_zeros = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062]
    
    # Off-line test points
    off_line_points = [
        (0.4, 14.0), (0.6, 14.0),  # Near first RH zero but off-line
        (0.3, 21.0), (0.7, 21.0),  # Near second RH zero but off-line
        (0.45, 25.0), (0.55, 25.0)  # Near third RH zero but off-line
    ]
    
    # Initialize validator
    validator = MathematicalTheoryValidator()
    
    # Run validation tests
    print("\n1. Testing First-Moment Cancellation")
    validator.test_first_moment_cancellation(known_rh_zeros)
    
    print("\n2. Testing Off-Line Linear Growth")
    validator.test_off_line_linear_growth(off_line_points)
    
    print("\n3. Testing Certificate Generation")
    validator.test_certificate_generation(known_rh_zeros, off_line_points)
    
    print("\n4. Testing Convergence Properties")
    validator.test_convergence_properties(known_rh_zeros[0])
    
    # Generate and save report
    print("\n5. Generating Validation Report")
    report = validator.generate_validation_report()
    
    print(f"\n📊 Validation Summary:")
    print(f"   Total Tests: {report['validation_summary']['total_tests']}")
    print(f"   Passed Tests: {report['validation_summary']['passed_tests']}")
    print(f"   Success Rate: {report['validation_summary']['overall_success_rate']:.1%}")
    
    # Save report
    filename = validator.save_report()
    
    print(f"\n✅ Mathematical theory validation complete!")
    print(f"   Report saved to: {filename}")
    
    return validator


if __name__ == "__main__":
    validator = main()