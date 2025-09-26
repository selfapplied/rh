#!/usr/bin/env python3
"""
Convergence Analysis from Mathematical Ledger

This analyzes the convergence rates of our computations using the ledger data
to understand how our numbers stabilize as we add more terms.
"""

import numpy as np
import math
import json
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt

class ConvergenceAnalyzer:
    """
    Analyzes convergence rates from the mathematical ledger data.
    """
    
    def __init__(self, ledger_file: str = "mathematical_ledger.json"):
        """Initialize with ledger data."""
        with open(ledger_file, 'r') as f:
            self.ledger = json.load(f)
        
        self.operations = self.ledger['operations']
    
    def analyze_series_convergence(self) -> Dict[str, Any]:
        """
        Analyze how the series sum converges as we add more terms.
        
        The series is: ∑_{n=1}^N 1/n² → π²/6
        """
        print("ANALYZING SERIES CONVERGENCE")
        print("=" * 40)
        
        # Extract series terms from ledger
        series_terms = []
        for op in self.operations:
            if (op['operation_type'] == 'series_summation' and 
                'n' in op['input_values'] and 
                op['input_values']['n'] <= 1000):  # Individual terms
                n = int(op['input_values']['n'])
                term_value = op['output_value']
                series_terms.append((n, term_value))
        
        # Sort by n
        series_terms.sort(key=lambda x: x[0])
        
        # Compute partial sums
        partial_sums = []
        cumulative_sum = 0.0
        theoretical_limit = math.pi**2 / 6
        
        for n, term in series_terms:
            cumulative_sum += term
            error = abs(cumulative_sum - theoretical_limit)
            partial_sums.append((n, cumulative_sum, error))
        
        # Analyze convergence rate
        convergence_data = {
            'terms': series_terms,
            'partial_sums': partial_sums,
            'theoretical_limit': theoretical_limit,
            'final_sum': cumulative_sum,
            'final_error': abs(cumulative_sum - theoretical_limit)
        }
        
        # Print convergence analysis
        print(f"Theoretical limit (π²/6): {theoretical_limit:.10f}")
        print(f"Computed sum (1000 terms): {cumulative_sum:.10f}")
        print(f"Error: {convergence_data['final_error']:.2e}")
        
        # Show convergence at key points
        key_points = [10, 50, 100, 500, 1000]
        print(f"\nConvergence at key points:")
        for n in key_points:
            if n <= len(partial_sums):
                _, sum_val, error = partial_sums[n-1]
                print(f"  n = {n:4d}: sum = {sum_val:.8f}, error = {error:.2e}")
        
        return convergence_data
    
    def analyze_prime_sum_convergence(self) -> Dict[str, Any]:
        """
        Analyze how the prime sum converges as we add more primes.
        
        The prime sum is: ∑_{p≡a(8)} (log p)/√p
        """
        print(f"\nANALYZING PRIME SUM CONVERGENCE")
        print("=" * 40)
        
        # Extract prime contributions from ledger
        prime_contributions = []
        for op in self.operations:
            if (op['operation_type'] == 'prime_sum_calculation' and 
                'prime' in op['input_values']):
                prime = int(op['input_values']['prime'])
                contribution = op['output_value']
                prime_contributions.append((prime, contribution))
        
        # Sort by prime
        prime_contributions.sort(key=lambda x: x[0])
        
        # Compute partial sums
        partial_sums = []
        cumulative_sum = 0.0
        
        for prime, contribution in prime_contributions:
            cumulative_sum += contribution
            partial_sums.append((prime, contribution, cumulative_sum))
        
        # Analyze convergence
        convergence_data = {
            'prime_contributions': prime_contributions,
            'partial_sums': partial_sums,
            'final_sum': cumulative_sum,
            'num_primes': len(prime_contributions)
        }
        
        # Print convergence analysis
        print(f"Number of primes used: {len(prime_contributions)}")
        print(f"Final prime sum: {cumulative_sum:.6f}")
        
        # Show convergence at key points
        key_indices = [10, 25, 50, 100, len(prime_contributions)]
        print(f"\nPrime sum convergence:")
        for i in key_indices:
            if i <= len(partial_sums):
                prime, contrib, sum_val = partial_sums[i-1]
                print(f"  Prime #{i:3d} (p={prime:3d}): contrib={contrib:.6f}, cumulative={sum_val:.6f}")
        
        # Analyze individual contributions
        contributions = [contrib for _, contrib in prime_contributions]
        print(f"\nContribution analysis:")
        print(f"  Max contribution: {max(contributions):.6f}")
        print(f"  Min contribution: {min(contributions):.6f}")
        print(f"  Average contribution: {np.mean(contributions):.6f}")
        print(f"  Std deviation: {np.std(contributions):.6f}")
        
        return convergence_data
    
    def analyze_threshold_convergence(self, series_data: Dict, prime_data: Dict) -> Dict[str, Any]:
        """
        Analyze how the threshold t_0 = C_A/C_P converges.
        """
        print(f"\nANALYZING THRESHOLD CONVERGENCE")
        print("=" * 40)
        
        # Compute C_A at different series truncations
        series_partial_sums = series_data['partial_sums']
        C_A_values = []
        
        for n, sum_val, error in series_partial_sums:
            C_A = 0.5 * sum_val  # Factor of 1/2
            C_A_values.append((n, C_A))
        
        # Compute C_P at different prime truncations
        prime_partial_sums = prime_data['partial_sums']
        C_P_values = []
        
        for i, (prime, contrib, sum_val) in enumerate(prime_partial_sums):
            C_P = sum_val / (i + 1)  # Average contribution
            C_P_values.append((i + 1, C_P))
        
        # Compute thresholds at different truncation levels
        threshold_values = []
        
        # For each series truncation level
        for n, C_A in C_A_values:
            # Find the closest prime truncation level
            closest_prime_idx = min(len(C_P_values) - 1, n // 6)  # Rough scaling
            if closest_prime_idx >= 0:
                _, C_P = C_P_values[closest_prime_idx]
                threshold = C_A / C_P
                threshold_values.append((n, threshold))
        
        convergence_data = {
            'C_A_values': C_A_values,
            'C_P_values': C_P_values,
            'threshold_values': threshold_values,
            'final_threshold': threshold_values[-1][1] if threshold_values else 0
        }
        
        # Print convergence analysis
        print(f"Final threshold: {convergence_data['final_threshold']:.6f}")
        
        # Show convergence at key points
        key_points = [100, 200, 500, 1000]
        print(f"\nThreshold convergence:")
        for n in key_points:
            if n <= len(threshold_values):
                _, threshold = threshold_values[n-1]
                print(f"  n = {n:4d}: threshold = {threshold:.6f}")
        
        return convergence_data
    
    def estimate_convergence_rates(self, series_data: Dict, prime_data: Dict, threshold_data: Dict) -> Dict[str, Any]:
        """
        Estimate convergence rates for each computation.
        """
        print(f"\nESTIMATING CONVERGENCE RATES")
        print("=" * 40)
        
        # Series convergence rate (should be O(1/n) for 1/n²)
        series_partial_sums = series_data['partial_sums']
        theoretical_limit = series_data['theoretical_limit']
        
        # Fit error = C/n^α
        n_values = [n for n, _, _ in series_partial_sums[-100:]]  # Last 100 points
        error_values = [error for _, _, error in series_partial_sums[-100:]]
        
        # Log-log regression to find α
        log_n = [math.log(n) for n in n_values]
        log_error = [math.log(error) for error in error_values]
        
        if len(log_n) > 1:
            # Simple linear regression
            n_mean = np.mean(log_n)
            error_mean = np.mean(log_error)
            numerator = sum((log_n[i] - n_mean) * (log_error[i] - error_mean) for i in range(len(log_n)))
            denominator = sum((log_n[i] - n_mean)**2 for i in range(len(log_n)))
            alpha = -numerator / denominator if denominator > 0 else 1.0
        else:
            alpha = 1.0
        
        # Prime sum convergence rate (should be O(log p/√p))
        prime_partial_sums = prime_data['partial_sums']
        if len(prime_partial_sums) > 10:
            # Analyze how contributions decay
            contributions = [contrib for _, contrib, _ in prime_partial_sums[-20:]]
            primes = [prime for prime, _, _ in prime_partial_sums[-20:]]
            
            # Expected: log(p)/√p decay
            expected_decay = [math.log(p) / math.sqrt(p) for p in primes]
            
            # Compare with actual contributions
            decay_ratio = [contrib / expected for contrib, expected in zip(contributions, expected_decay)]
            avg_decay_ratio = np.mean(decay_ratio)
        else:
            avg_decay_ratio = 1.0
        
        convergence_rates = {
            'series_rate': {
                'alpha': alpha,
                'formula': f'Error = O(1/n^{alpha:.2f})',
                'interpretation': 'How fast series approaches π²/6'
            },
            'prime_rate': {
                'decay_ratio': avg_decay_ratio,
                'formula': 'Contribution ≈ (log p/√p)',
                'interpretation': 'How individual prime contributions decay'
            },
            'threshold_rate': {
                'stability': 'Depends on series and prime convergence',
                'interpretation': 'How threshold stabilizes with more terms'
            }
        }
        
        # Print results
        print(f"Series convergence rate: α = {alpha:.3f}")
        print(f"  {convergence_rates['series_rate']['formula']}")
        print(f"  {convergence_rates['series_rate']['interpretation']}")
        
        print(f"\nPrime contribution decay: ratio = {avg_decay_ratio:.3f}")
        print(f"  {convergence_rates['prime_rate']['formula']}")
        print(f"  {convergence_rates['prime_rate']['interpretation']}")
        
        print(f"\nThreshold stability: {convergence_rates['threshold_rate']['interpretation']}")
        
        return convergence_rates
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run complete convergence analysis."""
        
        print("CONVERGENCE ANALYSIS FROM MATHEMATICAL LEDGER")
        print("=" * 60)
        
        # Analyze series convergence
        series_data = self.analyze_series_convergence()
        
        # Analyze prime sum convergence
        prime_data = self.analyze_prime_sum_convergence()
        
        # Analyze threshold convergence
        threshold_data = self.analyze_threshold_convergence(series_data, prime_data)
        
        # Estimate convergence rates
        rates_data = self.estimate_convergence_rates(series_data, prime_data, threshold_data)
        
        return {
            'series_convergence': series_data,
            'prime_convergence': prime_data,
            'threshold_convergence': threshold_data,
            'convergence_rates': rates_data
        }

def main():
    """Run convergence analysis."""
    analyzer = ConvergenceAnalyzer()
    results = analyzer.run_complete_analysis()
    
    return results

if __name__ == "__main__":
    results = main()
