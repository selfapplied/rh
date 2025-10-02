#!/usr/bin/env python3
"""
Capture Detailed Mathematical Output

This modifies the bridge algorithm to capture and save the actual mathematical content
that's being generated, not just the status messages.
"""

import os
import sys
from typing import Any, Dict, List


# Import our bridge algorithm
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from riemann.bridge import MathematicalBridgeAlgorithm


class DetailedOutputCapture:
    """Captures detailed mathematical output from the bridge algorithm."""
    
    def __init__(self):
        self.bridge = MathematicalBridgeAlgorithm()
        self.detailed_output = []
        
    def capture_observation_processing(self, observation: str) -> Dict[str, Any]:
        """Process one observation and capture all detailed output."""
        
        print(f"\n=== PROCESSING: {observation} ===")
        
        # Capture the mathematical objects created
        mathematical_objects = {}
        proof_components = []
        
        # Run the bridge steps manually to capture intermediate output
        result = self.bridge.run_complete_bridge([observation])
        
        # Extract detailed content
        for obj_name, obj in result['mathematical_objects'].items():
            mathematical_objects[obj_name] = {
                'name': obj_name,
                'heuristic_form': obj.heuristic_form,
                'numerical_value': obj.numerical_value,
                'analytic_form': obj.analytic_form,
                'rigorous_form': obj.rigorous_form,
                'formal_form': obj.formal_form,
                'verification_status': {level.value: status for level, status in obj.verification_status.items()}
            }
        
        for comp in result['proof_components']:
            proof_components.append({
                'component_type': comp.component_type.value,
                'statement': comp.statement,
                'proof': comp.proof,
                'dependencies': comp.dependencies,
                'verification': comp.verification,
                'rigor_level': comp.rigor_level.value
            })
        
        detailed_result = {
            'observation': observation,
            'mathematical_objects': mathematical_objects,
            'proof_components': proof_components,
            'bridge_steps': result['bridge_steps'],
            'bridge_complete': result['bridge_complete']
        }
        
        self.detailed_output.append(detailed_result)
        
        # Print the actual mathematical content
        print(f"\nMATHEMATICAL OBJECTS CREATED:")
        for obj_name, obj_data in mathematical_objects.items():
            print(f"\n--- {obj_name.upper()} ---")
            print(f"Heuristic: {obj_data['heuristic_form']}")
            print(f"Numerical: {obj_data['numerical_value']}")
            print(f"Analytic: {obj_data['analytic_form']}")
            print(f"Rigorous: {obj_data['rigorous_form'][:200]}...")
            print(f"Formal: {obj_data['formal_form'][:200]}...")
        
        print(f"\nPROOF COMPONENTS CREATED:")
        for i, comp in enumerate(proof_components):
            print(f"\n--- COMPONENT {i+1}: {comp['component_type'].upper()} ---")
            print(f"Statement: {comp['statement']}")
            print(f"Proof steps: {len(comp['proof'])}")
            for j, step in enumerate(comp['proof'][:3]):  # Show first 3 steps
                print(f"  {j+1}. {step}")
            if len(comp['proof']) > 3:
                print(f"  ... and {len(comp['proof']) - 3} more steps")
        
        return detailed_result
    
    def process_all_rh_observations(self) -> List[Dict[str, Any]]:
        """Process all RH observations and capture detailed output."""
        
        # Key RH observations to examine in detail
        key_observations = [
            "Archimedean term A_∞(φ_t) dominates prime sum terms for sufficiently large t",
            "Block positivity follows from archimedean dominance over prime contributions", 
            "Positive semidefinite kernel implies zeros on critical line",
            "Convergent series A_∞(φ_t) = (1/2) ∑_{n≥1} (1/n²) ∫_0^∞ φ_t''(y) e^{-2ny} dy",
            "Archimedean lower bound: A_∞(φ_t) ≥ C_A · t^{-1/2} for t ≥ t_0"
        ]
        
        print("CAPTURING DETAILED MATHEMATICAL OUTPUT")
        print("=" * 60)
        
        all_results = []
        for observation in key_observations:
            result = self.capture_observation_processing(observation)
            all_results.append(result)
        
        return all_results
    
    def save_detailed_output(self, filename: str = "detailed_mathematical_output.out"):
        """Save all detailed output to file."""
        
        with open(filename, 'w') as f:
            f.write("DETAILED MATHEMATICAL OUTPUT FROM BRIDGE ALGORITHM\n")
            f.write("=" * 60 + "\n\n")
            
            for i, result in enumerate(self.detailed_output):
                f.write(f"OBSERVATION {i+1}: {result['observation']}\n")
                f.write("-" * 60 + "\n\n")
                
                f.write("MATHEMATICAL OBJECTS:\n")
                for obj_name, obj_data in result['mathematical_objects'].items():
                    f.write(f"\n{obj_name.upper()}:\n")
                    f.write(f"  Heuristic: {obj_data['heuristic_form']}\n")
                    f.write(f"  Numerical: {obj_data['numerical_value']}\n")
                    f.write(f"  Analytic: {obj_data['analytic_form']}\n")
                    f.write(f"  Rigorous:\n{obj_data['rigorous_form']}\n")
                    f.write(f"  Formal:\n{obj_data['formal_form']}\n")
                
                f.write("\nPROOF COMPONENTS:\n")
                for j, comp in enumerate(result['proof_components']):
                    f.write(f"\nComponent {j+1} ({comp['component_type']}):\n")
                    f.write(f"  Statement: {comp['statement']}\n")
                    f.write(f"  Proof:\n")
                    for k, step in enumerate(comp['proof']):
                        f.write(f"    {k+1}. {step}\n")
                
                f.write("\n" + "=" * 60 + "\n\n")
        
        print(f"Detailed output saved to {filename}")

def main():
    """Capture and save detailed mathematical output."""
    capture = DetailedOutputCapture()
    
    # Process key observations
    results = capture.process_all_rh_observations()
    
    # Save to file
    capture.save_detailed_output()
    
    print(f"\nCaptured detailed output for {len(results)} observations")
    print("Check 'detailed_mathematical_output.out' for the actual mathematical content!")
    
    return results

if __name__ == "__main__":
    results = main()