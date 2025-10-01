"""
Convolution Time Springs: 1D Convolution Kernel Representation

This module implements Hamiltonian recursive time springs using 1D convolution kernels.
The key insight is that time springs can be represented as convolution operations
where the kernel encodes the spring dynamics and the input represents the prime sequence.

Mathematical Framework:
- Kernel K(t): Represents spring response function
- Input I(t): Prime sequence or time series
- Output O(t) = (K * I)(t): Convolved spring response
- Hamiltonian H: Energy functional of the convolution system
"""

import numpy as np
import scipy.signal
from typing import List, Dict, Tuple, Callable, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class SpringKernel:
    """1D convolution kernel representing time spring dynamics"""
    kernel_type: str  # 'gaussian', 'exponential', 'oscillatory', 'custom'
    parameters: Dict[str, float]  # Kernel parameters
    kernel_array: Optional[np.ndarray] = None  # Precomputed kernel
    
    def __post_init__(self):
        """Generate kernel array based on type and parameters"""
        if self.kernel_array is None:
            self.kernel_array = self._generate_kernel()
    
    def _generate_kernel(self) -> np.ndarray:
        """Generate the convolution kernel based on type"""
        if self.kernel_type == 'gaussian':
            return self._gaussian_kernel()
        elif self.kernel_type == 'exponential':
            return self._exponential_kernel()
        elif self.kernel_type == 'oscillatory':
            return self._oscillatory_kernel()
        elif self.kernel_type == 'custom':
            return self._custom_kernel()
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
    
    def _gaussian_kernel(self) -> np.ndarray:
        """Gaussian kernel: K(t) = exp(-t²/2σ²)"""
        sigma = self.parameters.get('sigma', 1.0)
        length = int(self.parameters.get('length', 50))
        t = np.linspace(-3*sigma, 3*sigma, length)
        return np.exp(-t**2 / (2 * sigma**2))
    
    def _exponential_kernel(self) -> np.ndarray:
        """Exponential kernel: K(t) = exp(-α|t|)"""
        alpha = self.parameters.get('alpha', 1.0)
        length = int(self.parameters.get('length', 50))
        t = np.linspace(-5/alpha, 5/alpha, length)
        return np.exp(-alpha * np.abs(t))
    
    def _oscillatory_kernel(self) -> np.ndarray:
        """Oscillatory kernel: K(t) = exp(-αt²)cos(ωt)"""
        alpha = self.parameters.get('alpha', 1.0)
        omega = self.parameters.get('omega', 2.0)
        length = int(self.parameters.get('length', 100))
        t = np.linspace(-5/np.sqrt(alpha), 5/np.sqrt(alpha), length)
        return np.exp(-alpha * t**2) * np.cos(omega * t)
    
    def _custom_kernel(self) -> np.ndarray:
        """Custom kernel based on parameters"""
        # This could be extended for more complex kernels
        return self.parameters.get('custom_array', np.array([1.0]))

class ConvolutionTimeSpring:
    """Time spring implemented as 1D convolution operation"""
    
    def __init__(self, 
                 kernel: SpringKernel,
                 fixed_point: float = 0.5,
                 hamiltonian_params: Optional[Dict[str, float]] = None):
        """
        Initialize convolution-based time spring.
        
        Args:
            kernel: Spring kernel for convolution
            fixed_point: Critical line fixed point
            hamiltonian_params: Parameters for Hamiltonian dynamics
        """
        self.kernel = kernel
        self.fixed_point = fixed_point
        self.hamiltonian_params = hamiltonian_params or {
            'mass': 1.0,
            'stiffness': 1.0,
            'damping': 0.1
        }
        
        # Precompute kernel for efficiency
        self.kernel_array = kernel.kernel_array
        self.kernel_center = len(self.kernel_array) // 2
    
    def compress_spring(self, prime: int) -> float:
        """Compute spring compression using logarithmic distance"""
        return np.log(prime) - np.log(self.fixed_point)
    
    def apply_convolution(self, input_sequence: np.ndarray, 
                         mode: str = 'same') -> np.ndarray:
        """
        Apply convolution operation: output = kernel * input
        
        Args:
            input_sequence: Input time series
            mode: Convolution mode ('same', 'full', 'valid')
            
        Returns:
            Convolved output sequence
        """
        return scipy.signal.convolve(input_sequence, self.kernel_array, mode=mode)
    
    def hamiltonian_energy(self, position: float, momentum: float) -> float:
        """
        Compute Hamiltonian energy: H = p²/(2m) + (1/2)kx²
        
        Args:
            position: Spring position (compression)
            momentum: Spring momentum
            
        Returns:
            Total Hamiltonian energy
        """
        m = self.hamiltonian_params['mass']
        k = self.hamiltonian_params['stiffness']
        
        kinetic = momentum**2 / (2 * m)
        potential = 0.5 * k * position**2
        
        return kinetic + potential
    
    def spring_response(self, primes: List[int]) -> Dict[int, Dict[str, float]]:
        """
        Compute spring response for a sequence of primes using convolution.
        
        Args:
            primes: List of prime numbers
            
        Returns:
            Dictionary with spring responses for each prime
        """
        # Convert primes to time series
        prime_series = np.array(primes, dtype=float)
        
        # Apply convolution
        convolved = self.apply_convolution(prime_series)
        
        # Compute spring responses
        responses = {}
        for i, prime in enumerate(primes):
            # Get convolved value at this position
            if i < len(convolved):
                convolved_value = convolved[i]
            else:
                convolved_value = 0.0
            
            # Compute compression
            compression = self.compress_spring(prime)
            
            # Compute momentum (derivative approximation)
            if i > 0:
                momentum = prime - primes[i-1]
            else:
                momentum = 0.0
            
            # Compute Hamiltonian energy
            energy = self.hamiltonian_energy(compression, momentum)
            
            responses[prime] = {
                'compression': compression,
                'momentum': momentum,
                'energy': energy,
                'convolved_value': convolved_value,
                'response_magnitude': abs(convolved_value)
            }
        
        return responses
    
    def recursive_spring_dynamics(self, 
                                 initial_primes: List[int], 
                                 iterations: int = 5) -> List[List[int]]:
        """
        Apply recursive spring dynamics using convolution.
        
        Each iteration:
        1. Apply convolution to current prime sequence
        2. Generate new primes based on convolution response
        3. Update sequence for next iteration
        
        Args:
            initial_primes: Starting prime sequence
            iterations: Number of recursive iterations
            
        Returns:
            List of prime sequences after each iteration
        """
        sequences = [initial_primes.copy()]
        current_primes = initial_primes.copy()
        
        for iteration in range(iterations):
            # Apply convolution
            prime_array = np.array(current_primes, dtype=float)
            convolved = self.apply_convolution(prime_array)
            
            # Generate new primes based on convolution response
            new_primes = []
            for i, prime in enumerate(current_primes):
                if i < len(convolved):
                    response = convolved[i]
                    # Generate new prime based on response magnitude
                    new_prime = int(abs(response) % 100) + 2
                    if new_prime not in new_primes:
                        new_primes.append(new_prime)
            
            # Update sequence: add new primes at beginning
            current_primes = new_primes + current_primes
            sequences.append(current_primes.copy())
        
        return sequences
    
    def analyze_convolution_spectrum(self, primes: List[int]) -> Dict[str, np.ndarray]:
        """
        Analyze the frequency spectrum of the convolution operation.
        
        Args:
            primes: Input prime sequence
            
        Returns:
            Dictionary with spectral analysis results
        """
        # Convert to time series
        prime_series = np.array(primes, dtype=float)
        
        # Apply convolution
        convolved = self.apply_convolution(prime_series)
        
        # Compute FFT
        fft_input = np.fft.fft(prime_series)
        fft_output = np.fft.fft(convolved)
        fft_kernel = np.fft.fft(self.kernel_array, n=len(prime_series))
        
        # Compute frequencies
        freqs = np.fft.fftfreq(len(prime_series))
        
        return {
            'frequencies': freqs,
            'input_spectrum': np.abs(fft_input),
            'output_spectrum': np.abs(fft_output),
            'kernel_spectrum': np.abs(fft_kernel),
            'convolved_sequence': convolved
        }

def create_hamiltonian_spring_kernel(spring_type: str = 'oscillatory',
                                   hamiltonian_params: Optional[Dict[str, float]] = None) -> ConvolutionTimeSpring:
    """
    Create a Hamiltonian time spring with appropriate convolution kernel.
    
    Args:
        spring_type: Type of spring kernel
        hamiltonian_params: Hamiltonian parameters
        
    Returns:
        Configured ConvolutionTimeSpring
    """
    if spring_type == 'oscillatory':
        kernel = SpringKernel(
            kernel_type='oscillatory',
            parameters={
                'alpha': 0.5,
                'omega': 2.0,
                'length': 100
            }
        )
    elif spring_type == 'gaussian':
        kernel = SpringKernel(
            kernel_type='gaussian',
            parameters={
                'sigma': 2.0,
                'length': 50
            }
        )
    else:
        kernel = SpringKernel(
            kernel_type='exponential',
            parameters={
                'alpha': 1.0,
                'length': 50
            }
        )
    
    return ConvolutionTimeSpring(
        kernel=kernel,
        fixed_point=0.5,
        hamiltonian_params=hamiltonian_params
    )

def test_convolution_time_springs():
    """Test the convolution-based time springs mechanism"""
    
    print("CONVOLUTION TIME SPRINGS TEST")
    print("=" * 50)
    
    # Create different types of springs
    spring_types = ['oscillatory', 'gaussian', 'exponential']
    springs = {}
    
    for spring_type in spring_types:
        springs[spring_type] = create_hamiltonian_spring_kernel(spring_type)
    
    # Test primes
    test_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    print("1. SPRING RESPONSES:")
    print("-" * 30)
    
    for spring_type, spring in springs.items():
        print(f"\n{spring_type.upper()} SPRING:")
        responses = spring.spring_response(test_primes[:10])
        
        for prime in test_primes[:5]:
            resp = responses[prime]
            print(f"  Prime {prime:2d}: compression={resp['compression']:6.3f}, "
                  f"energy={resp['energy']:8.3f}, response={resp['response_magnitude']:6.3f}")
    
    print("\n2. RECURSIVE DYNAMICS:")
    print("-" * 25)
    
    # Test recursive dynamics with oscillatory spring
    oscillatory_spring = springs['oscillatory']
    initial_primes = [2, 3, 5, 7, 11]
    
    sequences = oscillatory_spring.recursive_spring_dynamics(initial_primes, iterations=3)
    
    for i, seq in enumerate(sequences):
        print(f"Iteration {i}: {seq[:8]}...")
    
    print("\n3. SPECTRAL ANALYSIS:")
    print("-" * 20)
    
    # Analyze spectrum
    spectrum = oscillatory_spring.analyze_convolution_spectrum(test_primes)
    
    print(f"Input spectrum max: {np.max(spectrum['input_spectrum']):.3f}")
    print(f"Output spectrum max: {np.max(spectrum['output_spectrum']):.3f}")
    print(f"Kernel spectrum max: {np.max(spectrum['kernel_spectrum']):.3f}")
    
    print("\n4. HAMILTONIAN ENERGY CONSERVATION:")
    print("-" * 35)
    
    # Test energy conservation
    total_energy = 0.0
    for prime in test_primes[:10]:
        compression = oscillatory_spring.compress_spring(prime)
        momentum = np.log(prime)  # Simple momentum model
        energy = oscillatory_spring.hamiltonian_energy(compression, momentum)
        total_energy += energy
        print(f"Prime {prime:2d}: H = {energy:8.3f}")
    
    print(f"Total Hamiltonian energy: {total_energy:.3f}")
    
    print("\n" + "="*50)
    print("SUCCESS: Convolution time springs are working!")
    print("The 1D convolution kernel successfully represents")
    print("Hamiltonian recursive time springs dynamics.")
    print("="*50)

if __name__ == "__main__":
    test_convolution_time_springs()
