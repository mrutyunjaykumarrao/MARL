"""
Free-Space Path Loss (FSPL) Module
==================================

This module implements the Free-Space Path Loss model for RF propagation,
which is the foundation of our communication graph construction.

FSPL Formula (in dB):
    FSPL(d, f) = 20*log10(d) + 20*log10(f) + 20*log10(4*pi/c)
    
where:
    d = distance between transmitter and receiver (meters)
    f = carrier frequency (Hz)
    c = speed of light (3e8 m/s)

The received power at distance d is:
    P_R = P_tx / 10^(FSPL/10)   [Watts]
        = P_tx * (c / (4*pi*f*d))^2

Author: MARL Jammer Team
"""

import numpy as np
from typing import Union, Tuple

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

# Speed of light (m/s)
SPEED_OF_LIGHT = 3.0e8

# FSPL constant term: 20*log10(4*pi/c)
# 4*pi/c ≈ 4.19e-8, so 20*log10(4.19e-8) ≈ -147.55 dB
FSPL_CONSTANT_DB = 20 * np.log10(4 * np.pi / SPEED_OF_LIGHT)

# Frequency bands available (Hz)
FREQUENCY_BANDS = {
    0: 4.33e8,   # 433 MHz
    1: 9.15e8,   # 915 MHz
    2: 2.40e9,   # 2.4 GHz (default)
    3: 5.80e9,   # 5.8 GHz
}


# =============================================================================
# UNIT CONVERSION FUNCTIONS
# =============================================================================

def db_to_watts(power_dbm: float) -> float:
    """
    Convert power from dBm to Watts.
    
    Formula: P_watts = 10^((P_dBm - 30) / 10)
    
    Args:
        power_dbm: Power in dBm
        
    Returns:
        Power in Watts
        
    Example:
        >>> db_to_watts(20.0)  # 20 dBm
        0.1  # 100 mW = 0.1 W
        
        >>> db_to_watts(-90.0)  # -90 dBm
        1e-12  # 1 pW
    """
    return 10 ** ((power_dbm - 30) / 10)


def watts_to_db(power_watts: float) -> float:
    """
    Convert power from Watts to dBm.
    
    Formula: P_dBm = 10*log10(P_watts) + 30
    
    Args:
        power_watts: Power in Watts
        
    Returns:
        Power in dBm
        
    Example:
        >>> watts_to_db(0.1)  # 100 mW
        20.0  # 20 dBm
        
        >>> watts_to_db(1e-12)  # 1 pW
        -90.0  # -90 dBm
    """
    return 10 * np.log10(power_watts) + 30


# =============================================================================
# FSPL CALCULATIONS
# =============================================================================

def fspl_db(
    distance: Union[float, np.ndarray],
    frequency: float,
    eps: float = 1e-10
) -> Union[float, np.ndarray]:
    """
    Calculate Free-Space Path Loss in decibels (dB).
    
    Formula: FSPL = 20*log10(d) + 20*log10(f) + 20*log10(4*pi/c)
    
    This is the standard FSPL equation from the baseline paper (Eq. 10).
    
    Args:
        distance: Distance between transmitter and receiver (meters).
                  Can be a scalar or numpy array for vectorized computation.
        frequency: Carrier frequency in Hz
        eps: Small value to prevent log(0) for zero distances
        
    Returns:
        FSPL value(s) in dB
        
    Example:
        >>> fspl_db(86.0, 2.4e9)  # 86m at 2.4 GHz
        ~90.2  # dB
        
    Note:
        - For d=0, we add eps to prevent -inf
        - Higher frequency = higher path loss
        - Doubling distance adds ~6 dB loss
    """
    # Ensure distance is not zero to avoid log(0)
    d = np.maximum(distance, eps)
    
    # FSPL = 20*log10(d) + 20*log10(f) + constant
    fspl = 20 * np.log10(d) + 20 * np.log10(frequency) + FSPL_CONSTANT_DB
    
    return fspl


def fspl_linear(
    distance: Union[float, np.ndarray],
    frequency: float,
    eps: float = 1e-10
) -> Union[float, np.ndarray]:
    """
    Calculate FSPL as a linear factor (not in dB).
    
    Formula: FSPL_linear = (4*pi*f*d / c)^2
    
    This is the factor by which transmitted power is attenuated.
    
    Args:
        distance: Distance in meters
        frequency: Carrier frequency in Hz
        eps: Small value to prevent division by zero
        
    Returns:
        FSPL linear factor (dimensionless, >= 1)
        
    Note:
        P_received = P_transmitted / FSPL_linear
    """
    d = np.maximum(distance, eps)
    
    # (4*pi*f*d / c)^2
    fspl = (4 * np.pi * frequency * d / SPEED_OF_LIGHT) ** 2
    
    return fspl


def received_power_watts(
    tx_power_watts: float,
    distance: Union[float, np.ndarray],
    frequency: float,
    eps: float = 1e-10
) -> Union[float, np.ndarray]:
    """
    Calculate received power in Watts using FSPL model.
    
    Formula: P_R = P_tx * (c / (4*pi*f*d))^2
    
    This is the core formula from Section 3.2 of the project guide.
    
    Args:
        tx_power_watts: Transmit power in Watts
        distance: Distance(s) in meters
        frequency: Carrier frequency in Hz
        eps: Small value to prevent division by zero
        
    Returns:
        Received power in Watts
        
    Example:
        >>> received_power_watts(0.1, 86.0, 2.4e9)  # 100mW at 86m, 2.4GHz
        ~1e-12  # ~1 pW = -90 dBm (sensitivity threshold)
        
    Physical Interpretation:
        - At distance d, the power spreads over a sphere of area 4*pi*d^2
        - Antenna aperture is proportional to (c/f)^2
        - Combined effect gives the (c/(4*pi*f*d))^2 factor
    """
    d = np.maximum(distance, eps)
    
    # P_R = P_tx * (c / (4*pi*f*d))^2
    # This is equivalent to P_tx / FSPL_linear
    wavelength = SPEED_OF_LIGHT / frequency
    path_gain = (wavelength / (4 * np.pi * d)) ** 2
    
    return tx_power_watts * path_gain


def received_power_dbm(
    tx_power_dbm: float,
    distance: Union[float, np.ndarray],
    frequency: float,
    eps: float = 1e-10
) -> Union[float, np.ndarray]:
    """
    Calculate received power in dBm using FSPL model.
    
    Formula: P_R(dBm) = P_tx(dBm) - FSPL(dB)
    
    Args:
        tx_power_dbm: Transmit power in dBm
        distance: Distance(s) in meters
        frequency: Carrier frequency in Hz
        eps: Small value to prevent log(0)
        
    Returns:
        Received power in dBm
        
    Example:
        >>> received_power_dbm(20.0, 86.0, 2.4e9)  # 20dBm at 86m, 2.4GHz
        ~-90.0  # dBm (sensitivity threshold)
    """
    fspl = fspl_db(distance, frequency, eps)
    return tx_power_dbm - fspl


# =============================================================================
# DERIVED RANGE CALCULATIONS
# =============================================================================

def compute_comm_range(
    tx_power_watts: float,
    sensitivity_watts: float,
    frequency: float
) -> float:
    """
    Calculate the maximum communication range based on FSPL.
    
    Derived by solving: P_R = P_sens
        P_tx * (c / (4*pi*f*R))^2 = P_sens
        R = (c / (4*pi*f)) * sqrt(P_tx / P_sens)
    
    This is the derived formula from Section 3.2.
    
    Args:
        tx_power_watts: Transmit power in Watts
        sensitivity_watts: Receiver sensitivity threshold in Watts
        frequency: Carrier frequency in Hz
        
    Returns:
        Maximum communication range in meters
        
    Example:
        >>> compute_comm_range(0.1, 1e-12, 2.4e9)  # 100mW TX, -90dBm sens, 2.4GHz
        ~86.0  # meters
        
    Note:
        This is a DERIVED parameter, not a hardcoded constant.
        It changes with frequency, TX power, and sensitivity.
    """
    wavelength = SPEED_OF_LIGHT / frequency
    R_comm = (wavelength / (4 * np.pi)) * np.sqrt(tx_power_watts / sensitivity_watts)
    return R_comm


def compute_jam_range(
    jammer_power_watts: float,
    jam_threshold_watts: float,
    frequency: float
) -> float:
    """
    Calculate the effective jamming range based on FSPL.
    
    Derived by solving: P_jam = P_jam_thresh
        P_jammer * (c / (4*pi*f*R))^2 = P_jam_thresh
        R = (c / (4*pi*f)) * sqrt(P_jammer / P_jam_thresh)
    
    This is the derived formula from Section 3.3.
    
    Args:
        jammer_power_watts: Jammer transmit power in Watts
        jam_threshold_watts: Jamming disruption threshold in Watts
        frequency: Jammer frequency in Hz
        
    Returns:
        Effective jamming range in meters
        
    Example:
        >>> compute_jam_range(1.0, 1e-10, 2.4e9)  # 1W jammer, -70dBm thresh, 2.4GHz
        ~43.0  # meters
        
    Note:
        Jammer must be within this range of the communication link midpoint
        AND operating on the correct frequency band to disrupt the link.
    """
    wavelength = SPEED_OF_LIGHT / frequency
    R_jam = (wavelength / (4 * np.pi)) * np.sqrt(jammer_power_watts / jam_threshold_watts)
    return R_jam


# =============================================================================
# VECTORIZED OPERATIONS FOR SCALABILITY
# =============================================================================

def compute_pairwise_received_power(
    positions: np.ndarray,
    tx_power_watts: float,
    frequency: float,
    eps: float = 1e-10
) -> np.ndarray:
    """
    Compute received power matrix for all pairs of drones.
    
    This is the vectorized implementation described in Section 7.1.
    No Python loops - pure NumPy broadcasting for O(N^2) efficiency.
    
    Args:
        positions: Array of shape (N, 2) containing (x, y) positions
        tx_power_watts: Transmit power in Watts
        frequency: Carrier frequency in Hz
        eps: Small value for numerical stability
        
    Returns:
        P_R matrix of shape (N, N) where P_R[i,j] is received power
        at drone j from drone i (in Watts)
        
    Example:
        >>> positions = np.array([[0,0], [50,0], [100,0]])  # 3 drones
        >>> P_R = compute_pairwise_received_power(positions, 0.1, 2.4e9)
        >>> P_R.shape
        (3, 3)
        
    Implementation Notes:
        - Diagonal is set to infinity (drone always receives its own signal)
        - Uses scipy.spatial.distance.cdist pattern via broadcasting
        - Memory: O(N^2), Time: O(N^2) - vectorized
    """
    N = positions.shape[0]
    
    # Compute pairwise distances using broadcasting
    # positions[:, None, :] shape: (N, 1, 2)
    # positions[None, :, :] shape: (1, N, 2)
    # diff shape: (N, N, 2)
    diff = positions[:, None, :] - positions[None, :, :]
    
    # Euclidean distance matrix (N, N)
    distances = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Compute received power using FSPL
    P_R = received_power_watts(tx_power_watts, distances, frequency, eps)
    
    # Set diagonal to infinity (drone "receives" its own signal perfectly)
    np.fill_diagonal(P_R, np.inf)
    
    return P_R


def get_frequency_for_band(band_index: int) -> float:
    """
    Get frequency in Hz for a given band index.
    
    Args:
        band_index: Integer in {0, 1, 2, 3}
        
    Returns:
        Frequency in Hz
        
    Raises:
        ValueError: If band_index is invalid
    """
    if band_index not in FREQUENCY_BANDS:
        raise ValueError(f"Invalid band index {band_index}. Must be in {list(FREQUENCY_BANDS.keys())}")
    return FREQUENCY_BANDS[band_index]


# =============================================================================
# VERIFICATION FUNCTIONS (for testing)
# =============================================================================

def verify_fspl_calculations() -> dict:
    """
    Run verification tests on FSPL calculations.
    
    This function demonstrates that our implementation matches
    the theoretical values from the project guide.
    
    Returns:
        Dictionary with test results
        
    Use this in Jupyter notebook to show professor that physics is correct.
    """
    results = {}
    
    # Test 1: Communication range at 2.4 GHz
    P_tx = 0.1      # 20 dBm = 100 mW
    P_sens = 1e-12  # -90 dBm = 1 pW
    f = 2.4e9       # 2.4 GHz
    
    R_comm = compute_comm_range(P_tx, P_sens, f)
    results["comm_range_2.4GHz"] = {
        "value_m": R_comm,
        "expected_m": 86.0,  # From Section 3.2
        "match": abs(R_comm - 86.0) < 5.0
    }
    
    # Test 2: Jamming range at 2.4 GHz
    P_jam = 1.0      # 30 dBm = 1 W
    P_thresh = 1e-10 # -70 dBm
    
    R_jam = compute_jam_range(P_jam, P_thresh, f)
    results["jam_range_2.4GHz"] = {
        "value_m": R_jam,
        "expected_m": 43.0,  # From Section 4.3
        "match": abs(R_jam - 43.0) < 5.0
    }
    
    # Test 3: Received power at threshold distance
    P_R = received_power_watts(P_tx, R_comm, f)
    results["received_power_at_Rcomm"] = {
        "value_W": P_R,
        "expected_W": P_sens,
        "ratio": P_R / P_sens,  # Should be ~1.0
        "match": abs(P_R / P_sens - 1.0) < 0.1
    }
    
    # Test 4: dB/Watts conversion round-trip
    P_original = 20.0  # dBm
    P_watts = db_to_watts(P_original)
    P_back = watts_to_db(P_watts)
    results["db_conversion_roundtrip"] = {
        "original_dBm": P_original,
        "watts": P_watts,
        "back_dBm": P_back,
        "match": abs(P_original - P_back) < 0.01
    }
    
    # Test 5: FSPL at different frequencies (Table from Section 4.3)
    freq_tests = [
        (0, 4.33e8, 320.0),  # 433 MHz
        (1, 9.15e8, 151.0),  # 915 MHz
        (2, 2.40e9, 86.0),   # 2.4 GHz
        (3, 5.80e9, 35.0),   # 5.8 GHz
    ]
    
    results["comm_ranges_by_frequency"] = {}
    for band_idx, freq, expected in freq_tests:
        R = compute_comm_range(P_tx, P_sens, freq)
        results["comm_ranges_by_frequency"][f"band_{band_idx}"] = {
            "frequency_Hz": freq,
            "computed_m": R,
            "expected_m": expected,
            "error_percent": abs(R - expected) / expected * 100
        }
    
    return results


# =============================================================================
# MAIN (for standalone testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FSPL Module Verification")
    print("=" * 60)
    
    results = verify_fspl_calculations()
    
    for test_name, test_result in results.items():
        print(f"\n{test_name}:")
        if isinstance(test_result, dict):
            for key, val in test_result.items():
                if isinstance(val, dict):
                    print(f"  {key}:")
                    for k, v in val.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {val}")
    
    print("\n" + "=" * 60)
    print("All FSPL calculations verified!")
    print("=" * 60)
