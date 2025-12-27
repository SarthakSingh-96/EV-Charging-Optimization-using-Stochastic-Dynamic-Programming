"""
Baseline Policies for EV Charging Comparison
AMS 553 Final Project

This module implements alternative charging policies for comparison with SDP.
"""

import numpy as np
from abc import ABC, abstractmethod


class ChargingPolicy(ABC):
    """Abstract base class for charging policies"""
    
    @abstractmethod
    def get_action(self, soc: float, price: float, hour: int, **kwargs) -> float:
        """Get charging action for given state"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get policy name"""
        pass


class MyopicPolicy(ChargingPolicy):
    """
    Myopic (Greedy) Policy: Only charge when SoC is critically low
    Does not consider future prices or demands
    """
    
    def __init__(self, battery_capacity: float = 75.0, 
                 critical_soc: float = 0.30,
                 target_soc: float = 0.80,
                 max_charging_rate: float = 50.0):
        self.battery_capacity = battery_capacity
        self.critical_soc = critical_soc
        self.target_soc = target_soc
        self.max_charging_rate = max_charging_rate
        
    def get_action(self, soc: float, price: float, hour: int, **kwargs) -> float:
        """Charge at max rate if below critical, otherwise don't charge"""
        if soc < self.critical_soc:
            # Calculate how much to charge to reach target
            needed_energy = (self.target_soc - soc) * self.battery_capacity
            # Return charging rate (limited by max rate)
            return min(self.max_charging_rate, needed_energy)
        else:
            return 0.0
    
    def get_name(self) -> str:
        return "Myopic Policy"


class FixedSchedulePolicy(ChargingPolicy):
    """
    Fixed Schedule Policy: Charge at predetermined times regardless of price
    Typically charges at night (off-peak hours)
    """
    
    def __init__(self, battery_capacity: float = 75.0,
                 charging_hours: list = None,
                 target_soc: float = 0.90,
                 max_charging_rate: float = 50.0):
        self.battery_capacity = battery_capacity
        self.charging_hours = charging_hours if charging_hours else [0, 1, 2, 3, 4, 5]
        self.target_soc = target_soc
        self.max_charging_rate = max_charging_rate
        
    def get_action(self, soc: float, price: float, hour: int, **kwargs) -> float:
        """Charge during scheduled hours if below target"""
        if hour in self.charging_hours and soc < self.target_soc:
            # Calculate how much to charge to reach target
            needed_energy = (self.target_soc - soc) * self.battery_capacity
            return min(self.max_charging_rate, needed_energy)
        else:
            return 0.0
    
    def get_name(self) -> str:
        return "Fixed Schedule Policy"


class ThresholdBasedPolicy(ChargingPolicy):
    """
    Threshold-Based Policy: Charge when price is below threshold AND SoC is below target
    Simple heuristic that considers both price and SoC
    """
    
    def __init__(self, battery_capacity: float = 75.0,
                 price_threshold: float = 0.15,
                 soc_threshold: float = 0.70,
                 max_charging_rate: float = 50.0):
        self.battery_capacity = battery_capacity
        self.price_threshold = price_threshold
        self.soc_threshold = soc_threshold
        self.max_charging_rate = max_charging_rate
        
    def get_action(self, soc: float, price: float, hour: int, **kwargs) -> float:
        """Charge if price is low AND SoC is below threshold"""
        if price < self.price_threshold and soc < self.soc_threshold:
            needed_energy = (self.soc_threshold - soc) * self.battery_capacity
            return min(self.max_charging_rate, needed_energy)
        elif soc < 0.20:  # Emergency charging
            needed_energy = (0.40 - soc) * self.battery_capacity
            return min(self.max_charging_rate, needed_energy)
        else:
            return 0.0
    
    def get_name(self) -> str:
        return "Threshold-Based Policy"


class TimeOfUsePolicyOptimized(ChargingPolicy):
    """
    Optimized Time-of-Use Policy: Charge during off-peak hours with intelligent SoC management
    More sophisticated than fixed schedule - adapts based on current SoC
    """
    
    def __init__(self, battery_capacity: float = 75.0,
                 max_charging_rate: float = 50.0):
        self.battery_capacity = battery_capacity
        self.max_charging_rate = max_charging_rate
        
        # Define time periods
        self.super_off_peak = [0, 1, 2, 3, 4, 5]  # Midnight to 6 AM
        self.off_peak = [6, 22, 23]  # 6-7 AM and 10 PM - midnight
        self.peak = [17, 18, 19, 20, 21]  # 5 PM - 9 PM
        
    def get_action(self, soc: float, price: float, hour: int, **kwargs) -> float:
        """Smart charging based on time of day and SoC"""
        
        # Emergency charging at any time if critically low
        if soc < 0.15:
            needed_energy = (0.40 - soc) * self.battery_capacity
            return min(self.max_charging_rate, needed_energy)
        
        # Super off-peak: charge to high level
        if hour in self.super_off_peak:
            if soc < 0.85:
                needed_energy = (0.90 - soc) * self.battery_capacity
                return min(self.max_charging_rate, needed_energy)
        
        # Off-peak: charge to medium level
        elif hour in self.off_peak:
            if soc < 0.65:
                needed_energy = (0.75 - soc) * self.battery_capacity
                return min(self.max_charging_rate, needed_energy)
        
        # Peak hours: only charge if very low
        elif hour in self.peak:
            if soc < 0.25:
                needed_energy = (0.40 - soc) * self.battery_capacity
                return min(self.max_charging_rate, needed_energy * 0.5)  # Reduced rate
        
        # Mid-peak: moderate charging
        else:
            if soc < 0.50:
                needed_energy = (0.65 - soc) * self.battery_capacity
                return min(self.max_charging_rate, needed_energy)
        
        return 0.0
    
    def get_name(self) -> str:
        return "Time-of-Use Optimized Policy"


class SDPPolicy(ChargingPolicy):
    """
    Wrapper for SDP-learned policy
    """
    
    def __init__(self, mdp):
        self.mdp = mdp
        
    def get_action(self, soc: float, price: float, hour: int, **kwargs) -> float:
        """Get action from learned SDP policy"""
        return self.mdp.get_action(soc, price, hour)
    
    def get_name(self) -> str:
        return "SDP-Optimized Policy"


def get_all_baseline_policies(battery_capacity: float = 75.0, 
                              max_charging_rate: float = 50.0) -> dict:
    """
    Get dictionary of all baseline policies for comparison
    
    Returns:
        dict: Dictionary mapping policy names to policy objects
    """
    policies = {
        'myopic': MyopicPolicy(battery_capacity, max_charging_rate=max_charging_rate),
        'fixed_schedule': FixedSchedulePolicy(battery_capacity, max_charging_rate=max_charging_rate),
        'threshold': ThresholdBasedPolicy(battery_capacity, max_charging_rate=max_charging_rate),
        'tou_optimized': TimeOfUsePolicyOptimized(battery_capacity, max_charging_rate=max_charging_rate)
    }
    return policies


if __name__ == "__main__":
    print("Baseline Policies Module")
    print("=" * 50)
    
    # Test policies
    policies = get_all_baseline_policies()
    
    # Test scenario
    test_soc = 0.35
    test_price = 0.12
    test_hour = 2
    
    print(f"\nTest Scenario:")
    print(f"  SoC: {test_soc*100:.1f}%")
    print(f"  Price: ${test_price:.2f}/kWh")
    print(f"  Hour: {test_hour}:00")
    print(f"\nPolicy Actions:")
    
    for name, policy in policies.items():
        action = policy.get_action(test_soc, test_price, test_hour)
        print(f"  {policy.get_name()}: {action:.2f} kW")
