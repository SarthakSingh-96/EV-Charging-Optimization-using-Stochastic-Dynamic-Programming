"""
Main Execution Script for EV Charging Optimization Project
AMS 553 Final Project

This script orchestrates the entire project pipeline:
1. Data preprocessing
2. SDP training
3. Policy comparison
4. Visualization generation
5. Report generation

Run this script to execute the complete project.
"""

import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime

# Import project modules
from data_processing import DataProcessor, preprocess_data_for_simulation
from ev_charging_sdp import EVParameters, StochasticPriceModel, TripDemandModel, EVChargingMDP
from baseline_policies import get_all_baseline_policies, SDPPolicy
from simulation import EVSimulator, SimulationResult
from visualization import create_all_visualizations


def print_header(text: str):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")


def main():
    """Main execution function"""
    
    print_header("EV CHARGING OPTIMIZATION WITH STOCHASTIC DYNAMIC PROGRAMMING")
    print(f"AMS 553 Final Project")
    print(f"Execution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Configuration
    EV_DATA_PATH = "ev_charging_dataset.csv"
    OUTPUT_DIR = "results"
    FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
    MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # ========================================================================
    # STEP 1: DATA PREPROCESSING
    # ========================================================================
    print_header("STEP 1: DATA PREPROCESSING")
    
    processor = DataProcessor(EV_DATA_PATH)
    ev_data = processor.load_data()
    processor.explore_data()
    
    # Extract patterns
    hourly_prices, price_std, estimated_prices = processor.extract_price_patterns()
    energy_per_trip, trip_freq = processor.analyze_trip_patterns()
    params_dict = processor.get_typical_battery_params()
    
    # Create data visualizations
    print("\nGenerating data overview plots...")
    processor.plot_data_overview(save_path=os.path.join(FIGURES_DIR, "data_overview.png"))
    
    # ========================================================================
    # STEP 2: INITIALIZE MODELS
    # ========================================================================
    print_header("STEP 2: INITIALIZING MODELS")
    
    # EV parameters
    ev_params = EVParameters(
        battery_capacity=params_dict['battery_capacity'],
        min_soc=0.20,
        max_soc=1.0,
        charging_efficiency=0.92,
        max_charging_rate=params_dict['max_charging_rate'],
        degradation_cost_per_kwh=0.05,
        penalty_cost_per_kwh=10.0,
        soc_states=21,
        price_states=10,
        hour_states=24,
        action_states=11
    )
    
    print(f"EV Parameters:")
    print(f"  Battery Capacity: {ev_params.battery_capacity:.2f} kWh")
    print(f"  Max Charging Rate: {ev_params.max_charging_rate:.2f} kW")
    print(f"  Min SoC: {ev_params.min_soc*100:.0f}%")
    print(f"  State Space: {ev_params.soc_states} × {ev_params.price_states} × {ev_params.hour_states}")
    print(f"  Action Space: {ev_params.action_states} discrete actions")
    
    # Price model
    price_model = StochasticPriceModel(seed=42)
    print(f"\nPrice Model initialized with time-of-use structure")
    
    # Trip demand model
    trip_model = TripDemandModel(ev_data, seed=42)
    print(f"Trip Demand Model initialized from dataset")
    
    # ========================================================================
    # STEP 3: TRAIN SDP POLICY
    # ========================================================================
    print_header("STEP 3: TRAINING SDP POLICY (VALUE ITERATION)")
    
    mdp = EVChargingMDP(ev_params, price_model, trip_model)
    
    print("Starting value iteration...")
    print("This may take several minutes...\n")
    
    mdp.value_iteration(
        n_iterations=100,
        gamma=0.95,
        n_samples=50,
        verbose=True
    )
    
    # Save policy
    policy_path = os.path.join(MODELS_DIR, "sdp_policy.pkl")
    mdp.save_policy(policy_path)
    
    # ========================================================================
    # STEP 4: INITIALIZE ALL POLICIES
    # ========================================================================
    print_header("STEP 4: INITIALIZING BASELINE POLICIES")
    
    # Get baseline policies
    baseline_policies = get_all_baseline_policies(
        battery_capacity=ev_params.battery_capacity,
        max_charging_rate=ev_params.max_charging_rate
    )
    
    # Add SDP policy
    baseline_policies['sdp'] = SDPPolicy(mdp)
    
    print("Initialized policies:")
    for name, policy in baseline_policies.items():
        print(f"  - {policy.get_name()}")
    
    # ========================================================================
    # STEP 5: SIMULATE AND COMPARE POLICIES
    # ========================================================================
    print_header("STEP 5: POLICY SIMULATION AND COMPARISON")
    
    simulator = EVSimulator(
        battery_capacity=ev_params.battery_capacity,
        min_soc=ev_params.min_soc,
        max_soc=ev_params.max_soc,
        charging_efficiency=ev_params.charging_efficiency,
        degradation_cost_per_kwh=ev_params.degradation_cost_per_kwh,
        penalty_cost_per_kwh=ev_params.penalty_cost_per_kwh
    )
    
    # Compare policies over multiple trials
    print("\nRunning Monte Carlo simulations...")
    print(f"Number of trials: 10")
    print(f"Days per trial: 365")
    print(f"Total hours simulated: {10 * 365 * 24:,}\n")
    
    comparison_df = simulator.compare_policies(
        policies=baseline_policies,
        price_model=price_model,
        trip_model=trip_model,
        n_days=365,
        n_trials=10,
        initial_soc=0.80
    )
    
    # Save comparison results
    comparison_csv_path = os.path.join(OUTPUT_DIR, "policy_comparison.csv")
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"\nComparison results saved to: {comparison_csv_path}")
    
    # Print summary
    simulator.print_comparison_summary(comparison_df)
    
    # ========================================================================
    # STEP 6: DETAILED SIMULATION FOR VISUALIZATION
    # ========================================================================
    print_header("STEP 6: DETAILED SIMULATION FOR VISUALIZATION")
    
    print("Running detailed simulations for each policy...")
    detailed_results = []
    
    for name, policy in baseline_policies.items():
        print(f"  Simulating {policy.get_name()}...")
        result = simulator.simulate_policy(
            policy=policy,
            price_model=price_model,
            trip_model=trip_model,
            n_days=30,  # 30 days for detailed visualization
            initial_soc=0.80,
            seed=42
        )
        detailed_results.append(result)
    
    # ========================================================================
    # STEP 7: GENERATE VISUALIZATIONS
    # ========================================================================
    print_header("STEP 7: GENERATING VISUALIZATIONS")
    
    create_all_visualizations(
        results=detailed_results,
        comparison_df=comparison_df,
        mdp=mdp,
        output_dir=FIGURES_DIR
    )
    
    # ========================================================================
    # STEP 8: GENERATE SUMMARY REPORT
    # ========================================================================
    print_header("STEP 8: GENERATING SUMMARY REPORT")
    
    generate_summary_report(
        comparison_df=comparison_df,
        ev_params=ev_params,
        output_path=os.path.join(OUTPUT_DIR, "summary_report.txt")
    )
    
    # ========================================================================
    # COMPLETION
    # ========================================================================
    print_header("PROJECT EXECUTION COMPLETED")
    
    print(f"All results saved to: {OUTPUT_DIR}/")
    print(f"  - Figures: {FIGURES_DIR}/")
    print(f"  - Models: {MODELS_DIR}/")
    print(f"  - Data: {comparison_csv_path}")
    print(f"  - Report: {os.path.join(OUTPUT_DIR, 'summary_report.txt')}")
    
    print(f"\nExecution completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "=" * 80)


def generate_summary_report(comparison_df: pd.DataFrame, 
                           ev_params: EVParameters,
                           output_path: str):
    """Generate text summary report"""
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EV CHARGING OPTIMIZATION - SUMMARY REPORT\n")
        f.write("AMS 553 Final Project\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # System parameters
        f.write("SYSTEM PARAMETERS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Battery Capacity: {ev_params.battery_capacity:.2f} kWh\n")
        f.write(f"Minimum SoC: {ev_params.min_soc*100:.0f}%\n")
        f.write(f"Maximum SoC: {ev_params.max_soc*100:.0f}%\n")
        f.write(f"Charging Efficiency: {ev_params.charging_efficiency*100:.0f}%\n")
        f.write(f"Max Charging Rate: {ev_params.max_charging_rate:.2f} kW\n")
        f.write(f"Degradation Cost: ${ev_params.degradation_cost_per_kwh:.3f}/kWh\n")
        f.write(f"Penalty Cost: ${ev_params.penalty_cost_per_kwh:.2f}/kWh\n\n")
        
        # Policy comparison
        f.write("POLICY COMPARISON RESULTS\n")
        f.write("-" * 80 + "\n\n")
        
        summary = comparison_df.groupby('policy_name').agg({
            'total_cost': ['mean', 'std', 'min', 'max'],
            'electricity_cost': ['mean'],
            'degradation_cost': ['mean'],
            'penalty_cost': ['mean'],
            'avg_soc': ['mean'],
            'min_soc': ['mean', 'min'],
            'num_violations': ['mean', 'sum'],
            'num_charges': ['mean'],
            'total_energy_charged': ['mean']
        }).round(2)
        
        f.write(str(summary))
        f.write("\n\n")
        
        # Best policy
        best_policy = comparison_df.groupby('policy_name')['total_cost'].mean().idxmin()
        best_cost = comparison_df.groupby('policy_name')['total_cost'].mean().min()
        
        f.write("OPTIMAL POLICY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Best Policy: {best_policy}\n")
        f.write(f"Average Total Cost: ${best_cost:.2f}\n\n")
        
        # Cost savings vs baseline
        baseline_cost = comparison_df[comparison_df['policy_name'] == 'Myopic Policy']['total_cost'].mean()
        savings = baseline_cost - best_cost
        savings_pct = (savings / baseline_cost) * 100
        
        f.write(f"Cost Savings vs Myopic Policy:\n")
        f.write(f"  Absolute: ${savings:.2f}\n")
        f.write(f"  Percentage: {savings_pct:.2f}%\n\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"Summary report saved to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
