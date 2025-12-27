# AMS 553 Final Project Proposal
## Optimal Electric Vehicle (EV) Charging Strategy under Price Uncertainty

**Team Members:** [Your Name(s) Here]  
**Date:** November 20, 2024

---

## Project Overview

This project investigates optimal charging strategies for electric vehicles (EVs) under stochastic electricity prices and uncertain trip demands using **Simulation-Based Stochastic Dynamic Programming (SDP)**. We aim to minimize total operational costs while maintaining sufficient battery charge for daily trips, considering battery degradation and renewable energy availability.

## Problem Statement

EV owners face multiple uncertainties:
- **Stochastic electricity prices** varying hourly based on grid demand
- **Random trip demands** depleting battery charge unpredictably  
- **Battery degradation costs** from charging/discharging cycles
- **Renewable energy availability** affecting price and environmental impact

The challenge is determining **when** and **how much** to charge to minimize costs while ensuring operational reliability.

## Methodology

### 1. **Stochastic Dynamic Programming Formulation**
- **State variables:** Battery State of Charge (SoC), hour of day, price level
- **Decision variables:** Charging amount (kWh)
- **State transitions:** Battery dynamics with degradation
- **Objective:** Minimize expected total cost (electricity + degradation + penalty for insufficient charge)

### 2. **Data Sources**
- **EV Charging Dataset** (Kaggle): 64,947 records with battery capacity, SoC, energy consumption, charging rates, and environmental factors
- **Electricity Price Data:** Extracted from real-world patterns in the dataset or synthetic generation based on hourly distributions

### 3. **Policy Comparison**
We will implement and compare three charging policies:
1. **Myopic Policy:** Charge only when SoC is critically low (greedy)
2. **Fixed Schedule Policy:** Charge at predetermined times (e.g., midnight)
3. **SDP-Optimized Policy:** Learned optimal policy via value iteration

### 4. **Simulation Framework**
- Monte Carlo simulation over 365 days
- Stochastic price generation using ARIMA/Markov models
- Random trip demand sampling from dataset distributions
- Performance metrics: total cost, charging frequency, battery health, SoC violations

## Expected Deliverables

1. **Code Implementation:**
   - SDP solver with value iteration
   - Policy evaluation via Monte Carlo simulation
   - Data preprocessing and stochastic model generation

2. **Visualizations:**
   - Battery SoC vs. time for different policies
   - Electricity price heatmaps and policy response
   - Cost comparison across policies
   - Policy action heatmaps (SoC × Price)

3. **Analysis:**
   - Statistical comparison of policy performance
   - Sensitivity analysis on price volatility and trip uncertainty
   - Discussion on practical implementation

## Workload Distribution

- **Data Processing & Price Modeling** (25%): Extract and model stochastic price/demand from datasets
- **SDP Implementation** (35%): Develop value iteration algorithm and optimal policy
- **Baseline Policies & Simulation** (25%): Implement comparison policies and Monte Carlo framework
- **Visualization & Report** (15%): Generate figures, analyze results, write final report

## Timeline

- **Week 1-2:** Data exploration, preprocessing, and stochastic model fitting
- **Week 3:** SDP formulation and value iteration implementation  
- **Week 4:** Policy comparison and Monte Carlo simulation
- **Week 5:** Results analysis, visualization, and report writing

## Relevance to Course Topics

This project directly addresses **Topic 5: Simulation-based approaches for stochastic dynamic programming**, combining:
- Stochastic process modeling (electricity prices, trip demands)
- Optimization under uncertainty (SDP with value iteration)
- Monte Carlo simulation for policy evaluation
- Real-world application with public datasets

---

**Signature:** ___________________  
**Date:** November 20, 2024
