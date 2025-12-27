# EV Charging Optimization with Stochastic Dynamic Programming
## AMS 553 Final Project

**Project Title:** Optimal Electric Vehicle (EV) Charging Strategy under Price Uncertainty

---

## 📋 Project Overview

This project implements a **Stochastic Dynamic Programming (SDP)** solution to optimize electric vehicle charging strategies under uncertain electricity prices and trip demands. The goal is to minimize total operational costs while maintaining battery health and ensuring sufficient charge for daily trips.

### Key Features
- ✅ **Stochastic Dynamic Programming** with value iteration
- ✅ **Multiple baseline policies** for comparison
- ✅ **Monte Carlo simulation** for policy evaluation
- ✅ **Real-world EV dataset** (64,947 records from Kaggle)
- ✅ **Comprehensive visualizations** and analysis
- ✅ **Automated report generation**

---

## 📁 Project Structure

```
Simmod/
├── ev_charging_dataset.csv          # Main EV charging dataset
├── location_dataset.csv             # Location dataset (auxiliary)
├── PROJECT_PROPOSAL.md              # Project proposal (1-page)
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
│
├── main.py                          # Main execution script
├── ev_charging_sdp.py              # SDP implementation (core algorithm)
├── baseline_policies.py            # Alternative charging policies
├── simulation.py                   # Monte Carlo simulation framework
├── visualization.py                # Visualization tools
├── data_processing.py              # Data preprocessing module
│
└── results/                        # Output directory (created on run)
    ├── figures/                    # All generated plots
    ├── models/                     # Saved SDP policies
    ├── policy_comparison.csv       # Numerical results
    └── summary_report.txt          # Text summary report
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Complete Project

```bash
python main.py
```

This will execute the entire pipeline:
1. **Data preprocessing** and exploratory analysis
2. **SDP training** using value iteration
3. **Policy simulation** (10 trials × 365 days each)
4. **Visualization generation** (all figures)
5. **Report generation** (summary statistics)

**Expected Runtime:** 15-30 minutes (depending on hardware)

---

## 📊 Components

### 1. **Stochastic Dynamic Programming (`ev_charging_sdp.py`)**

Implements the core MDP formulation:

- **State Space:** (SoC, Price, Hour)
  - SoC: 21 discrete levels (0%, 5%, ..., 100%)
  - Price: 10 discrete levels ($0.05 - $0.50/kWh)
  - Hour: 24 hours of day

- **Action Space:** 11 charging rates (0, 5, 10, ..., 50 kW)

- **Objective:** Minimize expected total cost
  ```
  Cost = Electricity Cost + Battery Degradation + Penalty for Low SoC
  ```

- **Algorithm:** Value Iteration with Monte Carlo sampling

### 2. **Baseline Policies (`baseline_policies.py`)**

Four comparison policies:

1. **Myopic Policy:** Charge only when critically low (greedy)
2. **Fixed Schedule Policy:** Charge at predetermined times (e.g., midnight)
3. **Threshold-Based Policy:** Charge when price < threshold AND SoC < target
4. **Time-of-Use Optimized:** Smart charging based on time periods

### 3. **Simulation Framework (`simulation.py`)**

Monte Carlo simulation engine:
- Simulates EV charging over 365 days
- Evaluates policies across multiple trials
- Tracks costs, SoC violations, charging frequency

### 4. **Visualization (`visualization.py`)**

Generates comprehensive plots:
- Battery SoC vs. time
- Electricity price patterns
- Charging action timelines
- Cost comparisons
- Policy heatmaps
- Performance metrics

### 5. **Data Processing (`data_processing.py`)**

Analyzes the EV dataset:
- Extracts price patterns from charging load
- Analyzes trip demand distributions
- Computes typical battery parameters
- Creates data overview visualizations

---

## 📈 Results

### Expected Outcomes

The SDP-optimized policy typically achieves:
- **15-25% cost reduction** vs. myopic policy
- **Fewer SoC violations** (maintains battery health)
- **Intelligent charging** during low-price periods
- **Adaptive behavior** based on price and SoC

### Key Visualizations

1. **SoC and Price Over Time**
   - Shows battery charge levels for all policies
   - Overlays electricity price patterns
   - Demonstrates policy responsiveness

2. **Policy Heatmaps**
   - Optimal charging decisions for each (SoC, Price) state
   - Shows learned behavior at different hours
   - Reveals strategic charging patterns

3. **Cost Comparison**
   - Breakdown: electricity, degradation, penalty
   - Statistical comparison across trials
   - Identifies best-performing policy

4. **Performance Metrics**
   - Average SoC maintenance
   - Number of violations
   - Charging frequency
   - Total energy consumption

---

## 🧪 Testing Individual Modules

### Test Data Processing
```bash
python data_processing.py
```

### Test Baseline Policies
```bash
python baseline_policies.py
```

### Test SDP Core (Quick Demo)
```bash
python ev_charging_sdp.py
```

---

## 📝 Project Deliverables

### For Submission

1. **Proposal** (✅ Completed)
   - `PROJECT_PROPOSAL.md` (1-page summary)

2. **Presentation** (10-20 minutes)
   - Use figures from `results/figures/`
   - Show cost comparison and policy heatmaps
   - Demonstrate SDP convergence

3. **Final Report** (4+ pages)
   - Introduction and problem formulation
   - Methodology (SDP, value iteration)
   - Results and analysis
   - Conclusions and future work
   - Include all figures from `results/figures/`

### Report Template Structure

```
1. INTRODUCTION
   - Motivation (EV adoption, smart grids)
   - Problem statement (cost optimization under uncertainty)

2. METHODOLOGY
   - MDP Formulation (states, actions, transitions, costs)
   - Value Iteration Algorithm
   - Baseline Policies
   - Monte Carlo Simulation

3. DATASET
   - EV Charging Dataset description
   - Preprocessing steps
   - Stochastic models (price, trip demand)

4. RESULTS
   - Policy comparison (cost, violations, metrics)
   - Visualizations (SoC, prices, actions, heatmaps)
   - Statistical analysis

5. DISCUSSION
   - SDP benefits vs. heuristics
   - Practical implementation considerations
   - Sensitivity analysis

6. CONCLUSION
   - Summary of findings
   - Future work (real-time optimization, V2G integration)
```

---

## 🔧 Configuration

### Modifying Parameters

Edit `main.py` to adjust:

```python
# EV Parameters
ev_params = EVParameters(
    battery_capacity=75.0,           # kWh
    min_soc=0.20,                    # 20%
    max_charging_rate=50.0,          # kW
    degradation_cost_per_kwh=0.05,   # $/kWh
    penalty_cost_per_kwh=10.0,       # $/kWh
    soc_states=21,                   # Discretization
    price_states=10,
    action_states=11
)

# Simulation Settings
n_days = 365          # Days per trial
n_trials = 10         # Number of trials
gamma = 0.95          # Discount factor
n_iterations = 100    # Value iteration iterations
```

---

## 📊 Dataset Information

### EV Charging Dataset (64,947 records)

**Key Features:**
- Vehicle ID, Battery Capacity, State of Charge
- Energy Consumption Rate, Distance to Destination
- Charging Station details, Charging Rate
- Temperature, Weather Conditions, Traffic Data
- Session timing and duration

**Source:** Kaggle - EV Charging Dataset  
**Purpose:** Model stochastic trip demands and validate parameters

---

## 🎯 Course Alignment

### AMS 553 Topics Covered

- ✅ **Topic 5:** Simulation-based approaches for stochastic dynamic programming
- ✅ **Stochastic Processes:** Price uncertainty, trip demand modeling
- ✅ **Optimization:** Value iteration, policy evaluation
- ✅ **Monte Carlo Simulation:** Policy comparison over multiple trials
- ✅ **Real-world Application:** EV charging, smart grids, renewable energy

---

## 📚 References

1. Puterman, M. L. (2014). *Markov Decision Processes: Discrete Stochastic Dynamic Programming*. Wiley.
2. Bertsekas, D. P. (2017). *Dynamic Programming and Optimal Control*. Athena Scientific.
3. EV Charging Dataset: https://www.kaggle.com/datasets (Kaggle)
4. Smart Grid and V2G Literature (IEEE Transactions)

---

## 🤝 Team Members

[Add your name(s) here]

---

## 📧 Contact

For questions about this project:
- Email: [Your email]
- Course: AMS 553 - Simulation and Modeling
- Instructor: Professor Jiaqiao Hu

---

## ⚖️ License

This project is for academic purposes (AMS 553 Final Project).

---

## 🙏 Acknowledgments

- Professor Jiaqiao Hu for course instruction
- Kaggle for the EV Charging Dataset
- Stony Brook University AMS Department

---

**Last Updated:** November 20, 2024
