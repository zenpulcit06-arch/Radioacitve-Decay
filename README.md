# Radioactive Decay Simulation with Detector Effects

## Overview
This project presents an end-to-end simulation and analysis of radioactive decay, combining:
- Analytical solutions
- Numerical integration
- Monte Carlo methods
- Detector response modeling
- Statistical parameter estimation

The goal is to **recover the decay constant (λ)** from noisy detector data, mimicking a realistic particle physics measurement.

This project is designed to reflect techniques commonly used in **experimental high-energy and nuclear physics**, such as those at **CERN**.

---

## Physics Background
Radioactive decay follows the exponential law:

\[
N(t) = N_0 e^{-\lambda t}
\]

where:
- \(N_0\) is the initial number of nuclei  
- \(\lambda\) is the decay constant  
- \(t\) is time  

The half-life is given by:
\[
T_{1/2} = \frac{\ln 2}{\lambda}
\]

---

## Project Structure

### 1. Numerical Decay Simulation
- Solves the decay equation using **Euler integration**
- Compares numerical results with the **analytical solution**
- Validates correctness of the simulation

### 2. Monte Carlo Decay Simulation
- Simulates individual nuclei as stochastic processes
- Each nucleus decays with probability \( \lambda \Delta t \)
- Demonstrates statistical fluctuations and Poisson errors
- Confirms convergence to the analytical decay law

### 3. Detector Modeling
A realistic detector model is introduced:
- **Finite efficiency** (binomial detection)
- **Background noise** (Poisson-distributed)
- Measured counts differ from true decay rate

This step mimics real experimental limitations.

### 4. Parameter Estimation
- Background subtraction and efficiency correction
- Log-linearization of the decay law
- **Weighted least-squares fitting**
- Extraction of:
  - Decay constant \( \lambda \)
  - Statistical uncertainty on \( \lambda \)

The recovered value is compared against the true input parameter using a **z-score consistency test**.

---

## Key Features
- Monte Carlo simulation of radioactive decay
- Detector response with efficiency and background
- Proper statistical error propagation (√N)
- Weighted regression with covariance estimation
- Clear separation between physics, simulation, and analysis

---

## Technologies Used
- Python
- NumPy
- Matplotlib
- Pandas

---

## How to Run
Clone the repository and run:

```bash
python radioactive_decay.py
```

The script will:

Simulate decay

Generate Monte Carlo data

Apply detector effects

Fit the decay constant

Display plots and numerical results

## Example Output

Monte Carlo decay curves with Poisson errors

Detector count distributions

Log(counts) vs time with fitted decay constant

Agreement between fitted and true λ within statistical uncertainty

## Possible Extensions

Poisson likelihood fitting (unbinned)

Detector dead-time modeling

Time resolution smearing

Bayesian parameter estimation

Comparison of χ² vs likelihood methods

## Motivation

This project was developed as preparation for experimental physics research, with a focus on:

Statistical data analysis

Monte Carlo methods

Detector-level effects

It is directly relevant to research environments such as CERN.

## Author

Pulkit Jain
BS–MS Physics, BITS Pilani