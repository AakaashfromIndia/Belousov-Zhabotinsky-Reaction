# Belousov-Zhabotinsky Reaction Simulation

![Belousov-Zhabotinsky Reaction - Made with Clipchamp (2)](https://github.com/user-attachments/assets/f1f16f64-87ba-4c43-887e-fb4bac03d24e)

## Overview

This project simulates the **Belousov-Zhabotinsky (BZ) reaction** - a classical example of a non-equilibrium chemical oscillator known for producing mesmerizing spatial patterns. The simulation uses reaction-diffusion equations and models complex spiral waves and target patterns observed in laboratory BZ reactions, visualized in a circular "Petri dish."

The interactive dashboard enables users to experiment with reaction rates, diffusion coefficients, and spontaneous perturbations, providing real-time feedback as system dynamics evolve.

## How It Works

- The simulation uses two coupled reaction-diffusion equations to model the concentrations of two interacting chemicals (U and V).
- The finite difference method discretizes the equations on a 2D grid representing the Petri dish.
- Random perturbations initiate pattern formation, and reseeding regularly injects new disturbances within the domain.
- The system supports visualization with a masked circular boundary to mimic a real experiment.
- Real-time plots display chemical statistics as simulation proceeds.

## Controls

- **Parameter sliders** for:
  - q: reaction regime
  - f: feed rate
  - ν: removal rate
  - Du: diffusion of U
  - Dv: diffusion of V
- **Live statistics plots:**
  - Mean U, standard deviation of U, mean V, and range of U over time
- **Reset button** to randomize initial conditions and add fresh perturbations.

## Usage

1. **Clone the repository:**
   ```bash
   git clone 
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy matplotlib scipy
   ```

3. **Run the simulation:**
   ```bash
   python Belousov-Zhabotinsky-reaction.py
   ```

4. **Interact with controls:**
   - Adjust sliders for reaction parameters and diffusion rates.
   - Use the Reset button to inject new patterns and observe their evolution.

## Dependencies

- Python 3.x
- NumPy
- Matplotlib
- SciPy
