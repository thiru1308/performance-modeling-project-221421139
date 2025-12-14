# Supermarket Checkout Queue Performance Modeling

## Project Overview
This repository contains my mini-project submission for the **EEX5362 - Performance Modeling** course at The Open University of Sri Lanka. The project analyzes a supermarket checkout queue system using discrete-event simulation to identify performance bottlenecks and propose optimizations.

## Repository Contents

### **Code Files:**
- `supermarket_simulation.py` - Main Python simulation code
- `supermarket_checkout_data.csv` - Generated dataset used for analysis

### **Reports:**
- `EEX5362_mp_221421139.docx` - Final mini-project report (main submission)
- `mini_project_deliverable_01_221421139.docx` - Initial Deliverable 01

### **Visualization Outputs:**
- `waiting_time_comparison.png` - Chart 1: Average waiting time comparison
- `utilization_comparison.png` - Chart 2: Counter utilization by scenario
- `queue_length_trend.png` - Chart 3: Queue length over time
- `utilization_comparison_detailed.png` - Chart 4: Theoretical vs actual utilization
- `throughput_comparison.png` - Chart 5: System throughput comparison

## How to Run the Simulation

### **Prerequisites**
Make sure you have Python 3.7 or higher installed. You'll also need these Python libraries:

```bash
pip install simpy numpy pandas matplotlib

Clone or download this repository to your local machine

Open a terminal/command prompt in the project folder

Run the simulation:

python supermarket_simulation.py
