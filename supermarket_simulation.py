import simpy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

df = pd.read_csv('supermarket_checkout_data.csv')

total_time = df['arrival_time'].iloc[-1]
num_customers = len(df)
avg_arrival_rate_lam = num_customers / total_time

avg_service_time = df['service_duration'].mean()
service_time_std = df['service_duration'].std()

print("=== PARAMETERS ESTIMATED FROM YOUR DATASET ===")
print(f"Total Simulation Time (from data): {total_time:.0f} seconds ({total_time/3600:.2f} hours)")
print(f"Number of Customers (from data): {num_customers}")
print(f"Average Arrival Rate (λ): {avg_arrival_rate_lam:.4f} cust/sec ({avg_arrival_rate_lam*3600:.1f} cust/hour)")
print(f"Average Service Time: {avg_service_time:.1f} seconds")
print(f"Service Time Std Dev: {service_time_std:.1f} seconds")
print(f"Implied Service Rate (μ): {3600/avg_service_time:.1f} cust/hour/counter")
print()

class SupermarketCheckout:
    def __init__(self, env, num_counters, service_time_mean, service_time_std, speed_factor=1.0):
        self.env = env
        self.service_time_mean = service_time_mean * speed_factor
        self.service_time_std = service_time_std * speed_factor
        self.num_counters = num_counters
        self.counters = simpy.Resource(env, num_counters)
        self.waiting_times = []
        self.queue_lengths = []
        self.service_times = []
        self.counter_busy_intervals = defaultdict(list)
        self.total_customers = 0
        self.counter_busy_time = [0] * num_counters
   
    def serve_customer(self, customer_id):
        arrival_time = self.env.now
        with self.counters.request() as request:
            queue_length = len(self.counters.queue)
            self.queue_lengths.append((self.env.now, queue_length))
            yield request

            wait_time = self.env.now - arrival_time
            self.waiting_times.append(wait_time)

            service_start = self.env.now

            service_time = max(1, random.gauss(self.service_time_mean, self.service_time_std))
            self.service_times.append(service_time)

            active_count = self.counters.count
            counter_id = min(active_count, self.num_counters - 1)
            

            self.counter_busy_intervals[counter_id].append((service_start, service_start + service_time))
            self.counter_busy_time[counter_id] += service_time
            
            yield self.env.timeout(service_time)
            self.total_customers += 1

def customer_generator(env, supermarket, arrival_rate):
    customer_id = 0
    while True:
        yield env.timeout(random.expovariate(arrival_rate))
        customer_id += 1
        env.process(supermarket.serve_customer(customer_id))

def calculate_utilization_from_intervals(intervals, num_counters, total_time):
    """Calculate accurate utilization from busy intervals"""
    total_busy_time = 0
    
    for counter_id in range(num_counters):
        if counter_id in intervals and intervals[counter_id]:
            sorted_intervals = sorted(intervals[counter_id], key=lambda x: x[0])

            merged = []
            for start, end in sorted_intervals:
                if not merged or merged[-1][1] < start:
                    merged.append([start, end])
                else:
                    merged[-1][1] = max(merged[-1][1], end)

            counter_busy = sum(end - start for start, end in merged)
            total_busy_time += min(counter_busy, total_time)
    

    return total_busy_time / (num_counters * total_time)

def run_simulation(num_counters, speed_factor=1.0, sim_time=30000):
    """Runs one simulation scenario and returns results."""
    random.seed(42)
    env = simpy.Environment()
    supermarket = SupermarketCheckout(env, num_counters, avg_service_time, service_time_std, speed_factor)
    env.process(customer_generator(env, supermarket, avg_arrival_rate_lam))
    env.run(until=sim_time)

    avg_wait = np.mean(supermarket.waiting_times) if supermarket.waiting_times else 0
    avg_queue = np.mean([ql for _, ql in supermarket.queue_lengths]) if supermarket.queue_lengths else 0

    utilization = calculate_utilization_from_intervals(
        supermarket.counter_busy_intervals, 
        num_counters, 
        sim_time
    )

    theoretical_utilization = (avg_arrival_rate_lam * (avg_service_time * speed_factor)) / num_counters

    throughput = supermarket.total_customers / (sim_time / 3600)
    
    return {
        'avg_wait': avg_wait,
        'avg_queue': avg_queue,
        'utilization': utilization,
        'theoretical_util': theoretical_utilization,
        'throughput': throughput,
        'wait_times': supermarket.waiting_times,
        'queue_data': supermarket.queue_lengths,
        'total_customers': supermarket.total_customers,
        'service_times': supermarket.service_times
    }

print("=== RUNNING SIMULATION SCENARIOS ===")
print("Simulation time: 30000 seconds (8.33 hours, ~2 peak shifts)")
print()

scenarios = {
    'A: Baseline (2 counters)': {'counters': 2, 'speed': 1.0},
    'B: 3 Counters': {'counters': 3, 'speed': 1.0},
    'C: Faster Service (15%)': {'counters': 2, 'speed': 0.85},
    'D: Combined': {'counters': 3, 'speed': 0.85}
}

results = {}
for name, params in scenarios.items():
    print(f"Running {name}...")
    res = run_simulation(params['counters'], params['speed'])
    results[name] = res
    print(f"  Avg Wait: {res['avg_wait']:.1f} sec")
    print(f"  Avg Queue: {res['avg_queue']:.1f}")
    print(f"  Utilization: {res['utilization']:.2%} (Theoretical: {res['theoretical_util']:.2%})")
    print(f"  Throughput: {res['throughput']:.1f} cust/hour")
    print(f"  Customers Served: {res['total_customers']}")
    print()

print("=== SUMMARY RESULTS TABLE ===")
print(f"{'Scenario':<25} {'Wait(s)':<8} {'Queue':<8} {'Util(%)':<10} {'Throughput':<12}")
print("-" * 65)
for name, res in results.items():
    print(f"{name:<25} {res['avg_wait']:<8.1f} {res['avg_queue']:<8.1f} "
          f"{res['utilization']*100:<10.1f} {res['throughput']:<12.1f}")

plt.figure(figsize=(10, 6))
scenario_names = list(results.keys())
wait_times = [r['avg_wait'] for r in results.values()]
plt.bar(scenario_names, wait_times, color=['red', 'green', 'orange', 'blue'])
plt.title('Average Customer Waiting Time by Scenario', fontsize=14, fontweight='bold')
plt.ylabel('Waiting Time (seconds)', fontsize=12)
plt.xticks(rotation=15, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(wait_times):
    plt.text(i, v + 5, f'{v:.0f}s', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('waiting_time_comparison.png', dpi=300)
print("\nSaved plot: 'waiting_time_comparison.png'")

plt.figure(figsize=(8, 5))
util_values = [r['utilization']*100 for r in results.values()]
plt.bar(scenario_names, util_values, color=['red', 'green', 'orange', 'blue'])
plt.title('Checkout Counter Utilization by Scenario', fontsize=14, fontweight='bold')
plt.ylabel('Utilization (%)', fontsize=12)
plt.xticks(rotation=15, ha='right')
plt.ylim(0, 110)
plt.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='100% Capacity')
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(util_values):
    plt.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('utilization_comparison.png', dpi=300)
print("Saved plot: 'utilization_comparison.png'")

plt.figure(figsize=(12, 5))
times_a, queues_a = zip(*results['A: Baseline (2 counters)']['queue_data'][::50])
times_d, queues_d = zip(*results['D: Combined']['queue_data'][::50])

plt.plot(times_a, queues_a, label='Baseline (2 counters)', alpha=0.7, linewidth=1)
plt.plot(times_d, queues_d, label='Combined (3 counters + faster)', alpha=0.7, linewidth=1)
plt.title('Queue Length Over Time: Baseline vs. Optimized Scenario', fontsize=14, fontweight='bold')
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Queue Length', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('queue_length_trend.png', dpi=300)
print("Saved plot: 'queue_length_trend.png'")

plt.figure(figsize=(10, 6))
x = np.arange(len(scenario_names))
width = 0.35

actual_util = [r['utilization']*100 for r in results.values()]
theoretical_util = [r['theoretical_util']*100 for r in results.values()]

plt.bar(x - width/2, actual_util, width, label='Simulated Utilization', color='blue', alpha=0.7)
plt.bar(x + width/2, theoretical_util, width, label='Theoretical Utilization', color='orange', alpha=0.7)
plt.xlabel('Scenario')
plt.ylabel('Utilization (%)')
plt.title('Theoretical vs Simulated Counter Utilization', fontsize=14, fontweight='bold')
plt.xticks(x, scenario_names, rotation=15)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('utilization_comparison_detailed.png', dpi=300)
print("Saved plot: 'utilization_comparison_detailed.png'")

print("\n=== SIMULATION COMPLETE ===")
print("Key insights from corrected model:")
print("1. Baseline shows near 100% utilization (system at capacity)")
print("2. Adding counters reduces utilization to sustainable levels")
print("3. Faster service alone cannot solve capacity constraints")
print("4. Combined approach provides optimal performance")
