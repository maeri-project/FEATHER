#!/usr/bin/env python3
"""JAX Matrix Multiplication Benchmark with Optimal Sharding Selection

This script tests three sharding strategies for each GEMM configuration:
1. Shard M: A is sharded on M dimension, B is replicated
2. Shard N: A is replicated, B is sharded on N dimension
3. Shard MN: Both M and N dimensions are sharded across the mesh

The optimal sharding is selected based on profiled latency.
"""

import os
import sys

# Add the CROSS/jaxite_word directory to path for imports
sys.path.insert(0, '/home/jianming_gatech/CROSS/jaxite_word')

import jax
import jax.numpy as jnp
from jax import random
import pandas as pd
import numpy as np

# Import profiler utilities
from profiler import KernelWrapper, Profiler, collect_logs
import util

# JAX configuration
jax.config.update("jax_enable_x64", True)

# Check device
print(f"JAX devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")
print(f"Number of devices: {jax.device_count()}")

# Define matrix shapes from the CSV file
# Format: (Category, Operation, M, K, N)
SHAPES = [
    # FHE Bootstrapping BConv
    ("FHE", "Bootstrapping BConv", 65536, 40, 88),
    ("FHE", "Bootstrapping BConv", 65536, 40, 92),
    ("FHE", "Bootstrapping BConv", 65536, 40, 84),
    ("FHE", "Bootstrapping BConv", 65536, 40, 120),
    ("FHE", "Bootstrapping BConv", 65536, 40, 116),
    ("FHE", "Bootstrapping BConv", 65536, 44, 100),
    ("FHE", "Bootstrapping BConv", 65536, 44, 96),
    ("FHE", "Bootstrapping BConv", 65536, 44, 128),
    ("FHE", "Bootstrapping BConv", 65536, 44, 104),
    ("FHE", "Bootstrapping BConv", 65536, 44, 136),
    ("FHE", "Bootstrapping BConv", 65536, 44, 132),
    ("FHE", "Bootstrapping BConv", 65536, 48, 112),
    ("FHE", "Bootstrapping BConv", 65536, 48, 108),
    ("FHE", "Bootstrapping BConv", 65536, 48, 140),
    ("FHE", "Bootstrapping BConv", 65536, 48, 132),
    ("FHE", "Bootstrapping BConv", 65536, 48, 148),
    ("FHE", "Bootstrapping BConv", 65536, 48, 144),
    ("FHE", "Bootstrapping BConv", 65536, 52, 124),
    ("FHE", "Bootstrapping BConv", 65536, 52, 120),
    ("FHE", "Bootstrapping BConv", 65536, 52, 152),
    ("FHE", "Bootstrapping BConv", 65536, 52, 128),
    ("FHE", "Bootstrapping BConv", 65536, 52, 160),
    ("FHE", "Bootstrapping BConv", 65536, 52, 156),
    ("FHE", "Bootstrapping BConv", 65536, 56, 136),
    ("FHE", "Bootstrapping BConv", 65536, 56, 132),
    ("FHE", "Bootstrapping BConv", 65536, 60, 144),
    ("FHE", "Bootstrapping BConv", 65536, 56, 140),
    ("FHE", "Bootstrapping BConv", 65536, 60, 152),
    ("FHE", "Bootstrapping BConv", 65536, 60, 148),
    ("FHE", "Bootstrapping BConv", 65536, 28, 84),
    ("FHE", "Bootstrapping BConv", 65536, 28, 80),
    ("FHE", "Bootstrapping BConv", 65536, 32, 92),
    ("FHE", "Bootstrapping BConv", 65536, 28, 88),
    ("FHE", "Bootstrapping BConv", 65536, 32, 100),
    ("FHE", "Bootstrapping BConv", 65536, 32, 96),
    ("FHE", "Bootstrapping BConv", 65536, 36, 76),
    ("FHE", "Bootstrapping BConv", 65536, 36, 72),
    ("FHE", "Bootstrapping BConv", 65536, 36, 104),
    ("FHE", "Bootstrapping BConv", 65536, 36, 80),
    ("FHE", "Bootstrapping BConv", 65536, 36, 112),
    ("FHE", "Bootstrapping BConv", 65536, 36, 108),
    # FHE NTT
    ("FHE", "NTT", 64, 1024, 1024),
    ("FHE", "NTT", 64, 2048, 2048),
    ("FHE", "NTT", 128, 2048, 2048),
    ("FHE", "NTT", 128, 4096, 4096),
    ("FHE", "NTT", 256, 4096, 4096),
    # ChatGPT OSS
    ("ChatGPT OSS", "Q", 256, 2880, 4096),
    ("ChatGPT OSS", "fused QKV", 256, 2880, 5120),
    ("ChatGPT OSS", "attn out", 256, 4096, 2880),
    ("ChatGPT OSS", "scores per group", 256, 64, 2048),
]


def _jax_matmul_kernel(A, B):
    """Kernel wrapper entry point for matrix multiplication."""
    return jnp.matmul(A, B)


def create_sharding_configs(mesh, partition_spec):
    """Create different sharding configurations for GEMM.

    For GEMM: A(M, K) @ B(K, N) = C(M, N)

    Sharding strategies:
    1. Shard M: A sharded on dim 0, B replicated, C sharded on dim 0
    2. Shard N: A replicated, B sharded on dim 1, C sharded on dim 1
    3. Shard MN: A sharded on dim 0, B sharded on dim 1, C sharded on both
    """
    axis_names = mesh.axis_names

    # For 2D mesh (x, y), we have 2*4=8 devices
    # Shard M: use all devices for M dimension
    if len(axis_names) > 1:
        m_partition_full = (axis_names[0], axis_names[1])  # Shard M across both mesh axes
        n_partition_full = (axis_names[0], axis_names[1])  # Shard N across both mesh axes
        # For MN sharding, split mesh: M on x, N on y
        m_partition_half = axis_names[0]  # Shard M on x axis only
        n_partition_half = axis_names[1]  # Shard N on y axis only
    else:
        m_partition_full = axis_names[0]
        n_partition_full = axis_names[0]
        m_partition_half = axis_names[0]
        n_partition_half = None

    configs = {}

    # Strategy 1: Shard M only
    # A(M,K): shard M, replicate K -> (partition, None)
    # B(K,N): replicate both -> (None, None)
    # C(M,N): shard M, replicate N -> (partition, None)
    configs['shard_M'] = {
        'A_sharding': jax.sharding.NamedSharding(mesh, partition_spec(m_partition_full, None)),
        'B_sharding': jax.sharding.NamedSharding(mesh, partition_spec(None, None)),
        'C_sharding': jax.sharding.NamedSharding(mesh, partition_spec(m_partition_full, None)),
    }

    # Strategy 2: Shard N only
    # A(M,K): replicate both -> (None, None)
    # B(K,N): replicate K, shard N -> (None, partition)
    # C(M,N): replicate M, shard N -> (None, partition)
    configs['shard_N'] = {
        'A_sharding': jax.sharding.NamedSharding(mesh, partition_spec(None, None)),
        'B_sharding': jax.sharding.NamedSharding(mesh, partition_spec(None, n_partition_full)),
        'C_sharding': jax.sharding.NamedSharding(mesh, partition_spec(None, n_partition_full)),
    }

    # Strategy 3: Shard both M and N (2D sharding)
    # A(M,K): shard M on x axis -> (x, None)
    # B(K,N): shard N on y axis -> (None, y)
    # C(M,N): shard M on x, N on y -> (x, y)
    if len(axis_names) > 1:
        configs['shard_MN'] = {
            'A_sharding': jax.sharding.NamedSharding(mesh, partition_spec(m_partition_half, None)),
            'B_sharding': jax.sharding.NamedSharding(mesh, partition_spec(None, n_partition_half)),
            'C_sharding': jax.sharding.NamedSharding(mesh, partition_spec(m_partition_half, n_partition_half)),
        }
    else:
        # With 1D mesh, MN sharding is same as M sharding
        configs['shard_MN'] = configs['shard_M']

    return configs


def create_kernel_wrapper(kernel_name, M, K, N, mesh, sharding_config):
    """Create a KernelWrapper with specified sharding configuration."""
    input_shape_A = (M, K)
    input_shape_B = (K, N)

    return KernelWrapper(
        kernel_name=kernel_name,
        function_to_wrap=_jax_matmul_kernel,
        input_structs=[
            (input_shape_A, jnp.float32),
            (input_shape_B, jnp.float32)
        ],
        parameters=None,
        mesh=mesh,
        input_shardings=(sharding_config['A_sharding'], sharding_config['B_sharding']),
        output_sharding=sharding_config['C_sharding'],
        enable_sharding=True,
    )


def main():
    output_trace_root = "/home/jianming_gatech/matmul_benchmark_optimal_log"

    # Create output directory
    if not os.path.exists(output_trace_root):
        os.makedirs(output_trace_root)

    print("=" * 80)
    print("JAX Matrix Multiplication Benchmark - Optimal Sharding Selection")
    print("=" * 80)
    print(f"Device: {jax.devices()[0]}")
    print(f"Number of devices: {jax.device_count()}")
    print("=" * 80)

    # Setup sharding
    try:
        mesh, partition_spec = util.create_sharding()
        axis_names = mesh.axis_names
        print(f"Mesh shape: {mesh.shape}")
        print(f"Axis names: {axis_names}")

        # Create sharding configurations
        sharding_configs = create_sharding_configs(mesh, partition_spec)
        print(f"\nSharding strategies to test: {list(sharding_configs.keys())}")

    except RuntimeError as exc:
        print(f"ERROR: Could not setup sharding: {exc}")
        return

    # Profiler configuration
    profiler_config = {
        "iterations": 1,
        "save_to_file": True,
        "enable_sharding": True,
    }

    # Create profiler instance
    profiler_instance = Profiler(
        output_trace_path=output_trace_root,
        profile_naming="matmul_optimal_sharding",
        configuration=profiler_config,
    )

    total_shapes = len(SHAPES)
    total_configs = total_shapes * len(sharding_configs)
    config_count = 0

    print(f"\nTotal shapes: {total_shapes}")
    print(f"Sharding strategies per shape: {len(sharding_configs)}")
    print(f"Total configurations to test: {total_configs}")
    print("=" * 80)

    for idx, (category, operation, M, K, N) in enumerate(SHAPES):
        print(f"\n[Shape {idx+1}/{total_shapes}] {category} - {operation}: ({M}, {K}) x ({K}, {N})")

        for sharding_name, sharding_config in sharding_configs.items():
            config_count += 1

            # Sanitize kernel name
            kernel_name = f"matmul_{category.replace(' ', '_')}_{operation.replace(' ', '_')}_M{M}_K{K}_N{N}_{sharding_name}"

            try:
                kernel_wrapper = create_kernel_wrapper(
                    kernel_name=kernel_name,
                    M=M,
                    K=K,
                    N=N,
                    mesh=mesh,
                    sharding_config=sharding_config,
                )

                profiler_instance.add_profile(
                    name=kernel_name,
                    kernel_wrapper=kernel_wrapper,
                    kernel_setting_cols={
                        "category": category,
                        "operation": operation,
                        "M": M,
                        "K": K,
                        "N": N,
                        "sharding_strategy": sharding_name,
                        "num_devices": jax.device_count(),
                    },
                )
                print(f"   [{config_count}/{total_configs}] Added: {sharding_name}")

            except Exception as e:
                print(f"   [{config_count}/{total_configs}] ERROR ({sharding_name}): {e}")
                continue

    # Run profiling
    print("\n" + "=" * 80)
    print("Starting profiling...")
    print("=" * 80)

    profiler_instance.profile_all_profilers()

    # Post-process and collect results
    print("\n" + "=" * 80)
    print("Post-processing results...")
    print("=" * 80)

    profiler_instance.post_process_all_profilers()

    # Read results and find optimal sharding for each shape
    results_file = os.path.join(output_trace_root, "matmul_optimal_sharding", "matmul_optimal_sharding_results.csv")

    if os.path.exists(results_file):
        # Read raw results
        df = pd.read_csv(results_file, header=None,
                        names=['Kernel_Name', 'Category', 'Operation', 'M', 'K', 'N',
                               'Sharding_Strategy', 'Num_Devices', 'Latency_us'])

        # Save all results
        df['Latency_ms'] = df['Latency_us'] / 1000.0
        all_results_file = '/home/jianming_gatech/jax_matmul_all_sharding_results.csv'
        df.to_csv(all_results_file, index=False)
        print(f"\nAll results saved to: {all_results_file}")

        # Find optimal sharding for each shape
        # Group by (Category, Operation, M, K, N) and find min latency
        shape_cols = ['Category', 'Operation', 'M', 'K', 'N']

        optimal_results = []
        for name, group in df.groupby(shape_cols):
            best_row = group.loc[group['Latency_us'].idxmin()]
            optimal_results.append({
                'Category': best_row['Category'],
                'Operation': best_row['Operation'],
                'M': best_row['M'],
                'K': best_row['K'],
                'N': best_row['N'],
                'Optimal_Sharding': best_row['Sharding_Strategy'],
                'Latency_us': best_row['Latency_us'],
                'Latency_ms': best_row['Latency_ms'],
                'Num_Devices': best_row['Num_Devices'],
            })

        optimal_df = pd.DataFrame(optimal_results)
        optimal_file = '/home/jianming_gatech/jax_matmul_optimal_sharding_results.csv'
        optimal_df.to_csv(optimal_file, index=False)
        print(f"Optimal sharding results saved to: {optimal_file}")

        # Print summary
        print("\n" + "=" * 80)
        print("OPTIMAL SHARDING SUMMARY")
        print("=" * 80)

        # Count optimal sharding strategies
        sharding_counts = optimal_df['Optimal_Sharding'].value_counts()
        print("\nOptimal sharding strategy distribution:")
        for strategy, count in sharding_counts.items():
            print(f"  {strategy}: {count} shapes ({100*count/len(optimal_df):.1f}%)")

        # Statistics by category
        print("\n" + "-" * 40)
        print("Results by Category:")
        print("-" * 40)
        for category in optimal_df['Category'].unique():
            cat_df = optimal_df[optimal_df['Category'] == category]
            print(f"\n{category}:")
            print(f"  Configurations: {len(cat_df)}")
            print(f"  Mean latency: {cat_df['Latency_us'].mean():.2f} µs")
            print(f"  Min latency: {cat_df['Latency_us'].min():.2f} µs")
            print(f"  Max latency: {cat_df['Latency_us'].max():.2f} µs")

            # Show sharding breakdown within category
            cat_sharding = cat_df['Optimal_Sharding'].value_counts()
            print(f"  Optimal sharding breakdown:")
            for s, c in cat_sharding.items():
                print(f"    {s}: {c}")

        # Detailed comparison table
        print("\n" + "=" * 80)
        print("DETAILED COMPARISON (All Sharding Strategies)")
        print("=" * 80)

        # Pivot table: rows = shapes, columns = sharding strategies
        pivot_df = df.pivot_table(
            index=['Category', 'Operation', 'M', 'K', 'N'],
            columns='Sharding_Strategy',
            values='Latency_us',
            aggfunc='first'
        ).reset_index()

        # Add optimal column
        pivot_df['Optimal'] = pivot_df[['shard_M', 'shard_N', 'shard_MN']].idxmin(axis=1)
        pivot_df['Best_Latency_us'] = pivot_df[['shard_M', 'shard_N', 'shard_MN']].min(axis=1)

        comparison_file = '/home/jianming_gatech/jax_matmul_sharding_comparison.csv'
        pivot_df.to_csv(comparison_file, index=False)
        print(f"\nComparison table saved to: {comparison_file}")

        # Print comparison for first few shapes
        print("\nSample comparison (first 10 shapes):")
        print("-" * 100)
        print(f"{'Category':<12} {'Operation':<20} {'M':>6} {'K':>5} {'N':>5} | {'shard_M':>10} {'shard_N':>10} {'shard_MN':>10} | {'Optimal':<10}")
        print("-" * 100)
        for _, row in pivot_df.head(10).iterrows():
            print(f"{str(row['Category']):<12} {str(row['Operation']):<20} {row['M']:>6} {row['K']:>5} {row['N']:>5} | "
                  f"{row.get('shard_M', 'N/A'):>10.2f} {row.get('shard_N', 'N/A'):>10.2f} {row.get('shard_MN', 'N/A'):>10.2f} | "
                  f"{row['Optimal']:<10}")

        print("\n" + "=" * 80)
        print("OPTIMAL RESULTS TABLE")
        print("=" * 80)
        print(optimal_df.to_string(index=False))

    else:
        print(f"Results file not found: {results_file}")
        # List available files
        for root, dirs, files in os.walk(output_trace_root):
            for f in files:
                if f.endswith('.csv'):
                    print(f"Found: {os.path.join(root, f)}")

    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
