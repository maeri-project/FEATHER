#!/bin/bash

# Arrays to store the metrics
utilizations=()
pj_computes=()
cycles=()
config_names=()

# Function to extract metrics from timeloop output
extract_metrics() {
    local output="$1"
    local config_name="$2"
    
    # Extract utilization
    local utilization=$(echo "$output" | grep -o "Utilization = [0-9.]*" | grep -o "[0-9.]*")
    
    # Extract pJ/Compute - more robust parsing to handle varying whitespace
    local pj_compute=$(echo "$output" | grep -o "pJ/Compute = [ ]*[0-9.]*" | sed 's/pJ\/Compute = [ ]*//')
    
    # Extract Cycles
    local cycle=$(echo "$output" | grep -o "Cycles = [0-9]*" | grep -o "[0-9]*")
    
    # Store in arrays
    utilizations+=("$utilization")
    pj_computes+=("$pj_compute")
    cycles+=("$cycle")
    config_names+=("$config_name")
    
    echo "Config: $config_name"
    echo "  Utilization: $utilization"
    echo "  pJ/Compute: $pj_compute"
    echo "  Cycles: $cycle"
    echo "---"
}

echo "Running MXU tests and collecting metrics..."
echo "=========================================="

# Test 1: mxu_like with M16384_N128_K124
echo "Running Test 1: mxu_like + M16384_N128_K124..."
output1=$(source ~/.setup.sh && timeloop-model /home/ubuntu/FEATHER/LayoutLoop/configurations/arch_designs/tpuv4_like/mxu/mxu_like.yaml /home/ubuntu/FEATHER/LayoutLoop/configurations/layer_shapes/gemm/gemm_M16384_N128_K124.yaml /home/ubuntu/FEATHER/LayoutLoop/prerun_results/PROVE_TPU_MXU/WS_mapping_gemm_M16384_N128_K124.yaml 2>&1)
extract_metrics "$output1" "mxu_like_M16384_N128_K124"

# Test 2: mxu_like with M16384_N31_K1
echo "Running Test 2: mxu_like + M16384_N31_K1..."
output2=$(source ~/.setup.sh && timeloop-model /home/ubuntu/FEATHER/LayoutLoop/configurations/arch_designs/tpuv4_like/mxu/mxu_like.yaml /home/ubuntu/FEATHER/LayoutLoop/configurations/layer_shapes/gemm/gemm_M16384_N31_K1.yaml /home/ubuntu/FEATHER/LayoutLoop/prerun_results/PROVE_TPU_MXU/WS_mapping_gemm_M16384_N31_K1.yaml 2>&1)
extract_metrics "$output2" "mxu_like_M16384_N31_K1"

# Test 3: mxu_like with M8192_N128_K124
echo "Running Test 3: mxu_like + M8192_N128_K124..."
output3=$(source ~/.setup.sh && timeloop-model /home/ubuntu/FEATHER/LayoutLoop/configurations/arch_designs/tpuv4_like/mxu/mxu_like.yaml /home/ubuntu/FEATHER/LayoutLoop/configurations/layer_shapes/gemm/gemm_M8192_N128_K124.yaml /home/ubuntu/FEATHER/LayoutLoop/prerun_results/PROVE_TPU_MXU/WS_mapping_gemm_M8192_N128_K124.yaml 2>&1)
extract_metrics "$output3" "mxu_like_M8192_N128_K124"

# Test 4: mxu_like with M8192_N31_K1
echo "Running Test 4: mxu_like + M8192_N31_K1..."
output4=$(source ~/.setup.sh && timeloop-model /home/ubuntu/FEATHER/LayoutLoop/configurations/arch_designs/tpuv4_like/mxu/mxu_like.yaml /home/ubuntu/FEATHER/LayoutLoop/configurations/layer_shapes/gemm/gemm_M8192_N31_K1.yaml /home/ubuntu/FEATHER/LayoutLoop/prerun_results/PROVE_TPU_MXU/WS_mapping_gemm_M8192_N31_K1.yaml 2>&1)
extract_metrics "$output4" "mxu_like_M8192_N31_K1"

# Test 5: mxu_inf_off_chip with M16384_N128_K124
echo "Running Test 5: mxu_inf_off_chip + M16384_N128_K124..."
output5=$(source ~/.setup.sh && timeloop-model /home/ubuntu/FEATHER/LayoutLoop/configurations/arch_designs/tpuv4_like/mxu/mxu_inf_off_chip.yaml /home/ubuntu/FEATHER/LayoutLoop/configurations/layer_shapes/gemm/gemm_M16384_N128_K124.yaml /home/ubuntu/FEATHER/LayoutLoop/prerun_results/PROVE_TPU_MXU/WS_mapping_gemm_M16384_N128_K124.yaml 2>&1)
extract_metrics "$output5" "mxu_inf_off_chip_M16384_N128_K124"

# Test 6: mxu_inf_off_chip with M16384_N31_K1
echo "Running Test 6: mxu_inf_off_chip + M16384_N31_K1..."
output6=$(source ~/.setup.sh && timeloop-model /home/ubuntu/FEATHER/LayoutLoop/configurations/arch_designs/tpuv4_like/mxu/mxu_inf_off_chip.yaml /home/ubuntu/FEATHER/LayoutLoop/configurations/layer_shapes/gemm/gemm_M16384_N31_K1.yaml /home/ubuntu/FEATHER/LayoutLoop/prerun_results/PROVE_TPU_MXU/WS_mapping_gemm_M16384_N31_K1.yaml 2>&1)
extract_metrics "$output6" "mxu_inf_off_chip_M16384_N31_K1"

# Test 7: mxu_inf_off_chip with M8192_N128_K124
echo "Running Test 7: mxu_inf_off_chip + M8192_N128_K124..."
output7=$(source ~/.setup.sh && timeloop-model /home/ubuntu/FEATHER/LayoutLoop/configurations/arch_designs/tpuv4_like/mxu/mxu_inf_off_chip.yaml /home/ubuntu/FEATHER/LayoutLoop/configurations/layer_shapes/gemm/gemm_M8192_N128_K124.yaml /home/ubuntu/FEATHER/LayoutLoop/prerun_results/PROVE_TPU_MXU/WS_mapping_gemm_M8192_N128_K124.yaml 2>&1)
extract_metrics "$output7" "mxu_inf_off_chip_M8192_N128_K124"

# Test 8: mxu_inf_off_chip with M8192_N31_K1
echo "Running Test 8: mxu_inf_off_chip + M8192_N31_K1..."
output8=$(source ~/.setup.sh && timeloop-model /home/ubuntu/FEATHER/LayoutLoop/configurations/arch_designs/tpuv4_like/mxu/mxu_inf_off_chip.yaml /home/ubuntu/FEATHER/LayoutLoop/configurations/layer_shapes/gemm/gemm_M8192_N31_K1.yaml /home/ubuntu/FEATHER/LayoutLoop/prerun_results/PROVE_TPU_MXU/WS_mapping_gemm_M8192_N31_K1.yaml 2>&1)
extract_metrics "$output8" "mxu_inf_off_chip_M8192_N31_K1"

echo ""
echo "=========================================="
echo "SUMMARY OF ALL METRICS:"
echo "=========================================="

# Print summary table
printf "%-35s %-15s %-15s %-10s\n" "Configuration" "Utilization" "pJ/Compute" "Cycles"
echo "----------------------------------------------------------------------------"

for i in "${!config_names[@]}"; do
    printf "%-35s %-15s %-15s %-10s\n" "${config_names[$i]}" "${utilizations[$i]}" "${pj_computes[$i]}" "${cycles[$i]}"
done

echo ""
echo "=========================================="
echo "METRICS ANALYSIS:"
echo "=========================================="

# Find best utilization (highest)
best_utilization_idx=0
for i in "${!utilizations[@]}"; do
    # Use awk for floating point comparison
    if [ "$(echo "${utilizations[$i]} ${utilizations[$best_utilization_idx]}" | awk '{if ($1 > $2) print "true"; else print "false"}')" = "true" ]; then
        best_utilization_idx=$i
    fi
done

# Find best pJ/Compute (lowest)
best_pj_compute_idx=0
for i in "${!pj_computes[@]}"; do
    # Use awk for floating point comparison
    if [ "$(echo "${pj_computes[$i]} ${pj_computes[$best_pj_compute_idx]}" | awk '{if ($1 < $2) print "true"; else print "false"}')" = "true" ]; then
        best_pj_compute_idx=$i
    fi
done

# Find best cycles (lowest)
best_cycles_idx=0
for i in "${!cycles[@]}"; do
    if [ "${cycles[$i]}" -lt "${cycles[$best_cycles_idx]}" ]; then
        best_cycles_idx=$i
    fi
done

echo "Best Utilization: ${utilizations[$best_utilization_idx]} (${config_names[$best_utilization_idx]})"
echo "Best pJ/Compute: ${pj_computes[$best_pj_compute_idx]} (${config_names[$best_pj_compute_idx]})"
echo "Best Cycles: ${cycles[$best_cycles_idx]} (${config_names[$best_cycles_idx]})"

# Export arrays for potential use in other scripts
export utilizations
export pj_computes
export cycles
export config_names