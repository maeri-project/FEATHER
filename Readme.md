# Overview

FEATHER: A Reconfigurable Accelerator with Data Reordering Support for Low-Cost On-Chip Dataflow Switching

In this repository, we include all three key results of the papers, including
- Figure 12: end-to-end deployment on FPGA to run ResNet-50, check `End2end_Deployment` folder
- Figure 13: full design space automated exploration using LayoutLoop 
- Figure 14: the entire Verilog implementation of the whole on-chip computation part of FEATHER, that we use for ASIC synthesis and Place-and-Route

** The detailed execution of each step consumes significant amount of time to reproduce, and thus we attach all pre-run results in results_generation.py for reference. **

# Pre-run results and Figure Plot (Key results reproduction, Estimated 3 minutes)
```
pip3 install matplotlib numpy pandas
python results_generation.py
```

# Structure of the Repo: 
Each folder consists (1) pre-run results for each experiments, and (2) detailed step-by-step operation to reproduce the experiments.
## Experiment Set 1 - Figure 12 - End-to-end deployment on FPGA to run ResNet-50
- Experiment 1 Pre-run results.
- Experiment 1 Pre-built FPGA bitstream to ease the time of reviewer for ease of reproducing the results.
## Experiment Set 2 - Figure 13 - full design space automated exploration using LayoutLoop 
- Experiment 2 Pre-run results `LayoutLoop/pre_run_results`
- Experiment 2 Detail step-by-step running (Optional, Estimated 48 hours)
1. LayoutLoop Framework `LayoutLoop/layoutloop`
2. LayoutLoop configurations for FEATHER (arbitrary layout choice), SIGMA (arbitrary layout choice), SIGMA (off-chip reordering), MTIA-like (Transpose), TPU-like (Transpose + Shift), SIGMA-like (HWC_C4W8), SIGMA-like (HWC_C32), Medusa-like (Line Rotation), Eyeriss-like (HWC_C32), NVDLA-like (HWC_C32)
## Experiment Set 3 - Figure 14 - Verilog Implementation and ASIC Synthesis and Place-and-Routing
- Experiment 3 Pre-run results. 
- Experiment 3 Detail step-by-step running (Optional, Estimated 48 hours)


