# FEATHER: A Reconfigurable Accelerator with Data Reordering Support for Low-Cost On-Chip Dataflow Switching [[arXiv]](https://arxiv.org/abs/2405.13170) [[Slides]](https://docs.google.com/presentation/d/1i5jPQ9XK964oEhLqXkZdEt094bnd_4vm/edit?usp=sharing&ouid=105409439591814518444&rtpof=true&sd=true) [[Video]](https://youtu.be/FgTaYiEArrI)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

# What's FEATHER?
FEATHER is the first reconfigurable AI Accelerator that supports (dataflow, layout) co-switching per layer, with the architecture shown below.

<img src="./figure/FEATHER_arch.png" width="400">

Functionality-wise, FEATHER supports arbitrary reordering to ensure arbitrary layout changes.

<img src="./figure/arbitrary_reordering_function.png" width="400">
<!-- ![Arbitrary Reordering]() -->

Performance-wise, FEATHER implements Reorder In Reduction (RIR) to hide the reordering latency behind the critical path.

![Reorder In Reduction (RIR) to hide the reordering latency](./figure/RAR_vs_RIR.png)

- For dataflow switching, FEATHER proposes a reconfigurable 2D compute array, termed NEST (Neural Engine for Spatial forwarding and Temporal reduction)
- For layout switching, FEATHER proposes a reconfigurable NoC, termed BIRRD (Butterfly Interconnect for Reordering in Reduction of Dataflow)
1. BIRRD supports arbitrary reoredering in functionality.
2. BIIRD implements data reordering in data reduction (RIR) to hide reordering latency behind the critical path.

# Repository Overview.

In this repository, we include all three key results of the papers, including
- Figure 12: end-to-end deployment on FPGA to run ResNet-50, check `End2end_Deployment` folder
- Figure 13: full design space automated exploration using LayoutLoop 
- Figure 14: the entire Verilog implementation of the whole on-chip computation part of FEATHER, that we use for ASIC synthesis and Place-and-Route

** The detailed execution of each step consumes significant amount of time to reproduce, and thus we attach all pre-run results in results_generation.py for reference. **

# Pre-run results and Figure Plot (Key results reproduction, Estimated 3 minutes)
```
conda create -n <your_favoriate_name> python=3.8
conda activate <your_favoriate_name>
pip3 install matplotlib numpy pandas
python results_generation.py
```

# Structure of the Repo: 
Each folder consists (1) pre-run results for each experiments, and (2) detailed step-by-step operation to reproduce the experiments.
## Experiment Set 1 - Figure 12 - End-to-end deployment on FPGA to run ResNet-50
- Experiment 1 Pre-run per-layer results are stored in the output of provided jupyter notebook ``feather.ipynb".
- Experiment 1 Pre-built FPGA bitstream to reproduce pre-run results. (Mandatory, ~10 minutes)
## Experiment Set 2 - Figure 13 - full design space automated exploration using LayoutLoop 
- Experiment 2 Pre-run results `LayoutLoop/pre_run_results`
- Experiment 2 Detail step-by-step running (Optional, Estimated > 24 hours)
1. LayoutLoop Framework `LayoutLoop/layoutloop`
2. LayoutLoop configurations for FEATHER (arbitrary layout choice), SIGMA (arbitrary layout choice), SIGMA (off-chip reordering), MTIA-like (Transpose), TPU-like (Transpose + Shift), SIGMA-like (HWC_C4W8), SIGMA-like (HWC_C32), Medusa-like (Line Rotation), Eyeriss-like (HWC_C32), NVDLA-like (HWC_C32)
## Experiment Set 3 - Figure 14 - Verilog Implementation and ASIC Synthesis and Place-and-Routing
- Experiment 3 Pre-run results. 
- Experiment 3 Detail step-by-step running (Optional, Estimated > 96 hours)

# Maintainer
Jianming Tong (jianming.tong@gatech.edu)

Anirudh Itagi (aitagi7@gatech.edu)

Tushar Krishna 

# Citations
```
@inproceedings{tong2024FEATHER, 
  author = {Tong, Jianming and Itagi, Anirudh and Chatarasi, Parsanth and Krishna, Tushar},
   title = {FEATHER: A Reconfigurable Accelerator with Data Reordering Support for Low-Cost On-Chip Dataflow Switching}, 
   year = {2024}, 
   publisher = {Association for Computing Machinery}, 
   address = {Argentina}, 
   abstract = {The inference of ML models composed of diverse structures, types, and sizes boils down to the execution of different dataflows (i.e. different tiling, ordering, parallelism, and shapes). Using the optimal dataflow for every layer of workload can reduce latency by up to two orders of magnitude over a suboptimal dataflow. Unfortunately, reconfiguring hardware for different dataflows involves on-chip data layout reordering and datapath reconfigurations, leading to non-trivial overhead that hinders ML accelerators from exploiting different dataflows, resulting in suboptimal performance. To address this challenge, we propose FEATHER, an innovative accelerator that leverages a novel spatial array termed Nest and a novel multi-stage reduction network called BIRRD for performing flexible data reduction with layout reordering under the hood, enabling seamless switching between optimal dataflows with negligible latency and resources overhead. For systematically evaluating the performance interaction between dataflows and layouts, we enhance Timeloop, a state-of-the-art dataflow cost modeling and search framework, with layout assessment capabilities, and term it as Layoutloop. We model FEATHER into Layoutloop and also deploy FEATHER end-to-end on the edge ZCU104 FPGA. FEATHER delivers 1.27~2.89x inference latency speedup and 1.3~6.43x energy efficiency improvement compared to various SoTAs like NVDLA, SIGMA and Eyeriss under ResNet-50 and MobiletNet-V3 in Layoutloop. On practical FPGA devices, FEATHER achieves 2.65/3.91x higher throughput than Xilinx DPU/Gemmini. Remarkably, such performance and energy efficiency enhancements come at only 6% area over a fixed-dataflow Eyeriss-like accelerator.}, 
   booktitle = {Proceedings of the 51th Annual International Symposium on Computer Architecture}, 
   keywords = {flexible accelerator, dataflow-layout coswitching}, 
   location = {Argentina}, 
   series = {ISCA '24} }
```
