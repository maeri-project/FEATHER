# Figure 12: End-to-end FEATHER deployment on ZCU104 
In this folder, we provide the jupyter notebook which could be directly run end-to-end on the FPGA device. 
**Further, we open the access of our FPGA ZCU104 board for general public, such that you could directly log into the board and run the script to see the end-to-end performance results.**

## File Explanition
- FEATHER.ipynb: detailed on-FPGA runtime and data processing scripts to deploy convolutions layers down to FEATHER.
- FEATHER.bit: the prebuilt FEATHER bitstream
- FEATHER.hwh: the prebuilt FEATHER hardware handoff

## FEATHER: Deploy Once-for-all weight-sharing ResNet-50 down to **FEATHER** running on the ZCU104 FPGA (Mandatory, experiment takes ~5 minutes)
1. open the link in brower [link](https://jupyter_zcu104.lukezhang97.com/)
2. type in the password `isca2024artifact` to login.
3. open the jupyter notebook `feather/feather.ipynb`.
4. run all blocks in jupyter notebook, where we pick 6 SubNets of weights-sharing resnet-50 [link](https://github.com/mit-han-lab/once-for-all/). FETHER could accelerate all different types of layers with different dataflows - layout, and it requires offline compilation to search (dataflow, layout) pairs for each model. To simplify the reproduction complexity, we only show one type of workload in this reproduction, which delivers the key performance/functionality of FEATHER.

## Baseline: Deploy Once-for-all weight-sharing ResNet-50 down to **Xilinx DPU** running on the ZCU104 FPGA  (Optional, experiment takes ~4 minutes)
We also provide codes to run the commerical Xilinx DPU accelerator for the same set of workloads and obtain per-layer performance.
1. stay tuned.

By comparing the layer-by-layer performance, you could see that FEATHER outperforms Xilinx DPU by most of the layers, which is align with Figure 12.

Note: we provide this end-to-end flow to leverage FEATHER for processing different SubNets of ResNet-50 to demonstrate its superior performance than Xilinx DPU under various model sizes. In the paper with Figure 12, we only show the native versions of ResNet-50 with a single size.
