# Figure 12: End-to-end FEATHER deployment on ZCU104 
In this folder, we provide the jupyter notebook which could be directly run end-to-end on the FPGA device. 
**Further, we open the access of our FPGA ZCU104 board for general public, such that you could directly log into the board and run the script to see the end-to-end performance results.**

## File Explanition
- FEATHER.ipynb: detailed on-FPGA runtime and data processing scripts to deploy convolutions layers down to FEATHER.
- FEATHER.bit: the prebuilt FEATHER bitstream
- FEATHER.hwh: the prebuilt FEATHER hardware handoff

## Deploy Once-for-all weight-sharing ResNet-50 down to FEATHER running on the ZCU104 FPGA.
1. open the link in brower [link](blabababa)
2. type in the password `xilinx` to login.
3. open the jupyter notebook `blabla/FEATHER.ipynb` 
4. run the jupyter notebook block by block and enjoy XD.

## Deploy Once-for-all weight-sharing ResNet-50 down to Xilinx DPU running on the ZCU104 FPGA.
1. open the jupyter notebook `blabla/FEATHER.ipynb` 
2. run the jupyter notebook block by block and enjoy XD.

By comparing the layer-by-layer performance, you could see that FEATHER outperforms Xilinx DPU by most of the layers, which is align with Figure 12.

Note: we provide this end-to-end flow to leverage FEATHER for processing different SubNets of ResNet-50 to demonstrate its superior performance than Xilinx DPU under various model sizes. In the paper with Figure 12, we only show the native versions of ResNet-50 with a single size.
