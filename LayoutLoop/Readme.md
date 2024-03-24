# Figure 13 Reproduction -- LayoutLoop

## 1. Overview
LayoutLoop is built from [Timeloop](https://parashar.org/ispass19.pdf) to augment Timeloop with layout consideration to coexplore dataflow-layout together for various different AI accelerators including FEATHER.
In this folder, we demonstrate steps to leverage proposed Layoutloop to enable dataflow-layout co-exploration for various AI accelerators.

**[TL,DR] The searching takes long time (~2 hours on average per design), such that we offer pre-run results in the `FEATHER/LayoutLoop/pre_run_results` folder. (The analysis is in 5 of this readme.md)**

[TL,DR] In the following text, we first show 
- environment setup (2),
- Experiment of co-searching dataflow-layout (3), 
- Experiemnt of exploring different rerordering capabilities (4) 
- results analysis (5) including pre-run results and how to understand new results.

## 2. Setup
To avoid the tedious dependency, we offer the docker with all dependencies and code being setup.

### 2.1 Software Dependency -- Docker installation
```
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```
### 2.2 Download and Setup prebuilt docker.
Steps: download the docker [link](https://drive.google.com/file/d/1-BdCdKI00gXY8GKtNpwHVtn4b-r_vNCC/view?usp=sharing) and install it
```
docker load -i feather_docker.tar 
docker image ls
```
View the image name from the all available docker images.


## 3. Experiment: Launch the run for different accelerators setup (Optional, > 24 hours)

```
docker run -it -rm <docker_img_name>
git clone <provided_url>
cd FEATHER/LayoutLoop/configurations
git pull
make clean
make conv_dse // launch dataflow-layout design space exploration for convolution layers in ResNet-50 and MobileNet-V3
make gemm_dse // launch dataflow-layout design space exploration for Berts
```

The pre-searched results are listed in the pre_run_results, and the collected results are listed in the function named figure13() in `FEATHER/results_generation.py`.

## 4. Experiment: Explore different layout reordering capabilities provided by Layoutloop (Optional, > 24 hours)
The proposed Layoutloop could be configured to support different functionalities including Line Rotation (Fig. 6b), Transpose (Fig. 6c). Row-reordering does not affect the layout analysis, while arbitrary reordering is natively supported when do not consider impact of data layout. 

To modify Layoutloop code to enable Line Rotation or Transpose analysis, please uncomment/comment the macro in the `./FEATHER/LayoutLoop/layoutloop/src/model/buffer.cpp`. Specifically, the knobs from line 30~36 to enable Layoutloop are shown as below.
```
#define STANDARD // Comment out this for MEDUSA to enable line rotation (Fig. 6b) among banks.

#ifndef STANDARD
#define MEDUSA
#endif

// #define TRANSPOSE // Uncomment out this for enabling transpose analysis
```
1. Enabling Line Rotation proposed in MEDUSA (Change the header into the following)
```
// #define STANDARD // Comment out this for MEDUSA to enable line rotation (Fig. 6b) among banks.

#ifndef STANDARD
#define MEDUSA
#endif

// #define TRANSPOSE // Uncomment out this for enabling transpose analysis
```
2. Enabling Transpose (Change the header into the following)
```
#define STANDARD // Comment out this for MEDUSA to enable line rotation (Fig. 6b) among banks.

#ifndef STANDARD
#define MEDUSA
#endif

#define TRANSPOSE // Uncomment out this for enabling transpose analysis
```
3. Enabling both Line Rotation and Transpose (Change the header into the following)
```
// #define STANDARD // Comment out this for MEDUSA to enable line rotation (Fig. 6b) among banks.

#ifndef STANDARD
#define MEDUSA
#endif

#define TRANSPOSE // Uncomment out this for enabling transpose analysis
```

## 5. Results Analysis
### 5.1 Pre-run results analysis (Mandatory, just reading the prerun-results, take ~5 minutes)
1. All pre-run results are sitting in the folder `FEATHER/LayoutLoop/pre_run_results`
```
├── eyeriss_like_bert
├── eyeriss_like_ResNet50_MobV3
├── FEATHER_bert
├── FEATHER_ResNet50_MobV3
├── medusa_ResNet5_MobV3
├── MTIA_like_ResNet50_MobV3
├── sigma_like_bert
├── sigma_off_chip_reordering_ResNet50_MobV3
└── TPU_like_ResNet50_MobV3
```

2. In each folder, there are 4 different csv files 
```
├──slowdown.csv: a the slowdown of search dataflow under given layout (1 means no slowdown. 0.64 mean 36% slowdown. Please ignore this file if it only consists of 0).
├──utilization.csv: the average computation utilization of searched dataflow under designated layout (e.g. 1 mean 100% utilization)
├──cycle.csv: the overall clock cycle of processing given workload levearging the searched dataflow under designated layout (e.g. 452313.00 mean 452313 clock cycles)
└──pj_commpute.csv: computation energy efficiency of processing given workload levearging the searched dataflow under designated layout (e.g 2.17 mean 2.17 pJ/MAC)
```
The number of row in each csv file is the total number of layer for given workloads. For ease of reading searched results, we also provide the `interleave_layoutloop_search.csv` to merge all above four files together.
- column 1: slowdown.csv
- column 2: utilization.csv
- column 3: cycle.csv
- column 4: pj_commpute.csv

3. The searched dataflow is located at the `mapping_search` directory. 
The number of row in each csv file is the total number of layer for given workloads, and the index indicating the layer index. 

### 5.1 New results analysis (Optional, only needed if you run 2,3 above, take ~5 minutes  )
Finishing experiments, the results are stored in the `configurations/results`, while the name pattern of the results is shown as followes
```
{design_name}_{layout_policy}_interleave_layoutloop_search.csv
```
where, 
- design_name could be `"SIGMA", "TPU_like", "MTIA_like", "Medusa_like", "eyeriss_like", "simba_like"`
- layout_policy could be `"HWC_C32", "HWC_W32", "HWC_H32", "HWC_C4W8", "HWC_C4H8", "HWC_W4H8", "HWC_C4W4H2", "HWC_W32_W2", "HWC_H32_H2"`
- column 1: slowdown
- column 2: utilization
- column 3: cycle
- column 4: pj_commpute


Have fun! Enjoy XD
