# LayoutLoop

## 1.

LayoutLoop is a tool based on [TimeLoop](https://timeloop.csail.mit.edu/). It integrates the functionalities of more accurate layout-based memory modeling.

The key contributions of SquareLoop over previous tools are:
* realistic layout-based memory model utilizing accurate dataspace-wise evaluation
* introduction of physical ranks, allowing for independent per-dataspace layout and AuthBlock specification 
* Layout-Mapping co-search algorithm


## 2. Setup
To avoid the tedious dependency, we offer the docker with all dependencies and code being setup.

### 2.0 Files Overview

We use the following files in the experiments:

* Architecture
    * SIGMA (vector256, full-flege flexible accelerator)
        * `benchmarks/arch_designs/vector_256.yaml`
    * SIMBA (reconfigurable systolic array)
        * `benchmarks/arch_designs/simba_like.yaml`
        * `benchmarks/arch_designs/components/*`
        * `benchmarks/arch_designs/constraints/*`
    * Edge-TPU (systolic)
        * `benchmarks/arch_designs/vector_256.yaml`
        * `benchmarks/arch_designs/systolic_constraint/mapspace_XY_OS.yaml`
        * `benchmarks/arch_designs/systolic_constraint_depthwise/mapspace_XY_OS.yaml`
    * Eyeriss (eyeriss)
        * `benchmarks/arch_designs/eyeriss_like/arch/eyeriss_like.yaml`
        * `benchmarks/arch_designs/eyeriss_like/arch/components/*`
        * `benchmarks/arch_designs/eyeriss_like/constraints/*` (constraint for convolution workload only)
        * `benchmarks/arch_designs/eyeriss_like/constraints_depthwise/*`  (constraint for depth-wise convolution workload only)
* Workloads
    * ResNet18
        * `benchmarks/layer_shapes/resnet18/*`
    * ResNet50
        * `benchmarks/layer_shapes/resnet50/*`
    * MobileNetV3 (mobv3)
        * `benchmarks/layer_shapes/mobv3/*`
    * bert
        * `benchmarks/layer_shapes/bert/*`
    * bert_conv (converted matrix multiplication as the form of convolution)
        * `benchmarks/layer_shapes/bert_conv/*`
    * vgg small
        * `benchmarks/layer_shapes/vgg01/*`
    * vgg large
        * `benchmarks/layer_shapes/vgg02/*`
    * AlexNet
        * `benchmarks/layer_shapes/AlexNet/*`
* Mapper
    * `benchmarks/mapper/mapper.yaml`


### 2.1 Software Dependency -- Docker installation
```
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```
### 2.2 Download and Setup prebuilt docker.
Steps: download the docker [link](https://drive.google.com/file/d/1-BdCdKI00gXY8GKtNpwHVtn4b-r_vNCC/view?usp=sharing) and install it
```bash
docker image ls
docker load -i feather_layoutloop_docker.tar.gz 
```
View the image name from the all available docker images.

## 3. Experiment: Launch the run for different accelerators setup (Optional, > 24 hours)

```bash
#docker run -it <docker_img_name>
docker run -it feather_layoutloop

```

When inside the docker
```bash
pip install torch torchlens pyyaml torchvision pandas
git clone <provided_url>
#e.g. git clone https://github.com/maeri-project/FEATHER.git
```

```bash
cd FEATHER/LayoutLoop/layoutloop
scons -j<number_of_available_threads>
cd FEATHER/LayoutLoop/configurations
make clean
make dse  # launch dataflow design space exploration for various architectures under ResNet-18, MobileNet-V3 and Bert -- using layoutloop based precise memory modeling
```

The old pre-searched results are listed in the pre_run_results, and the collected results are listed in the function named figure13() in `FEATHER/results_generation.py`.

## 4. Results Analysis
### 4.1 Pre-run results analysis (Mandatory, just reading the prerun-results, take ~5 minutes)
1. All pre-run results are sitting in the folder `FEATHER/LayoutLoop/pre_run_results`
```
└── results_precise_layout_modeling
```

2. In each folder, there are 4 different csv files 
```
├──utilization.csv: the average computation utilization of searched dataflow under designated layout (e.g. 1 mean 100% utilization)
├──cycle.csv: the overall clock cycle of processing given workload levearging the searched dataflow under designated layout (e.g. 452313.00 mean 452313 clock cycles)
└──pj_commpute.csv: computation energy efficiency of processing given workload levearging the searched dataflow under designated layout (e.g 2.17 mean 2.17 pJ/MAC)
```
The number of row in each csv file is the total number of layer for given workloads. For ease of reading searched results, we also provide the `interleave_layoutloop_search.csv` to merge all above four files together.
- column 1: utilization.csv
- column 2: cycle.csv
- column 3: pj_commpute.csv

3. The searched dataflow is located at the `mapping_search` directory. 
The number of row in each csv file is the total number of layer for given workloads, and the index indicating the layer index. 

### 4.2 New results analysis (Optional, only needed if you run 2,3 above, take ~5 minutes  )
Finishing experiments, the results are stored in the `configurations/results`, while the name pattern of the results is shown as followes
```
{design_name}_interleave_layoutloop_search.csv
```
where, 
- workloads are "resnet50" (53 layers), "mobv3" (62 layers), "bert" (3 layers). In total, you will see 118 layers (rows in the csv). 
- design_name could be "gemmini", "eyeriss", "sigma", "simba", "medusa", "systolic_array".
- layout_policy could be "SRCQPMNHW_Cx32", "SRCQPMNHW_Hx32", "SRCQPMNHW_Mx32", "SRCQPMNHW_Wx32", "SRCQPMNHW_Cx4Hx8", "SRCQPMNHW_Cx8Hx4", "SRCQPMNHW_Cx8Wx4", "SRCQPMNHW_Wx8Hx4", "SRCQPMNHW_Cx8Wx2Hx2".
- column 1: utilization
- column 2: cycle
- column 3: pj_commpute


Have fun! Enjoy XD
