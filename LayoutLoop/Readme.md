# Figure 13 Reproduction -- LayoutLoop

## About
LayoutLoop is built from [Timeloop](https://parashar.org/ispass19.pdf) to augment Timeloop with layout consideration to coexplore dataflow-layout together for various different AI accelerators including FEATHER.

## Setup
To avoid the tedious dependency, we offer the docker with all dependencies and code being setup.

Steps: download the docker and put it into <path>
```
docker load -i feather_docker.tar 
docker image ls
```
View the image name from the all available docker images.

## Launch the run for different accelerators setup (Optional, > 24 hours)

```
docker run -it -rm <docker_img_name>
git clone <provided_url>
cd FEATHER/LayoutLoop/configurations
make clean
make conv_dse // launch dataflow-layout design space exploration for convolution layers in ResNet-50 and MobileNet-V3
make gemm_dse // launch dataflow-layout design space exploration for Berts
```

The pre-searched results are listed in the pre_run_results, and the collected results are listed in the function named figure13() in `FEATHER/results_generation.py`.


## Fine grain layout reordering modeling control for Layoutloop (Optional, > 24 hours)
In the `./FEATHER/LayoutLoop/layoutloop/src/model/buffer.cpp`, there are some knobs from line 30~36 to enable Layoutloop to consider different layout reordering functionalities.
```
#define STANDARD // Comment out this for MEDUSA to enable row rotation among banks

#ifndef STANDARD
#define MEDUSA
#endif

// #define TRANSPOSE // Uncomment out this for enabling transpose analysis
```

Have fun! Enjoy XD