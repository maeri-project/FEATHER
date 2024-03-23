# Figure 13 Reproduction -- LayoutLoop

## About
LayoutLoop is built from [Timeloop](https://parashar.org/ispass19.pdf) to augment Timeloop with layout consideration to coexplore dataflow-layout together for various different AI accelerators including FEATHER.

## Setup
To avoid the tedious dependency, we offer the docker with all dependencies and code being setup.

Steps: download the docker and put it into <path>
```
unzip <path>/feather_docker.zip
docker load --input <decompressed_image>
docker image ls
```
View the image name from the all available docker images.

## Launch the run for different accelerators setup

```
docker run -it -rm <docker_img_name>
git clone <provided_url>
cd FEATHER/LayoutLoop/configurations/scripts
python3 search_layoutloop_specific_arch.py ../test
```

We'll update this readme.md with details setup to change for each individual design soon.

Enjoy XD