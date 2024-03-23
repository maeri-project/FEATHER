# Figure 14: Synthesis and PnR for FEATHER
In this folder, we open source the Synthesis and PnR flow of FEATHER.

First, we analyze the pre-run results, which are the synthesis and PnR logs from our long-latency local run. 
Second, we provide code base and detailed step-by-step ASIC synthesis and PnR flow to rerun Syn and PnR for FEATHER.
Note that: Synthesis and PnR estimated times >96 hours for the largest design[64x64], and thus it's optional.

## 0 !Pre-run Results Analysis! (Mandatory, reading takes 4 minutes)
@Anirudh, adding the prerun results

## 1. Dependency
1. Synthesis    -   Synopsys Design Compiler
2. PnR          -   Cadence Innovus

## 2. Environment Description
```bash
├── alib-52
├── cons
├── lib_setup
├── scripts
├── outputs
├── reports
├── RTL
├── mmmc.view
└── PnR.tcl
```
### 2.1 alib-52
lib files relevant to TSMC 28 nm process (don't touch!)

The provided codebase is intended exclusively for artifact evaluation purposes. Please refrain from distributing the files externally or utilizing them for any other purpose.


### 2.2 cons
Synthesis Constraints
1. **defaults.tcl** \
    This file contains the essential constraints such as input clock constraint, skew, input/output constraints etc. \
2. **design_cons.tcl** \
    This file contains the input clock frequency (at line 2)
    ```tcl
    set clk_period 1
    ```
    Where 1 here is 1 ns.
### 2.3 lib_setup

The dc_setup_lvt.tcl and dc_setup_svt.tcl files will contain the paths to the TSMC Library.

Please ensure the correct paths are set for:

```tcl
set target_library {absolute/path/to/tsmc_libfile.db}
set synthetic_library {absolute/path/to/tsmc_libfile.db}
set link_library {absolute/path/to/tsmc_libfile.db}
```

### 2.4 scripts
1. **syn.tcl** \
    set the top file at:
    ```tcl
    #set TOP_DESIGN top_most_heriarchy_module_name
    set TOP_DESIGN feather_top
    ```

2. **:run_syn**


    This is the script to launch synthesis

## 3. Running Synthesis with Synopsys Design Compiler - (Optional, >96 hours)

```bash
$ csh
> source /tools/software/synopsys/setup.csh
> source /tools/software/cadence/innovus/cshrc.latest
> source /tools/software/cadence/genus/cshrc.latest
> source :run_syn
```

### 3.1 outputs
After successful synthesis the following files are generated
1. feather_top.**g.v**
2. feather_top.**sdc**
3. feather_top.**sdf**
4. feather_top.**ddc**

### 3.2 reports
After successful synthesis the following reports are generated
1. feather_top_area.rpt
2. feather_top_dw_area.rpt
3. feather_top_power.rpt
4. feather_top_timing.rpt

### 3.3 RTL
All the RTL design files

### 3.4 PnR Scripts
Please ensure synthesis is run successfully before updating and running the below scripts
1. **mmmc.view** 
    1. Ensure **qx_tech_file**, **library_file** and **timing** library paths are set
    2. Set the path to the synthesis generated **feather_top.sdc** file
    ```tcl
    create_constraint_mode -name cons -sdc_files {absolute/path/to/top_most_heriarchy_module_name.sdc}
    ```

2. PnR.tcl 

    1. update the lef fille path:
    ```tcl
    set init_lef_file { \
        absolute/path/to/tsmc_libfile.tlef \
        absolute/path/to/tsmc_libfile.lef 
    }
    ```

    2. Update path to synthesis generated **top_most_heriarchy_module_name.g.v**
    ```tcl
    set init_verilog absolute/path/to/top_most_heriarchy_module_name.g.v
    ```

## 4. Running PnR with Innovus - (Optional, >96 hours)
1. After sourcing all the environments:
```bash
> innovus
```
2. Wait for innovus to launch, and then in the innovus cmd line source the PnR.tcl to launch the Place and Route
```bash
> source PnR.tcl
```
3. after it is completed the innovus the following files will be generated:
```bash
.
├── save
├── area.rpt
├── innovus.cmd
├── innovus.log
├── innovus.logv
├── power.rpt
├── rc_model.bin
├── time
└── timingReports
```
4. work files will be generated and saved in the folder **save** and the area and timing reports as shown above

5. the checkpoints during the PnR can be found in:
```bash
└── save
    ├── 0_init.enc
    ├── 0_init.enc.dat
    ├── 1_FloorPlan.enc
    ├── 1_FloorPlan.enc.dat
    ├── 2_PowerPlan.enc
    ├── 2_PowerPlan.enc.dat
    ├── 3_sroute.enc
    ├── 3_sroute.enc.dat
    ├── 4_PlaceDesign.enc
    ├── 4_PlaceDesign.enc.dat
    ├── 5_CTS.enc
    ├── 5_CTS.enc.dat
    ├── 6_routed.enc
    └── 6_routed.enc.dat
``` 
