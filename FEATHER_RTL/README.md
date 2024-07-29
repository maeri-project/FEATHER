# Figure 14: Synthesis and PnR for FEATHER
In this folder, we open source the Synthesis and PnR flow of FEATHER.

First, we analyze the pre-run results, which are the synthesis and PnR logs from our long-latency local run. 
Second, we provide Verilog Implementation of FEATHER for synthesis and PnR purpose.

# 0 !Pre-run Results Analysis! (Mandatory, reading takes ~2 minutes)

|Config |Area       |Power      |Frequency (GHz)|
|-------|-----------|-----------|---------------|
|64x128 |36920519.69|   26400.00|    1.00       |
|64x64  |18389176.19|   13200.00|    1.00       |
|32x32  | 2727906.70|     961.70|    1.00       |
|16x32  |  965665.10|     655.55|    1.00       |
|16x16  |  475897.19|     323.48|    1.00       |
|8x8    |   97976.46|      65.25|    1.00       |
|4x4    |   24693.98|      16.28|    1.00       |

# Step-by-Step Synthesis and PnR flow for FEATHER (Optional, experiment takes ~ 5 days)
## 1. Dependency
1. Synthesis    -   Synopsys Design Compiler
2. PnR          -   Cadence Innovus

## 2. Environment Description
```bash
├── reports
├── RTL
```

### 3 reports
After successful synthesis the following reports are generated
1. feather_top_area.rpt
2. feather_top_dw_area.rpt
3. feather_top_power.rpt
4. feather_top_timing.rpt

### 4 RTL
All the RTL design files
