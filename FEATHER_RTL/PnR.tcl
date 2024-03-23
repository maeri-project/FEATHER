setDesignMode -process 28
setGenerateViaMode -auto true
set soft_stack_size_limit 102400
setMultiCpuUsage -localCpu 32 -cpuPerRemoteHost 1 -remoteHost 0 -keepLicense true

set init_gnd_net {VSS}
set init_lef_file { \
    /usr/scratch/TSMC_LIB/sixu_tlef_tcbn28hpcplusbwp30p140_190a/tsmcn28_9lm6X1Z1UUTRDL.tlef \
    /usr/scratch/TSMC_LIB/tcbn28hpcplusbwp30p140_190a/tcbn28hpcplusbwp30p140_190a/TSMCHOME/digital/Back_End/lef/tcbn28hpcplusbwp30p140_110a/lef/tcbn28hpcplusbwp30p140.lef 
}
set init_design_settop 0
set init_verilog outputs/feather_top.g.v
set init_mmmc_file mmmc.view
set init_pwr_net {VDD}

init_design
setAnalysisMode -analysisType onChipVariation
saveDesign save/0_init.enc

#49.9762 % utilization
floorPlan -coreMarginsBy die -site core -r 0.99 0.499792 50 50 50 50
#70.0 % utilization
#floorPlan -coreMarginsBy die -site core -r 0.99 0.70 50 50 50 50

saveDesign save/1_FloorPlan.enc


clearGlobalNets
globalNetConnect VDD -type pgpin -pin VDD -instanceBasename *
globalNetConnect VSS -type pgpin -pin VSS -instanceBasename *
globalNetConnect VDD -type tiehi -instanceBasename *
globalNetConnect VSS -type tielo -instanceBasename *


addRing -skip_via_on_wire_shape Noshape -skip_via_on_pin Standardcell -center 1 -stacked_via_top_layer AP -type core_rings -jog_distance 0.065 -threshold 0.065 -nets {VDD VSS} -follow core -stacked_via_bottom_layer M1 -layer {bottom M8 top M8 right M7 left M7} -width 4.5 -spacing 4.5 -offset 10

saveDesign save/2_PowerPlan.enc

sroute -connect { corePin } -deleteExistingRoutes  -nets { VDD  }
sroute -connect { corePin } -deleteExistingRoutes  -nets { VSS  }
saveDesign save/3_sroute.enc

addEndCap -prefix EndCap -preCap BOUNDARY_LEFTBWP30P140 -postCap BOUNDARY_RIGHTBWP30P140
addWellTap -checkerBoard -cellInterval 40 -cell TAPCELLBWP30P140

setDesignMode -bottomRoutingLayer 1
setDesignMode -topRoutingLayer 6
setNanoRouteMode -drouteUseMultiCutViaEffort "high"


setPlaceMode -place_detail_legalization_inst_gap 2
setOptMode -effort high -powerEffort high -leakageToDynamicRatio 0
setPlaceMode -congEffort high
setPlaceMode -place_global_uniform_density true
assignIoPins
place_design

saveDesign save/4_PlaceDesign.enc

setTieHiLoMode -cell {TIEHBWP30P140 TIELBWP30P140}  
addTieHiLo

create_route_type -name top_rule -top_preferred_layer M6 -bottom_preferred_layer M1
create_route_type -name leaf_rule -top_preferred_layer M6 -bottom_preferred_layer M1
create_route_type -name trunk_rule -top_preferred_layer M6 -bottom_preferred_layer M1

set_ccopt_property -route_type top_rule -net_type top
set_ccopt_property -route_type trunk_rule -net_type trunk
set_ccopt_property -route_type leaf_rule -net_type leaf 

set_ccopt_property buffer_cells { CKBD0BWP30P140 CKBD12BWP30P140 CKBD16BWP30P140 CKBD1BWP30P140 CKBD20BWP30P140 CKBD24BWP30P140 CKBD2BWP30P140 CKBD3BWP30P140  CKBD4BWP30P140  CKBD6BWP30P140  CKBD8BWP30P140 }
set_ccopt_property inverter_cells { CKND0BWP30P140 CKND12BWP30P140 CKND16BWP30P140 CKND1BWP30P140 CKND20BWP30P140 CKND24BWP30P140 CKND2BWP30P140 CKND3BWP30P140 CKND4BWP30P140 CKND6BWP30P140 CKND8BWP30P140 }
set_ccopt_property clock_gating_cells  { CKLHQD12BWP30P140 CKLHQD16BWP30P140 CKLHQD1BWP30P140 CKLHQD20BWP30P140 CKLHQD24BWP30P140 CKLHQD2BWP30P140 CKLHQD3BWP30P140 CKLHQD4BWP30P140 CKLHQD6BWP30P140 CKLHQD8BWP30P140 CKLNQD12BWP30P140 CKLNQD16BWP30P140 CKLNQD1BWP30P140 CKLNQD20BWP30P140 CKLNQD24BWP30P140 CKLNQD2BWP30P140 CKLNQD3BWP30P140 CKLNQD4BWP30P140 CKLNQD6BWP30P140 CKLNQD8BWP30P140}

set_ccopt_property target_skew 150ps
set_ccopt_property -net_type top target_max_trans 150ps
set_ccopt_property -net_type trunk target_max_trans 150ps
set_ccopt_property -net_type leaf target_max_trans 150ps

setOptMode -holdFixingCells { DEL025D1BWP30P140 DEL050MD1BWP30P140 DEL075MD1BWP30P140 DEL100MD1BWP30P140 DEL150MD1BWP30P140 DEL200MD1BWP30P140 DEL250MD1BWP30P140}
ccopt_design

saveDesign save/5_CTS.enc


routeDesign
saveDesign save/6_routed.enc
optDesign -postRoute

timeDesign -postRoute -outDir time -numPaths 10
report_area -depth 3 >> area.rpt
report_power -hierarchy 3 -outfile power.rpt

