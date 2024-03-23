
set search_path "."

# Set Library Paths
# TSMC std cell libs are in following directories
# /home/green1/DKIT/tsmc_muse/IP/TSMC<process_node>_STD_CELLS/custom_utils_ck/tsmc_ip/<library_name>/nldm/<library_name><process_corner><op_voltage><op_temp>.db
#
# Following example is for TMSC 28nm Std Cell Library
# <process_node>	: 28
# <library_name>	: tcbn28hpcplusbwp30p140
# <process_corner>	: tt
# <op_voltage>		: 0.8V
# <op_temp>		: 25C
#set link_library {* /usr/scratch/DRBE-shared/TSMC-library/IP/TSMC28_STD_CELLS/unzipped/tcbn28hpcplusbwp30p140lvt_190a/tcbn28hpcplusbwp30p140lvt_180a_nldm/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp30p140lvt_180a/tcbn28hpcplusbwp30p140lvttt0p8v25c.db}
#set target_library {/usr/scratch/DRBE-shared/TSMC-library/IP/TSMC28_STD_CELLS/unzipped/tcbn28hpcplusbwp30p140lvt_190a/tcbn28hpcplusbwp30p140lvt_180a_nldm/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp30p140lvt_180a/tcbn28hpcplusbwp30p140lvttt0p8v25c.db}#set target_library {/home/green1/DKIT/tsmc_muse/IP/TSMC28_STD_CELLS/custom_utils_ck/tsmc_ip/tcbn28hpcplusbwp30p140/nldm/tcbn28hpcplusbwp30p140tt0p8v25c.db}
#set target_library {/home/green1/DKIT/tsmc_muse/IP/TSMC28_STD_CELLS/custom_utils_ck/tsmc_ip/tcbn28hpcplusbwp30p140/nldm/tcbn28hpcplusbwp30p140tt0p8v25c.db}
set synthetic_library "/usr/scratch/jianming/AreaPower/sequential_circuit/lib/NanGate_15nm_OCL.db"
set link_library "/usr/scratch/jianming/AreaPower/sequential_circuit/lib/NanGate_15nm_OCL.db"
set target_library "/usr/scratch/jianming/AreaPower/sequential_circuit/lib/NanGate_15nm_OCL.db"
report_lib "/usr/scratch/jianming/AreaPower/sequential_circuit/lib/NanGate_15nm_OCL.db"

set generic_symbol_library [list generic.sdb]
set synthetic_library [list dw_foundation.sldb ]

set link_library [concat \
	[concat "*" $target_library] $synthetic_library]

