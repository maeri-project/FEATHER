def figure_12():
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.patches as patches

    ##### Default Parameters
    color_list = ["#6C8EBF",  "#B85450",   "#000000", "#D79B00",  "#82B366", "#B46504", "#D6B656","#23445D"]
    # Style for the added patch
    style = "Simple, tail_width=0.5, head_width=4, head_length=8"
    patterns = [ "//" , "\\\\" , "oo" , "*" , "." , "xx", "o", "O", "o-", " " ]
    kw = dict(arrowstyle=style, color="k")

    # latency = [[0.364,1.241,3.52,1.242,1.241,1.241,1.891,1.559,2.259,1.545,4.1,2.583,2.547,2.551,1.726,1.726,1.725,2.635,2.433,4.673,1.549],[1.314488649,1.030504704,1.804955006,0.650472641,0.6486463547,0.649600029,1.11974597,3.076827526,1.22366786,0.8738791943,2.147263288,4.394212961,1.337183714,1.330988407,0.9275352955,0.9316790104,0.9204244614,4.940673113,1.776784658,3.425372839,1.195193529]]
    normalized_latency_speedup_ori = [[9.612704927, 3.689799384, 3.979867895, 3.878334143, 2.165835855, 4.34880074, 3.713235222, 3.59773046, 4.350333581, 3.545281462, 3.601696454, 4.261384203, 4.916975143, 4.280533936, 9.753116037, 3.470726318, 3.071001232, 4.613892878, 3.328430706, 3.16191283, 4.489446547, 3.458365098, 3.066382703, 4.483779962, 2.823516752, 5.447681139, 3.174684961, 15.62306086, 3.306038308, 3.156878772, 3.158919482, 3.301516736, 3.176653338, 3.142215214, 3.327101013, 3.170782066, 3.127342353, 3.419490048, 3.089733896, 3.132584174, 3.429804402, 3.088128723, 3.133144369, 3.159865964, 4.311966399, 3.75896804, 10.88167834, 4.56571244, 3.756183996, 3.706525197, 4.600694855, 3.648865156, 3.813134559],
                                    [1.533258353, 17.50278977, 12.46335781, 10.79974192, 2.192368989, 21.60430605, 4.385197002, 11.35242946, 4.59003381, 5.590571118, 5.554651695, 2.297265378, 5.554273166, 4.598151758, 11.10535463, 4.596918339, 5.902161049, 2.596657747, 5.879007601, 2.934071471, 2.61581125, 5.844073923, 1.309276091, 5.872852887, 1.307526683, 2.934148663, 2.6197113, 5.870552816, 2.617351786, 3.210939073, 1.50042186, 3.494327121, 1.58929948, 1.498911045, 3.175884447, 0.751247861, 1.500367371, 1.501023709, 0.7431270233, 1.485864372, 1.485761174, 0.7426126857, 1.49832323, 0.7461333803, 0.7499983426, 1.500270778, 1.600015838, 1.586745762, 0.7500920461, 1.499942197, 1.598527589, 0.7455459787, 1.499773778],
    [9.300683147,18.74471109,4.788034712,20.51810712,10.21029982,12.75799078,4.58162771,20.30595478,10.57090106,3.548678966,18.70905141,10.25589934,2.759146025,15.93569322,6.614737713,6.455378323,1.886181533,11.59502278,5.428181993,1.841424047,11.7025878,6.096857294,2.025968263,14.23327654,5.165194663,1.821420615,7.97852046,3.517514855,3.45250314,1.377686679,8.302203052,3.454569951,1.409787113,8.104106171,3.454918484,1.381139988,7.95251439,3.606166269,1.375684845,7.925899015,3.456861314,1.366743038,7.979980472,2.746462455,1.145888382,4.250431811,2.817454569,2.239374079,0.9163402925,4.282494339,2.186027652,0.9126709387,4.226411341]]


    normalized_latency_speedup = np.log(normalized_latency_speedup_ori)
    x = [i for i in range(len(normalized_latency_speedup[1]))]

    # Create figure
    fig = plt.figure(figsize=[10,3.5])
    ax = plt.subplot(111)
    barWidth=0.3
    acu_matrix = []
    sum_matrix = [normalized_latency_speedup[0]]
    for i in range(len(sum_matrix)): 
        if i == 0:
            acu_matrix.append(sum_matrix[0])
        else:
            temp = []
            for j in range(len(sum_matrix[i])):
                temp.append(acu_matrix[i-1][j] + sum_matrix[i][j])
            acu_matrix.append(temp)

    plt_handler = []

    for i in range(len(x)):
        for j in range(len(acu_matrix)):
            if j==0:
                plt.bar(x[i]-2*barWidth/3, sum_matrix[j][i], width=2*barWidth/3, bottom=0, color=color_list[j])
            else:
                plt.bar(x[i]-2*barWidth/3, sum_matrix[j][i], width=2*barWidth/3, bottom=0, color=color_list[j])
            
            if i == len(x) - 1:
                if j==0:
                    plt_handler.append(plt.bar(x[i]-2*barWidth/3, sum_matrix[j][i], width=2*barWidth/3, bottom=0, color=color_list[0]))
                else:
                    plt_handler.append(plt.bar(x[i]-2*barWidth/3, sum_matrix[j][i], width=2*barWidth/3, bottom=0, color=color_list[0]))

    acu_matrix_xilinx_dpu = []
    sum_matrix_xilinx_dpu = [normalized_latency_speedup[1]]
    for i in range(len(sum_matrix_xilinx_dpu)):
        if i == 0:
            acu_matrix_xilinx_dpu.append(sum_matrix_xilinx_dpu[0])
        else:
            temp = []
            for j in range(len(sum_matrix_xilinx_dpu[i])):
                temp.append(acu_matrix_xilinx_dpu[i-1][j] + sum_matrix_xilinx_dpu[i][j])
            acu_matrix_xilinx_dpu.append(temp)

    for i in range(len(x)):
        for j in range(len(acu_matrix_xilinx_dpu)):
            if j==0:
                plt.bar(x[i], sum_matrix_xilinx_dpu[j][i], width=2*barWidth/3, bottom=0, color=color_list[1])
            else:
                plt.bar(x[i], sum_matrix_xilinx_dpu[j][i], width=2*barWidth/3, bottom=0, color=color_list[1])
            
            if i == len(x) - 1:
                if j==0:
                    plt_handler.append(plt.bar(x[i], sum_matrix_xilinx_dpu[j][i], width=2*barWidth/3, bottom=0, color=color_list[1]))
                else:
                    plt_handler.append(plt.bar(x[i], sum_matrix_xilinx_dpu[j][i], width=2*barWidth/3, bottom=0, color=color_list[1]))



    acu_matrix_edge_tpu = []
    sum_matrix_edge_tpu = [normalized_latency_speedup[2]]
    for i in range(len(sum_matrix_edge_tpu)):
        if i == 0:
            acu_matrix_edge_tpu.append(sum_matrix_edge_tpu[0])
        else:
            temp = []
            for j in range(len(sum_matrix_edge_tpu[i])):
                temp.append(acu_matrix_edge_tpu[i-1][j] + sum_matrix_edge_tpu[i][j])
            acu_matrix_edge_tpu.append(temp)

    for i in range(len(x)):
        for j in range(len(acu_matrix_edge_tpu)):
            if j==0:
                plt.bar(x[i]+2*barWidth/3, sum_matrix_edge_tpu[j][i], width=2*barWidth/3, bottom=0, color=color_list[2])
            else:
                plt.bar(x[i]+2*barWidth/3, sum_matrix_edge_tpu[j][i], width=2*barWidth/3, bottom=0, color=color_list[2])
            
            if i == len(x) - 1:
                if j==0:
                    plt_handler.append(plt.bar(x[i]+2*barWidth/3, sum_matrix_edge_tpu[j][i], width=2*barWidth/3, bottom=0, color=color_list[2]))
                else:
                    plt_handler.append(plt.bar(x[i]+2*barWidth/3, sum_matrix_edge_tpu[j][i], width=2*barWidth/3, bottom=0, color=color_list[2]))

    print(f"latency saving ratio range over Xilinx DPU is [{1-np.max(np.array(acu_matrix_xilinx_dpu[-1])/np.array(acu_matrix[-1]))},  {1-np.min(np.array(acu_matrix_xilinx_dpu[-1])/np.array(acu_matrix[-1]))}]")
    print(f"latency saving ratio range over Edge TPU is [{1-np.max(np.array(acu_matrix_edge_tpu[-1])/np.array(acu_matrix[-1]))},  {1-np.min(np.array(acu_matrix_edge_tpu[-1])/np.array(acu_matrix[-1]))}]")

    SMALL_SIZE = 16
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 22
    plt.xlabel(r"Layer ID in ResNet50", fontsize=SMALL_SIZE)
    plt.ylabel(r"Normalized $Log_2(Thrpt./PE)$", fontsize=15)
    # plt.xticks(x, [f"{i}" for i in [0, 1, 2, 4, 8, 11, 14, 17, 21, 24, 27, 30, 34, 37, 40, 43, 46, 49, 53, 56, 59]], fontsize=SMALL_SIZE)
    # plt.xticks(x, [f"{i}" for i in [1, 2, 4, 8, 11, 14, 21, 24, 27, 34, 37, 40, 43, 46, 53, 56, 59]], fontsize=SMALL_SIZE)
    # plt.yticks([15,30,45,60,75], fontsize=BIGGER_SIZE)
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    # plt.ylim((min(accuracy_pre_train)*0.5,max(accuracy_post_train)*1.1))
    # plt.legend(plt_handler, [r"Xilinx DPU", r"LAMBDA"], fontsize=SMALL_SIZE)
    geomean= [np.prod(normalized_latency_speedup_ori[0])**(1/len(normalized_latency_speedup_ori[0])), np.prod(normalized_latency_speedup_ori[1])**(1/len(normalized_latency_speedup_ori[1])), np.prod(normalized_latency_speedup_ori[2])**(1/len(normalized_latency_speedup_ori[2]))]
    # plt.plot([-0.5, 17.5], [geomean[0], geomean[0]], linewidth=2, color=color_list[0], linestyle="--")
    # plt.plot([-0.5, 17.5], [geomean[1], geomean[1]], linewidth=2, color=color_list[1], linestyle="--")
    plt_handler.append(plt.plot([0, 52], [np.log(geomean[0]), np.log(geomean[0])], linewidth=2, color=color_list[0], linestyle="--")[0])#, marker = 'o'))
    plt_handler.append(plt.plot([0, 52], [np.log(geomean[1]), np.log(geomean[1])], linewidth=2, color=color_list[1], linestyle="--")[0])#, marker = 'o'))
    plt_handler.append(plt.plot([0, 52], [np.log(geomean[2]), np.log(geomean[2])], linewidth=2, color=color_list[2], linestyle="--")[0])#, marker = 'o'))
    # ax.annotate('SushiAccel Speedup 19.5% over Xilinx DPU', xy=(4, np.mean(normalized_latency_speedup[1])), xytext=(4, 2.2), arrowprops=dict(facecolor='black', shrink=0.005))
    plt.text(55.5, np.log(geomean[0]*0.9), f'{(geomean[0]):0.2f}X ',color=color_list[0], horizontalalignment='center')
    plt.text(55.5, np.log(geomean[1]*0.85), f'{(geomean[1]):0.2f}X ',color=color_list[1], horizontalalignment='center')
    plt.text(55.5, np.log(geomean[2]*1.1), f'{(geomean[2]):0.2f}X ', color=color_list[2],horizontalalignment='center')
    plt.legend(plt_handler, [r"Speedup Over Gemmini", r"Speedup Over Xilinx DPU", r"Speedup Over Edge TPU", r"Speedup Over Gemmini (GeoMean)", r"Speedup Over Xilinx DPU (GeoMean)", r"Speedup Over Edge TPU (GeoMean)"], ncol=2, fontsize=14, labelspacing = 0, columnspacing=0.5, frameon=False, loc='center', bbox_to_anchor=(0.5, 0.88),)

    # plt.text(1.5, 2, f'MaxPooling', horizontalalignment='center')
    plt.ylim([np.min(normalized_latency_speedup)*2, 4])
    plt.xlim([-1, 58])
    # plt.plot([5.5, 5], [geomean[0], 3], linewidth=1, color="k", linestyle="--")
    # plt.text(20, np.mean(normalized_latency_speedup[1]), 'SushiAccel GeoMean', horizontalalignment='center', color=color_list[0])
    # plt.text(20, np.mean(normalized_latency_speedup[0]), 'Xilinx GeoMean', horizontalalignment='center', color=color_list[1])
    plt.savefig(r'Latency_Comparison_LAMBDA_DPU.pdf', bbox_inches="tight", transparent=True) 


def figure_13():
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.patches import Ellipse
    import csv
    from matplotlib import gridspec
    from matplotlib import rc
    import matplotlib.patches as patches
    rc('text',usetex=False)

    SMALL_SIZE = 19
    MEDIUM_SIZE = 21
    BIGGER_SIZE = 26
    color_list = ["#6C8EBF",  "#D79B00", "#82B366", "#B85450", "#9673A6", "#B46504", "#D6B656","#23445D"]
    layout_list = ["HWC_C32","HWC_C32","HWC_C32","HWC_C4W8", "off-chip\nreorder", "line rotation", "Transpose", "Trans.+Shuff.", "RIR"]
    # layout_list = ["HWC_C32","HWC_C32","HWC_C32","HWC_C4W8", "off-chip\nreorder", "line\nrotation", "RIR"]
    gemm_layout_list = ["MK_K32","MK_K32","MK_K32","RIR"]

    # Simple data to display in various forms

    fig = plt.figure(figsize=[27,5.5])
    # set height ratios for subplots
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[2,4.8,4.8]) 

    #####################
    ## Bert
    #####################

    x = np.array([0, 1, 2, 3])

    normalized_pj_compute = np.array([6.43430066,5.980508158,1.439967617,1])
    normalized_latency = np.array([2,1.434626437,1,1])
    reorder = np.array([1, 1, 1, 1])
    slowdown = np.array([1, 0.6231506849,1,1])
    normalized_latency_dataflow = slowdown*normalized_latency
    normalized_latency_bank_conflict_slow = (np.ones([slowdown.size])-slowdown)*normalized_latency

    utilization = np.array([0.5, 0.6231506849, 1, 1])

    barWidth = 0.7
    # the first subplot
    ax0 = plt.subplot(gs[0])
    # log scale for axis Y of the first subplot
    # ax0.set_yscale("log")
    # plt.bar(x[:-1], normalized_pj_compute[:-1], width=barWidth, color=color_list[0])
    # plt.bar(x[-1], normalized_pj_compute[-1], width=barWidth, color=color_list[3])
    plt.bar(x, normalized_pj_compute, width=barWidth, color=color_list[0])

    for i in range(x.size):
        ax0.text(x[i]-0.46, normalized_pj_compute[i]*1.02, f'{(normalized_pj_compute[i]):0.2f}x', fontsize=SMALL_SIZE)
    # plt.ylim((min(normalized_pj_compute)*0.8,max(normalized_pj_compute)*1.2))
    # plt.ylim((min(normalized_pj_compute)*0.8,max(normalized_pj_compute)*1.2))
    plt.yticks([1, 4, 7], fontsize=MEDIUM_SIZE)
    plt.ylim((0, 7.2))

    # the second subplot
    # shared axis X
    ax1 = plt.subplot(gs[3], sharex = ax0)

    data = []
    # plt.bar(x[:-1], normalized_latency[:-1], width=barWidth, color=color_list[0])
    # plt.bar(x[-1], normalized_latency[-1], width=barWidth, color=color_list[3])
    for i in range(x.size-1):
        plt.bar(x[i], normalized_latency_dataflow[i], width=barWidth, color=color_list[0], alpha=0.6)
        plt.bar(x[i], normalized_latency_bank_conflict_slow[i], width=barWidth, bottom=normalized_latency_dataflow[i], color=color_list[3])
    plt_handler = []
    plt_handler.append(plt.bar(x[-1], normalized_latency_dataflow[-1], width=barWidth, color=color_list[0], alpha=0.6))
    plt_handler.append(plt.bar(x[-1], normalized_latency_bank_conflict_slow[-1], width=barWidth, bottom=normalized_latency_dataflow[-1], color=color_list[3]))
    # plt.legend(plt_handler, [r"Dataflow Latency", r"STALL from bank conflict"], ncol=1, fontsize=13)

    for i in range(x.size):
        ax1.text(x[i]-0.46, normalized_latency[i]*1.03, f'{(normalized_latency[i]):0.2f}x', fontsize=SMALL_SIZE)
    # plt.xticks([0, 1, 2, 3, 4, 5, 6], ["Medusa\nHWC_C32","Simba\nHWC_C32","Eyeriss\nHWC_C32","Sigma\nHWC_W32","Sigma\nHWC_C32","Sigma\nHWC_C8W4",r"LAMBDA"], rotation = 90,  fontsize=MEDIUM_SIZE)
    # plt.xticks([0, 1, 2, 3], ["Simba  \n        ","Eyeriss\n        ", "Sigma  \n        ",r"LAMBDA"], rotation = 90,  fontsize=MEDIUM_SIZE)
    plt.xticks([0, 1, 2, 3], ["NVDLA-like\n        ","Eyeriss-like\n        ","SIGMA-like\n        ", r"FEATHER"], rotation = 90,  fontsize=MEDIUM_SIZE)
    # plt.xticks([0, 1, 2, 3, 4, 5], ["Simba\n        ","Eyeriss\n        ","Sigma\n        ","Sigma\n        ","Sigma\n        ",r"LAMBDA"], rotation = 90,  fontsize=MEDIUM_SIZE)
    plt.ylim((0, 3.3))
    plt.yticks([1, 2, 3], fontsize=MEDIUM_SIZE)
    for i in range(len(utilization)):
        ax1.text(i+0.25, -0.9, gemm_layout_list[i], rotation = 90, fontsize=SMALL_SIZE, color="red", horizontalalignment='center', verticalalignment='center')

    for i in range(len(utilization)):
        ax1.text(i, 0.5, f'\n{(utilization[i])*100:0.0f}%', rotation = 0, fontsize=SMALL_SIZE, color="black", horizontalalignment='center', verticalalignment='center')


    #####################
    ## ResNet50
    #####################

    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    normalized_pj_compute = np.array([1.302765933, 3.089171941, 1.092913227, 1.460550483, 1.989101158, 1.897094523, 2.201718716, 2.201718716, 1])
    normalized_latency = np.array([2, 1.26542556, 1.005442722, 1.027111789, 1.698392499, 1.005442722, 1.145923028, 1.145923028, 1])
    reorder = np.array([1, 1, 1, 1, (normalized_latency[-5]-1)/normalized_latency[-5], 1, 1, 1, 1])
    slowdown = np.array([1, 0.790247986, 1, 1, 1, 0.9952830189, 0.9009433962, 0.9009433962, 1])
    normalized_latency_dataflow = slowdown*normalized_latency
    normalized_latency_bank_conflict_slow = (np.ones([slowdown.size])-slowdown)*normalized_latency

    utilization = np.array([0.5, 0.8349056604, 0.99452830189, 0.9766037736, 1, 0.9945867411, 0.9009433962, 0.9009433962, 1])

    # the first subplot
    ax2 = plt.subplot(gs[1], sharey=ax0)
    # log scale for axis Y of the first subplot
    # ax2.set_yscale("log")
    # plt.bar(x[:-1], normalized_pj_compute[:-1], width=barWidth, color=color_list[0])
    # plt.bar(x[-1], normalized_pj_compute[-1], width=barWidth, color=color_list[3])

    plt.bar(x, normalized_pj_compute, width=barWidth, color=color_list[0])

    for i in range(x.size):
        ax2.text(x[i]-0.45, normalized_pj_compute[i]*1.05, f'{(normalized_pj_compute[i]):0.2f}x', fontsize=SMALL_SIZE)
    # plt.ylim((min(normalized_pj_compute)*0.8,max(normalized_pj_compute)*1.2))
    # plt.ylim((min(normalized_pj_compute)*0.8,max(normalized_pj_compute)*1.2))
    # plt.yticks([1, 3, 5], fontsize=MEDIUM_SIZE)

    # the second subplot
    # shared axis X
    ax3 = plt.subplot(gs[4], sharex = ax2, sharey=ax1)


    data = []
    # plt.bar(x[:-1], normalized_latency[:-1], width=barWidth, color=color_list[0])
    # plt.bar(x[-1], normalized_latency[-1], width=barWidth, color=color_list[3])
    for i in range(x.size-1):
        plt.bar(x[i], normalized_latency_dataflow[i], width=barWidth, color=color_list[0], alpha=0.6)
        plt.bar(x[i], normalized_latency_bank_conflict_slow[i], width=barWidth, bottom=normalized_latency_dataflow[i], color=color_list[3])
    plt_handler = []
    plt_handler.append(plt.bar(x[-1], normalized_latency_dataflow[-1], width=barWidth, color=color_list[0], alpha=0.6))
    plt_handler.append(plt.bar(x[-1], normalized_latency_bank_conflict_slow[-1], width=barWidth, bottom=normalized_latency_dataflow[-1], color=color_list[3]))
    ax3.text(x[-5], (1-reorder[-5])*normalized_latency_dataflow[-5], f'{(reorder[-5])*100:0.0f}%\n', rotation = 0, fontsize=SMALL_SIZE, color="k", horizontalalignment='center', verticalalignment='center')
    plt_handler.append(plt.bar(x[-5], normalized_latency_dataflow[-5]-1+normalized_latency_bank_conflict_slow[-5], width=barWidth, bottom=1-normalized_latency_bank_conflict_slow[-5], color=color_list[1]))
    plt.legend(plt_handler, [r"Dataflow Latency", r"STALL from bank conflict"], ncol=2, fontsize=SMALL_SIZE)

    for i in range(x.size):
        ax3.text(x[i]-0.45, normalized_latency[i]*1.05, f'{(normalized_latency[i]):0.2f}x', fontsize=SMALL_SIZE)
    # plt.xticks([0, 1, 2, 3, 4, 5, 6], ["Medusa\nHWC_C32","Simba\nHWC_C32","Eyeriss\nHWC_C32","Sigma\nHWC_W32","Sigma\nHWC_C32","Sigma\nHWC_C8W4",r"LAMBDA"], rotation = 90,  fontsize=MEDIUM_SIZE)
    # plt.xticks([0, 1, 2, 3, 4, 5, 6], ["Simba  \n        ","Eyeriss\n        ","Sigma  \n        ","Sigma  \n        ","Sigma  \n        ","Sigma  \n        ",r"LAMBDA"], rotation = 90,  fontsize=MEDIUM_SIZE)
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], ["NVDLA-like\n        ","Eyeriss-like\n        ","SIGMA-like\n        ","SIGMA-like\n        ","SIGMA-like\n        ","Medusa-like\n        ","MTIA-like\n        ", "TPU-like\n        ", r"FEATHER"], rotation = 90,  fontsize=MEDIUM_SIZE)
    # plt.xticks([0, 1, 2, 3, 4, 5], ["Simba\n        ","Eyeriss\n        ","Sigma\n        ","Sigma\n        ","Sigma\n        ",r"LAMBDA"], rotation = 90,  fontsize=MEDIUM_SIZE)
    plt.ylim((0, 3.3))
    plt.yticks([1, 2, 3], fontsize=MEDIUM_SIZE)
    for i in range(len(layout_list)):
        ax3.text(i+0.25, -1.5, layout_list[i], rotation = 90, fontsize=SMALL_SIZE, color="red", horizontalalignment='center', verticalalignment='center')

    # for i in range(len(utilization)):
    #     ax3.text(i, 0.5, f'utilize\n{(utilization[i])*100:0.0f}%\nPEs', rotation = 0, fontsize=11, color="black", horizontalalignment='center', verticalalignment='center')
    for i in range(len(utilization)):
        ax3.text(i, 0.5, f'\n{(utilization[i])*100:0.0f}%', rotation = 0, fontsize=SMALL_SIZE, color="black", horizontalalignment='center', verticalalignment='center')


    #####################
    ## MobV3
    #####################
    plt_handler = []
    normalized_pj_compute_mobv3 = np.array([1.352104301, 1.919504783, 1.292944762, 1.540942568, 1.663742376, 1.849868258, 2.064568072, 2.064568072, 1])
    normalized_latency_mobv3 = np.array([2.888026647, 1.871190102, 1.167502334, 1.066488025, 1.698392499, 1.177497698, 1.356102916, 1.356102916, 1])
    reorder_mobv3 = np.array([1, 1, 1, 1,  0.2351612903, 1, 1, 1, 1])
    slowdown_mobv3 = np.array([1, 0.7454677031,  0.9653368052, 0.9893548387, 1, 0.8943333333, 0.8226666667, 0.8226666667, 0.9836065574]) # use averagae value
    normalized_latency_dataflow_mobv3 = slowdown_mobv3 * normalized_latency_mobv3
    normalized_latency_bank_conflict_slow_mobv3 = (np.ones([slowdown_mobv3.size])-slowdown_mobv3)*normalized_latency_mobv3
    utilization_mobv3 = np.array([0.3861290323, 0.6006451613, 0.8724193548, 0.9225806452, 0.7648387097, 0.8796721311, 0.8091803279, 0.8091803279, 0.9836065574]) #  use average value
    # utilization_mobv3 = np.array([0.3403943539, 0.5257149158, 0.7782450996, 0.8411937817, 0.9187971334, 0.9832202586])
    ax4 = plt.subplot(gs[2], sharey = ax2)
    # log scale for axis Y of the first subplot
    # ax2.set_yscale("log")

    plt.bar(x, normalized_pj_compute_mobv3, width=barWidth, color=color_list[0])

    for i in range(x.size):
        ax4.text(x[i]-0.45, normalized_pj_compute_mobv3[i]*1.05, f'{(normalized_pj_compute_mobv3[i]):0.2f}X', fontsize=SMALL_SIZE)
    # plt.ylim((0, max(normalized_pj_compute)*1.2))
    # plt.ylim((min(normalized_pj_compute_mobv3)*0.8,max(normalized_pj_compute)*1.2))
    # plt.yticks([1, 3, 6], fontsize=MEDIUM_SIZE)
    # plt.ylim((0, 6.5))
    ax5 = plt.subplot(gs[5], sharey = ax3)

    # log scale for axis Y of the first subplot
    # ax2.set_yscale("log")
    # plt.bar(x[:-1], normalized_latency_mobv3[:-1], width=barWidth, color=color_list[0])
    # plt.bar(x[-1], normalized_latency_mobv3[-1], width=barWidth, color=color_list[3])
    for i in range(x.size-1):
        plt.bar(x[i], normalized_latency_dataflow_mobv3[i], width=barWidth, color=color_list[0], alpha=0.6)
        plt.bar(x[i], normalized_latency_bank_conflict_slow_mobv3[i], width=barWidth, bottom=normalized_latency_dataflow_mobv3[i], color=color_list[3])

    plt.bar(x[-1], normalized_latency_dataflow_mobv3[-1], width=barWidth, color=color_list[0], alpha=0.6)
    plt.bar(x[-1], normalized_latency_bank_conflict_slow_mobv3[-1], width=barWidth, bottom=normalized_latency_dataflow_mobv3[-1], color=color_list[3])
    # plt.annotate(text="", xy=(x[-5],(1-reorder_mobv3[-5])*normalized_latency_mobv3[-5]), xytext=(x[-5],normalized_latency_mobv3[-5]+0.05), color="red", arrowprops=dict(arrowstyle='|-|', color=color_list[1]))
    # ax5.text(x[-5]+0.1, (1-reorder_mobv3[-5])*normalized_latency_mobv3[-5]-0.25, f'reorder cost        \n {(reorder_mobv3[-5])*100:0.0f}%\n', rotation = 0, fontsize=SMALL_SIZE, color="red", horizontalalignment='center', verticalalignment='center')
    ax5.text(x[-5], (1-reorder_mobv3[-5])*normalized_latency_mobv3[-5]-0.25, f'{(reorder_mobv3[-5])*100:0.0f}%\n', rotation = 0, fontsize=SMALL_SIZE, color="k", horizontalalignment='center', verticalalignment='center')
    plt_handler.append(plt.bar(x[-5], normalized_latency_dataflow_mobv3[-5]-1+normalized_latency_bank_conflict_slow_mobv3[-5], width=barWidth, bottom=1-normalized_latency_bank_conflict_slow_mobv3[-5], color=color_list[1]))
    # plt.annotate(text="", xy=(x[-2],(1-reorder_mobv3[-2])*normalized_latency_mobv3[-2]), xytext=(x[-2],normalized_latency_mobv3[-2]), color="red", arrowprops=dict(arrowstyle='<->, head_width=0.2', color="red"))
    # plt.annotate(text="", xy=(x[-2],(1-reorder_mobv3[-2])*normalized_latency_mobv3[-2]), xytext=(x[-2],normalized_latency_mobv3[-2]), color="red", arrowprops=dict(arrowstyle='<->, head_width=0.2', color="red"))
    plt.legend(plt_handler, [r"Off-chip Reordering Cost"], ncol=1, fontsize=SMALL_SIZE)

    for i in range(x.size):
        ax5.text(x[i]-0.45, normalized_latency_mobv3[i]*1.05, f'{(normalized_latency_mobv3[i]):0.2f}X', fontsize=SMALL_SIZE)
    plt.ylim((0, 7))

    for i in range(len(utilization_mobv3)):
        ax5.text(i, 0.5, f'\n{(utilization_mobv3[i])*100:0.0f}%', rotation = 0, fontsize=SMALL_SIZE, color="black", horizontalalignment='center', verticalalignment='center')

    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], ["NVDLA-like\n        ","Eyeriss-like\n        ","SIGMA-like\n        ","SIGMA-like\n        ","SIGMA-like\n        ","Medusa-like\n        ","MTIA-like\n        ", "TPU-like\n        ", r"FEATHER"], rotation = 90,  fontsize=MEDIUM_SIZE)
    for i in range(len(layout_list)):
        ax5.text(i+0.25, -1.5, layout_list[i], rotation = 90, fontsize=SMALL_SIZE, color="red", horizontalalignment='center', verticalalignment='center')
    plt.ylim((0, 3.4))
    plt.yticks([1, 2, 3], fontsize=MEDIUM_SIZE)
    ax0.set_ylabel("Norm. pJ/MAC", fontsize=MEDIUM_SIZE)
    ax1.set_ylabel("Norm. Lat.", fontsize=MEDIUM_SIZE)
    ax1.set_xlabel("Bert", fontsize=MEDIUM_SIZE)
    ax3.set_xlabel("ResNet-50", fontsize=MEDIUM_SIZE)
    ax5.set_xlabel("MobileNet-V3", fontsize=MEDIUM_SIZE)
    plt.setp(ax0.get_xticklabels(), visible=False)
    # plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    plt.setp(ax4.get_xticklabels(), visible=False)
    plt.setp(ax4.get_yticklabels(), visible=False)
    plt.setp(ax5.get_yticklabels(), visible=False)
    # yticks = ax3.yaxis.get_major_ticks()
    # yticks[-1].label1.set_visible(False)
    # yticks2 = ax5.yaxis.get_major_ticks()
    # yticks2[-1].label1.set_visible(False)
    # plt.legend(plt_handler, [r"Dataflow Latency", r"STALL from bank conflict"], ncol=1, fontsize=13)

    # remove vertical gap between subplots
    plt.subplots_adjust(hspace=.0)
    plt.subplots_adjust(wspace=.0)
    #
    plt.savefig(r'LAMBDA_Cmp_SotA.pdf', bbox_inches="tight", transparent=True)



def figure_14():
    import matplotlib.pyplot as plt
    import numpy as np

    import matplotlib.patches as patches
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    # from matplotlib._png import read_png

    # need matploblib-3.1.3
    
    import numpy as np

    # im = read_png('/home/jimmy/MAERI/Fast_Switch/Figure_Drawer/lambda/lambda_die_photo.png')
    # imagebox_python = OffsetImage(im, zoom=.2)
    xy = [5, 10]

    # multiplier_cost_lambda=[0.56782,0.64078,0.15456,0.15456,0.16603,0.64675,0.2036,0.2036,0.29879,0.2036,1.12808,0.78754]

    # resources_breakdown = [[99789.86, 2375.95, 20906.83, 24811.29, 508929.49, 332836.94, 0.00],
    #  [243297.06,135281.66,5231.52,24186.82,0.00,17685.36,5168.02],
    #  [243297.06, 16384, 5231.52, 24186.82, 0, 0, 0]]
    resources_breakdown = [
    [243297.06, 140513.18, 24186.82, 0, 0, 0],
    [99789.86, 23282.78, 24811.29, 508929.49, 332836.94, 0.00],
    [243297.06, 140513.18, 24186.82, 0, 17685.36, 5168.02]
    ]

    resources_breakdown = np.array(resources_breakdown)
    resources_breakdown = resources_breakdown / 1000000
    # 45577.73,103219.20	89600.00	77403.80	0.00	17035.96	5168.02


    color_list = ["#6C8EBF", "#82B366", "#D79B00",  "#B85450", "#9673A6", "#B46504", "#D6B656","#23445D"]
    patterns = [ "//" , "\\\\" , "oo" , "*" , "." , "xx", "o", "O", "o-", " " ]
    # create data
    fig = plt.figure(figsize=[7,5.5])
    ax = plt.subplot(111)

    plt_handler = []
    barWidth = 0.6

    acu_matrix = []
    acu_matrix.append(resources_breakdown[0][0])
    for j in range(len(resources_breakdown[0])-1):
        acu_matrix.append(acu_matrix[-1] + resources_breakdown[0][j+1])
    max_range = np.max(acu_matrix)
    max_range1 = np.max(acu_matrix)

    for j in range(len(resources_breakdown[0])-1, -1, -1):
        if j==0:
            plt_handler.append(plt.bar(0, resources_breakdown[0][j], width=barWidth, bottom=0, color=color_list[j], edgecolor="white", hatch=patterns[j]))
        else:
            plt_handler.append(plt.bar(0, resources_breakdown[0][j], width=barWidth, bottom=acu_matrix[j-1], color=color_list[j], edgecolor="white", hatch=patterns[j]))

    acu_matrix = []
    acu_matrix.append(resources_breakdown[1][0])
    for j in range(len(resources_breakdown[1])-1):
        acu_matrix.append(acu_matrix[-1] + resources_breakdown[1][j+1])
    max_range_2 = np.max(acu_matrix)
    max_range = np.max([max_range_2,max_range])

    for j in range(len(resources_breakdown[0])):
        if j==0:
            plt.bar(1, resources_breakdown[1][j], width=barWidth, bottom=0, color=color_list[j], edgecolor="white", hatch=patterns[j])
        else:
            plt.bar(1, resources_breakdown[1][j], width=barWidth, bottom=acu_matrix[j-1], color=color_list[j], edgecolor="white", hatch=patterns[j])


    acu_matrix = []
    acu_matrix.append(resources_breakdown[2][0])
    for j in range(len(resources_breakdown[2])-1):
        acu_matrix.append(acu_matrix[-1] + resources_breakdown[2][j+1])
    max_range_3 = np.max(acu_matrix)
    max_range = np.max([max_range_3,max_range])

    for j in range(len(resources_breakdown[0])):
        if j==0:
            plt.bar(2, resources_breakdown[2][j], width=barWidth, bottom=0, color=color_list[j], edgecolor="white", hatch=patterns[j])
        else:
            plt.bar(2, resources_breakdown[2][j], width=barWidth, bottom=acu_matrix[j-1], color=color_list[j], edgecolor="white", hatch=patterns[j])

    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 22

    plt.ylabel(r"Area (LAMBDA v.s. SIGMA) $mm^2$", fontsize=MEDIUM_SIZE)
    # plt.xlabel(r"Number of PEs", fontsize=BIGGER_SIZE)
    # plt.xlabel(r"$Log_2(N)$ for $N\times N$ PEs", fontsize=BIGGER_SIZE)
    # plt.xticks(x, ["A", "B", "C", "D", "E", "F"], fontsize=MEDIUM_SIZE)
    # plt.xticks(x, ["A", "B", "C", "D", "E", "F"], fontsize=MEDIUM_SIZE)
    # plt.title("Scores by Teams in 4 Rounds")

    # plt.legend(["zcu104", 'Alevo U280'])#, fontsize=SMALL_SIZE)
    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.ylim([0, max_range*1.05])
    plt.xticks([0,1,2],[ "SIMBA-like-256", "SIGMA-256","FEATHER-256"], fontsize=19)

    space = 0
    # plt.legend(plt_handler, ["no layout constraints", "CSWTHT", "CTWSHT", "CTWTHS", "CSWSHT", "CSWTHS", "CTWSHS", "CSWSHS", "CTWSHTD", "CTWTHSD"], loc='upper center', ncol=5, fontsize=SMALL_SIZE)
    # plt.legend(plt_handler, ["no layout constraints", "CX32", "WX32", "HX32", "CX4WX8", "CX4HX8", "WX4HX8", "WX4HX8", "WX32D", "HX32D"], loc='upper center', ncol=5, fontsize=SMALL_SIZE)
    plt.legend(plt_handler, ["Comp. NoC", "Redn. NoC", "Dist. NoC", "Controller", "local mem.", "MAC" ], loc='upper center', ncol=3, bbox_to_anchor=(0.4, 1.25), columnspacing=0.5, labelspacing=space, fontsize=SMALL_SIZE)
    # plt.legend(plt_handler, ["Comp. NoC", "Redn. NoC", "Dist. NoC", "Controller", "local reduction", "local mem.", "MAC" ], loc='best', fontsize=15)
    # plt.legend(plt_handler, ["MAC", "local mem.", "local reduction", "Controller", "Dist. NoC", "Red. NoC", "Comp. NoC"], loc='best', fontsize=15)
    # plt.legend(plt_handler, ["MAC", "local mem.", "local reduction", "Controller", "Dist. NoC", "Red. NoC", "Comp. NoC"], loc='best', ncol=1, fontsize=13)
    # Conv->BN->ReLU->MaxPooling 
    style = "Simple, tail_width=2, head_width=8, head_length=8"

    kw = dict(arrowstyle=style, color="k")
    kw1 = dict(arrowstyle=style, color="red")
    plt.gca().add_patch(patches.FancyArrowPatch((0, max_range1), (1, max_range_2), connectionstyle="arc3,rad=-.0", **kw1))
    kw2 = dict(arrowstyle=style, color="green")
    plt.gca().add_patch(patches.FancyArrowPatch((1, max_range_2), (2, max_range_3), connectionstyle="arc3,rad=+.01", **kw2))
    plt.text(1.3,max_range_3*1.1, f"only {(max_range_3)/max_range_2*100:0.0f}% area", color='green', fontsize=MEDIUM_SIZE)
    # plt.text(1.5,(max_range_2+15*max_range_3)/16, f"than SIMBA", color='red', fontsize=MEDIUM_SIZE)

    # plt.text(-0.8,(max_range1+3*max_range_2)/4.1, f"${max_range1/max_range_2:0.2f}$X reduction to ", color='red', fontsize=MEDIUM_SIZE)
    plt.text(-0.18,max_range_3*1.25, f"{max_range_2/max_range1:0.2f}X area", color='red', fontsize=MEDIUM_SIZE)
    # plt.text(0.43,0.95, f"16x16 LAMBDA", color='red', fontsize=MEDIUM_SIZE)
    plt.text(1.57,0.6, f" Die Photo", color='blue', fontsize=MEDIUM_SIZE)

    kw3 = dict(arrowstyle=style, color="purple")
    plt.gca().add_patch(patches.FancyArrowPatch((0, max_range1), (2, max_range_3), connectionstyle="arc3,rad=-.06", **kw3))
    plt.text(-0.13,max_range_3*1.05, f"{max_range_3/max_range1:0.2f}X area", color='purple', fontsize=MEDIUM_SIZE)


    # ab_pythonlogo = AnnotationBbox(imagebox_python, [1.75, 0.94], frameon=False, xybox=(30., -30.), boxcoords='offset points')
    # ax.add_artist(ab_pythonlogo)

    plt.text(-0.6,1.05, f"BIRRD (4% area and 3.3% power of entire die)", color='blue', fontsize=SMALL_SIZE)
    # plt.text(0.85,0.7, f"", color='red',  fontsize=MEDIUM_SIZE)

    plt.savefig('resource_breakdown.pdf', bbox_inches="tight", transparent=True) 

if __name__ == "__main__":
    figure_12()
    figure_13()
    figure_14()
