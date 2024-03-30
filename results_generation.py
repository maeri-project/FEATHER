def figure_2():
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib import gridspec
    import matplotlib.patches as patches

    ##### Default Parameters
    color_list = ["#6C8EBF", "#82B366", "#F2CE61", "#B85450", "#9673A6", "#B46504","#D79B00",  "#D6B656","#23445D"]
    # Style for the added patch
    style = "Simple, tail_width=0.5, head_width=4, head_length=8"
    patterns = [ "//" , "\\\\" , "" , "." , "*"  , "xx", "o", "O", "o-", " " ]
    kw = dict(arrowstyle=style, color=color_list[0])

    image_upper_bound = 64
    image_lower_bound = -1
    plt.figure(figsize=[13,3])

    # set height ratios for subplots
    gs = gridspec.GridSpec(1, 2)

    ## the first plot
    normalized_latency = [[4.571428571, 2, 1.142857143, 1.5623489], [1,1,1,1], [8, 2, 4, 2.052078146], [1,1,1,1]]
    x = [i for i in range(len(normalized_latency[1]))]
    # Create figure
    ax0 = plt.subplot(gs[0])
    barWidth=0.24
    ax0.set_yscale('log', base=2)

    plt_handler = []
    evaluate_output_stationary_dataflow_under_layouts = [[32.00000217,4.571428571,4.571428571,4.571428571,4.571428571,1.306122449,7.836734694,4.571428571,4.571428571],[1,2,4,1,1.142857143,1,1,2,4],[1,1.142857143,1.142857143,1,1.142857143,1.142857143,1,1.142857143,1.142857143],[2.191738189,1.5623489,1.87627021,1.472395222,1.50585431,1.354259134,1.336217554,1.712129872,2.152328168]]
    evaluate_theoretical_results_on_layouts = [[8,1,8,1.5,8,2.5,2.5,1,8],[2,64,64,16,16,64,16,64,64],[4,16,128,4,32,32,8,16,128],[1.51297908,17.25069741,23.04760772,4.639468163,6.071347727,16.93041947,4.638380858,17.25069741,23.04760772]]

    evaluate_theoretical_results_on_layouts_pos=[]
    for j in range(len(normalized_latency)):
        for i in range(len(x)):
            if i == 0:
                if j == 2:
                    evaluate_theoretical_results_on_layouts_pos.append(x[i]-3*barWidth/4+j/2*barWidth)
                    plt_handler.append(plt.bar(x[i]-3*barWidth/4+j/2*barWidth,  np.mean(evaluate_theoretical_results_on_layouts[i][:]), width=1/2*barWidth, bottom=0, color=color_list[j],  edgecolor="white", hatch=patterns[j]))
                elif j == 0:
                    plt_handler.append(plt.bar(x[i]-3*barWidth/4+j/2*barWidth,  np.mean(evaluate_output_stationary_dataflow_under_layouts[i][:]), width=1/2*barWidth, bottom=0, color=color_list[j],  edgecolor="white", hatch=patterns[j]))
                else:
                    plt_handler.append(plt.bar(x[i]-3*barWidth/4+j/2*barWidth,  normalized_latency[j][i], width=1/2*barWidth, bottom=0, color=color_list[j],  edgecolor="white", hatch=patterns[j]))
            else:
                if j == 2:
                    evaluate_theoretical_results_on_layouts_pos.append(x[i]-3*barWidth/4+j/2*barWidth)
                    plt.bar(x[i]-3*barWidth/4+j/2*barWidth, np.mean(evaluate_theoretical_results_on_layouts[i][:]), width=1/2*barWidth, bottom=0, color=color_list[j],  edgecolor="white", hatch=patterns[j])
                elif j == 0:
                    plt.bar(x[i]-3*barWidth/4+j/2*barWidth,  np.mean(evaluate_output_stationary_dataflow_under_layouts[i][:]), width=1/2*barWidth, bottom=0, color=color_list[j],  edgecolor="white", hatch=patterns[j])
                else:
                    plt.bar(x[i]-3*barWidth/4+j/2*barWidth,  normalized_latency[j][i], width=1/2*barWidth, bottom=0, color=color_list[j],  edgecolor="white", hatch=patterns[j])

    SMALL_SIZE = 16
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 22
    plt.ylabel("Normalized Latency", fontsize=SMALL_SIZE)
    plt.rc('font',   size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes',   titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes',   labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('ytick',  labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('xtick',  labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.ylim(image_lower_bound, image_upper_bound)
    for j in range(len(normalized_latency[0])):
        if j >0:
            plt.annotate(text="", xy=(-0.03+j, np.mean(evaluate_theoretical_results_on_layouts[j][:])), xytext=(-0.03+j, normalized_latency[1][j]), color="red", arrowprops=dict(arrowstyle='<->, head_length=0.1', color=color_list[3]))
        plt.annotate(text="", xy=(+0.07+j, np.min(evaluate_theoretical_results_on_layouts[j][:])), xytext=(+0.07+j, np.min([np.max(evaluate_theoretical_results_on_layouts[j][:]), image_upper_bound])), color="k", arrowprops=dict(arrowstyle="|-|,widthA=0.2,widthB=0.2", color="k"))

    for j in range(len(normalized_latency[0])):
        plt_handler.append(plt.annotate(text="", xy=(-0.18+j, np.min(evaluate_output_stationary_dataflow_under_layouts[j][:])), xytext=(-0.18+j, np.min([np.max(evaluate_output_stationary_dataflow_under_layouts[j][:]), image_upper_bound])), color="k", arrowprops=dict(arrowstyle="|-|,widthA=0.2,widthB=0.2", color="k")))

    for j in range(len(normalized_latency[0])):
        if j == len(normalized_latency[0])-1 and np.mean(evaluate_theoretical_results_on_layouts[j][:]) >= image_upper_bound:
            plt.text(x[j]-1.2*barWidth, image_upper_bound-0.5, f'{np.min(evaluate_theoretical_results_on_layouts[j][:]):0.0f}~{np.max(evaluate_theoretical_results_on_layouts[j][:]):0.0f}X', rotation = 0, fontsize=14, color="red", horizontalalignment='center', verticalalignment='center')
        elif np.mean(evaluate_theoretical_results_on_layouts[j][:]) >= image_upper_bound:
            plt.text(x[j]+3/2*barWidth+0.1, image_upper_bound-0.5, f'{np.min(evaluate_theoretical_results_on_layouts[j][:]):0.0f}~{np.max(evaluate_theoretical_results_on_layouts[j][:]):0.0f}X', rotation = 0, fontsize=14, color="red", horizontalalignment='center', verticalalignment='center')
        elif j == 0:
            plt.text(x[j]+3/2*barWidth, np.mean(evaluate_theoretical_results_on_layouts[j][:]),  f'{np.min(evaluate_theoretical_results_on_layouts[j][:]):0.0f}~{np.max(evaluate_theoretical_results_on_layouts[j][:]):0.0f}X', rotation = 0, fontsize=14, color="red", horizontalalignment='center', verticalalignment='center')
        else:
            plt.text(x[j]+1/4*barWidth, np.mean(evaluate_theoretical_results_on_layouts[j][:])+5,  f'{np.min(evaluate_theoretical_results_on_layouts[j][:]):0.0f}~{np.max(evaluate_theoretical_results_on_layouts[j][:]):0.0f}X', rotation = 0, fontsize=14, color="red", horizontalalignment='center', verticalalignment='center')

    ax0.text(1.9, (normalized_latency[2][2] +normalized_latency[2][1]), f'theory practice gap', rotation = 90, fontsize=12, color="red", horizontalalignment='center', verticalalignment='center')
    ax0.text(1.75, (normalized_latency[2][2] +normalized_latency[2][1]), f'impact of layout', rotation = 90, fontsize=12, color="red", horizontalalignment='center', verticalalignment='center')

    for j in range(len(normalized_latency[0])):
        if normalized_latency[0][j] >= image_upper_bound:
            plt.text(x[j]+barWidth-0.55, image_upper_bound-0.5, f'{normalized_latency[0][j]/normalized_latency[1][j]:0.0f}X', rotation = 0, fontsize=14, color="red", horizontalalignment='center', verticalalignment='center')

    plt.gca().add_patch(patches.FancyArrowPatch((-3*barWidth/4, np.mean(evaluate_output_stationary_dataflow_under_layouts[0][:])), (-3*barWidth/4+1/2*barWidth, normalized_latency[1][0]),connectionstyle="arc3,rad=-.1", **kw))#**kw)
    ax0.text(-1*barWidth/5, 1.6*np.mean(evaluate_output_stationary_dataflow_under_layouts[0][:]), f'impact of dataflows', rotation = 90, fontsize=12, color="#10739E", horizontalalignment='center', verticalalignment='center')

    ax0.legend(plt_handler, ["output-stationary dataflow + fixed layout, error bar shows impact of diff layout (Fixed dataflow-layout)", "searched dataflow w/o layout consideration (theoretical best)", "evaluate theoretical best dataflow under various layouts, error bar shows impact of diff layouts (practice)", "flexible dataflow with data layout switching support (FEATHER, this work)", "error bar indicates the impact of various layout"], bbox_to_anchor=(-0.15, 1., .6, .6), loc='lower left', ncol=1, fontsize=14, labelspacing=.1)
    # ax0.legend(plt_handler, ["weight-stationary dataflow + fixed layout (HWC_C32), (Fixed dataflow-layout accelerator, 1X Area)", "searched dataflow w/o layout consideration (theoretical best)", "evaluate theoretical best dataflow under HWC_C32 layout (practice)", "flexible dataflow searched with fixed-layout concordance consideration (Practical SotA, 2.43X Area)", "flexible dataflow with data layout switching support (FEATHER, this work, 1.06X Area)"], bbox_to_anchor=(0., 1., 1., 1.), loc='lower left', ncol=1, fontsize=13, labelspacing=.1)
    plt.xticks([0, 1, 2, 3], ["Layer 1", "Layer 14", "Layer 41", "Full Model"], rotation = 0,  fontsize=16)
    plt.xlabel("ResNet-50", fontsize=16)

    # the second subplot
    # shared axis X
    ax1 = plt.subplot(gs[1], sharey = ax0)
    normalized_latency = [[1.523809524,1.219047619,7.5,2.727313071], [1,1,1,1], [7,22.5,2,5.431171764], [1,1,1,1]]
    ax1.set_yscale('log', base=2)
    x = [i for i in range(len(normalized_latency[1]))]
    # Create figure
    barWidth=0.3
    evaluate_output_stationary_dataflow_under_layouts = [[2.666666667,1.523809524,6.095238095,1.333333333,2.285714286,1.015873016,1.333333333,1.523809524,6.095238095],[1.6,1.219047619,1.828571429,1.142857143,1,1.523809524,1.066666667,1.219047619,1.828571429],[1.25,7.5,7.5,1.875,1.875,7.5,1.875,7.5,7.5],[2.266875368,2.727313071,2.902085967,1.690016571,1.685904118,2.242222371,1.704776098,2.727313071,2.902085967]]
    evaluate_theoretical_results_on_layouts = [[44,8,44,6,11,12,6,8,44],[22.5,12,60,6,15,16,8,12,60],[2,60,60,15,15,60,15,60,60],[5.431171764,7.986715466,12.09655517,3.763159658,4.85374675,6.543326775,3.531519746,8.367862654,12.50914648]]

    plt_handler = []
    evaluate_theoretical_results_on_layouts_pos=[]
    for j in range(len(normalized_latency)):
        for i in range(len(x)):
            if i == 0:
                if j == 2:
                    evaluate_theoretical_results_on_layouts_pos.append(x[i]-3*barWidth/4+j/2*barWidth)
                    plt_handler.append(plt.bar(x[i]-3*barWidth/4+j/2*barWidth,  np.mean(evaluate_theoretical_results_on_layouts[i][:]), width=1/2*barWidth, bottom=0, color=color_list[j],  edgecolor="white", hatch=patterns[j]))
                elif j == 0:
                    plt_handler.append(plt.bar(x[i]-3*barWidth/4+j/2*barWidth,  np.mean(evaluate_output_stationary_dataflow_under_layouts[i][:]), width=1/2*barWidth, bottom=0, color=color_list[j],  edgecolor="white", hatch=patterns[j]))
                else:
                    plt_handler.append(plt.bar(x[i]-3*barWidth/4+j/2*barWidth,  normalized_latency[j][i], width=1/2*barWidth, bottom=0, color=color_list[j],  edgecolor="white", hatch=patterns[j]))
            else:
                if j == 2:
                    evaluate_theoretical_results_on_layouts_pos.append(x[i]-3*barWidth/4+j/2*barWidth)
                    plt.bar(x[i]-3*barWidth/4+j/2*barWidth, np.mean(evaluate_theoretical_results_on_layouts[i][:]), width=1/2*barWidth, bottom=0, color=color_list[j],  edgecolor="white", hatch=patterns[j])
                elif j == 0:
                    plt.bar(x[i]-3*barWidth/4+j/2*barWidth,  np.mean(evaluate_output_stationary_dataflow_under_layouts[i][:]), width=1/2*barWidth, bottom=0, color=color_list[j],  edgecolor="white", hatch=patterns[j])
                else:
                    plt.bar(x[i]-3*barWidth/4+j/2*barWidth,  normalized_latency[j][i], width=1/2*barWidth, bottom=0, color=color_list[j],  edgecolor="white", hatch=patterns[j])

    plt.rc('font',   size=SMALL_SIZE)        # controls default text sizes
    plt.rc('axes',   titlesize=SMALL_SIZE)   # fontsize of the axes title
    plt.rc('axes',   labelsize=SMALL_SIZE)   # fontsize of the x and y labels
    plt.rc('ytick',  labelsize=SMALL_SIZE)   # fontsize of the tick labels
    plt.rc('xtick',  labelsize=SMALL_SIZE)   # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    geomean= [np.prod(normalized_latency[0])**(1/len(normalized_latency[0])), np.prod(normalized_latency[1])**(1/len(normalized_latency[1]))]
    plt.ylim(image_lower_bound, image_upper_bound)
    for j in range(len(normalized_latency[0])):
        if  np.mean(evaluate_theoretical_results_on_layouts[j][:]) >= image_upper_bound:
            plt.annotate(text="", xy=(-0.05+j, image_upper_bound), xytext=(-0.05+j, normalized_latency[1][j]), color="red", arrowprops=dict(arrowstyle='<->, head_length=0.1', color=color_list[3]))
            plt.annotate(text="", xy=(+0.07+j, np.min(evaluate_theoretical_results_on_layouts[j][:])), xytext=(+0.07+j, np.min([np.max(evaluate_theoretical_results_on_layouts[j][:]), image_upper_bound])), color="k", arrowprops=dict(arrowstyle="|-|,widthA=0.2,widthB=0.2", color="k"))
        else:
            plt.annotate(text="", xy=(-0.05+j, np.mean(evaluate_theoretical_results_on_layouts[j][:])), xytext=(-0.05+j, normalized_latency[1][j]), color="red", arrowprops=dict(arrowstyle='<->, head_length=0.1', color=color_list[3]))
            plt.annotate(text="", xy=(+0.07+j, np.min(evaluate_theoretical_results_on_layouts[j][:])), xytext=(+0.07+j, np.min([np.max(evaluate_theoretical_results_on_layouts[j][:]), image_upper_bound])), color="k", arrowprops=dict(arrowstyle="|-|,widthA=0.2,widthB=0.2", color="k"))

    for j in range(len(normalized_latency[0])):
        plt.annotate(text="", xy=(-0.22+j, np.min(evaluate_output_stationary_dataflow_under_layouts[j][:])), xytext=(-0.22+j, np.min([np.max(evaluate_output_stationary_dataflow_under_layouts[j][:]), image_upper_bound])), color="k", arrowprops=dict(arrowstyle="|-|,widthA=0.2,widthB=0.2", color="k"))

    for j in range(len(normalized_latency[0])):
        if normalized_latency[0][j] >= image_upper_bound:
            plt.text(x[j], image_upper_bound-0.5, f'{normalized_latency[0][j]/normalized_latency[1][j]:0.0f}X', rotation = 0, fontsize=14, color="red", horizontalalignment='center', verticalalignment='center')

    for j in range(len(normalized_latency[0])):
        if j == len(normalized_latency[0])-1 and np.mean(evaluate_theoretical_results_on_layouts[j][:]) >= image_upper_bound:
            plt.text(x[j]-1.2*barWidth, image_upper_bound-1, f'{np.min(evaluate_theoretical_results_on_layouts[j][:]):0.0f}~{np.max(evaluate_theoretical_results_on_layouts[j][:]):0.0f}X', rotation = 0, fontsize=14, color="red", horizontalalignment='center', verticalalignment='center')
        elif np.mean(evaluate_theoretical_results_on_layouts[j][:]) >= image_upper_bound:
            plt.text(x[j]+3/2*barWidth, image_upper_bound-0.5, f'{np.min(evaluate_theoretical_results_on_layouts[j][:]):0.0f}~{np.max(evaluate_theoretical_results_on_layouts[j][:]):0.0f}X', rotation = 0, fontsize=14, color="red", horizontalalignment='center', verticalalignment='center')
        else:
            plt.text(x[j]+1/4*barWidth+0.1, np.mean(evaluate_theoretical_results_on_layouts[j][:])+5,  f'{np.min(evaluate_theoretical_results_on_layouts[j][:]):0.0f}~{np.max(evaluate_theoretical_results_on_layouts[j][:]):0.0f}X', rotation = 0, fontsize=14, color="red", horizontalalignment='center', verticalalignment='center')

    plt.xticks([0, 1, 2, 3], ["Layer 7", "Layer 25", "Layer 40", "Full Model"], rotation = 0,  fontsize=16)
    plt.xlabel("MobileNet-V3", fontsize=16)

    ## Final Common Results
    plt.setp(ax1.get_yticklabels(), visible=False)
    plt.subplots_adjust(hspace=.0)
    plt.subplots_adjust(wspace=.0)
    plt.savefig('figure2.pdf', bbox_inches="tight", transparent=True) 
    plt.show()

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

    # print(f"latency saving ratio range over Xilinx DPU is [{1-np.max(np.array(acu_matrix_xilinx_dpu[-1])/np.array(acu_matrix[-1]))},  {1-np.min(np.array(acu_matrix_xilinx_dpu[-1])/np.array(acu_matrix[-1]))}]")
    # print(f"latency saving ratio range over Edge TPU is [{1-np.max(np.array(acu_matrix_edge_tpu[-1])/np.array(acu_matrix[-1]))},  {1-np.min(np.array(acu_matrix_edge_tpu[-1])/np.array(acu_matrix[-1]))}]")

    SMALL_SIZE = 16
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 22
    plt.xlabel(r"Layer ID in ResNet50", fontsize=SMALL_SIZE)
    plt.ylabel(r"Normalized $Log_2(Thrpt./PE)$", fontsize=15)
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    geomean= [np.prod(normalized_latency_speedup_ori[0])**(1/len(normalized_latency_speedup_ori[0])), np.prod(normalized_latency_speedup_ori[1])**(1/len(normalized_latency_speedup_ori[1])), np.prod(normalized_latency_speedup_ori[2])**(1/len(normalized_latency_speedup_ori[2]))]
    plt_handler.append(plt.plot([0, 52], [np.log(geomean[0]), np.log(geomean[0])], linewidth=2, color=color_list[0], linestyle="--")[0])#, marker = 'o'))
    plt_handler.append(plt.plot([0, 52], [np.log(geomean[1]), np.log(geomean[1])], linewidth=2, color=color_list[1], linestyle="--")[0])#, marker = 'o'))
    plt_handler.append(plt.plot([0, 52], [np.log(geomean[2]), np.log(geomean[2])], linewidth=2, color=color_list[2], linestyle="--")[0])#, marker = 'o'))
    plt.text(55.5, np.log(geomean[0]*0.9), f'{(geomean[0]):0.2f}X ',color=color_list[0], horizontalalignment='center')
    plt.text(55.5, np.log(geomean[1]*0.85), f'{(geomean[1]):0.2f}X ',color=color_list[1], horizontalalignment='center')
    plt.text(55.5, np.log(geomean[2]*1.1), f'{(geomean[2]):0.2f}X ', color=color_list[2],horizontalalignment='center')
    plt.legend(plt_handler, [r"Speedup Over Gemmini", r"Speedup Over Xilinx DPU", r"Speedup Over Edge TPU", r"Speedup Over Gemmini (GeoMean)", r"Speedup Over Xilinx DPU (GeoMean)", r"Speedup Over Edge TPU (GeoMean)"], ncol=2, fontsize=14, labelspacing = 0, columnspacing=0.5, frameon=False, loc='center', bbox_to_anchor=(0.5, 0.88),)

    plt.ylim([np.min(normalized_latency_speedup)*2, 4])
    plt.xlim([-1, 58])
    plt.savefig(r'figure12.pdf', bbox_inches="tight", transparent=True) 


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
    plt.bar(x, normalized_pj_compute, width=barWidth, color=color_list[0])

    for i in range(x.size):
        ax0.text(x[i]-0.46, normalized_pj_compute[i]*1.02, f'{(normalized_pj_compute[i]):0.2f}x', fontsize=SMALL_SIZE)
    plt.yticks([1, 4, 7], fontsize=MEDIUM_SIZE)
    plt.ylim((0, 7.2))

    # the second subplot
    # shared axis X
    ax1 = plt.subplot(gs[3], sharex = ax0)

    data = []
    for i in range(x.size-1):
        plt.bar(x[i], normalized_latency_dataflow[i], width=barWidth, color=color_list[0], alpha=0.6)
        plt.bar(x[i], normalized_latency_bank_conflict_slow[i], width=barWidth, bottom=normalized_latency_dataflow[i], color=color_list[3])
    plt_handler = []
    plt_handler.append(plt.bar(x[-1], normalized_latency_dataflow[-1], width=barWidth, color=color_list[0], alpha=0.6))
    plt_handler.append(plt.bar(x[-1], normalized_latency_bank_conflict_slow[-1], width=barWidth, bottom=normalized_latency_dataflow[-1], color=color_list[3]))

    for i in range(x.size):
        ax1.text(x[i]-0.46, normalized_latency[i]*1.03, f'{(normalized_latency[i]):0.2f}x', fontsize=SMALL_SIZE)
    plt.xticks([0, 1, 2, 3], ["NVDLA-like\n        ","Eyeriss-like\n        ","SIGMA-like\n        ", r"FEATHER"], rotation = 90,  fontsize=MEDIUM_SIZE)
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

    plt.bar(x, normalized_pj_compute, width=barWidth, color=color_list[0])

    for i in range(x.size):
        ax2.text(x[i]-0.45, normalized_pj_compute[i]*1.05, f'{(normalized_pj_compute[i]):0.2f}x', fontsize=SMALL_SIZE)

    # the second subplot
    # shared axis X
    ax3 = plt.subplot(gs[4], sharex = ax2, sharey=ax1)


    data = []
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
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], ["NVDLA-like\n        ","Eyeriss-like\n        ","SIGMA-like\n        ","SIGMA-like\n        ","SIGMA-like\n        ","Medusa-like\n        ","MTIA-like\n        ", "TPU-like\n        ", r"FEATHER"], rotation = 90,  fontsize=MEDIUM_SIZE)
    plt.ylim((0, 3.3))
    plt.yticks([1, 2, 3], fontsize=MEDIUM_SIZE)
    for i in range(len(layout_list)):
        ax3.text(i+0.25, -1.5, layout_list[i], rotation = 90, fontsize=SMALL_SIZE, color="red", horizontalalignment='center', verticalalignment='center')

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
    ax4 = plt.subplot(gs[2], sharey = ax2)
    # log scale for axis Y of the first subplot

    plt.bar(x, normalized_pj_compute_mobv3, width=barWidth, color=color_list[0])

    for i in range(x.size):
        ax4.text(x[i]-0.45, normalized_pj_compute_mobv3[i]*1.05, f'{(normalized_pj_compute_mobv3[i]):0.2f}X', fontsize=SMALL_SIZE)
    ax5 = plt.subplot(gs[5], sharey = ax3)

    # log scale for axis Y of the first subplot
    for i in range(x.size-1):
        plt.bar(x[i], normalized_latency_dataflow_mobv3[i], width=barWidth, color=color_list[0], alpha=0.6)
        plt.bar(x[i], normalized_latency_bank_conflict_slow_mobv3[i], width=barWidth, bottom=normalized_latency_dataflow_mobv3[i], color=color_list[3])

    plt.bar(x[-1], normalized_latency_dataflow_mobv3[-1], width=barWidth, color=color_list[0], alpha=0.6)
    plt.bar(x[-1], normalized_latency_bank_conflict_slow_mobv3[-1], width=barWidth, bottom=normalized_latency_dataflow_mobv3[-1], color=color_list[3])
    ax5.text(x[-5], (1-reorder_mobv3[-5])*normalized_latency_mobv3[-5]-0.25, f'{(reorder_mobv3[-5])*100:0.0f}%\n', rotation = 0, fontsize=SMALL_SIZE, color="k", horizontalalignment='center', verticalalignment='center')
    plt_handler.append(plt.bar(x[-5], normalized_latency_dataflow_mobv3[-5]-1+normalized_latency_bank_conflict_slow_mobv3[-5], width=barWidth, bottom=1-normalized_latency_bank_conflict_slow_mobv3[-5], color=color_list[1]))
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
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    plt.setp(ax4.get_xticklabels(), visible=False)
    plt.setp(ax4.get_yticklabels(), visible=False)
    plt.setp(ax5.get_yticklabels(), visible=False)
    # remove vertical gap between subplots
    plt.subplots_adjust(hspace=.0)
    plt.subplots_adjust(wspace=.0)
    #
    plt.savefig(r'figure13.pdf', bbox_inches="tight", transparent=True)

def figure_14_a():
    import matplotlib.pyplot as plt
    import numpy as np
    area_overhead = [
        [12105.70193,24806.75385,51080.39967,102299.1474,205687.4387],
        [15597.66591,35800.75779,79601.63354,173167.973,371737.798],
        [17714.59146,46406.30257,115536.9564,278054.7754,653109.6397]
    ]

    power_overhead = [
        [4.544,9.247,18.95,37.798,76.045],
        [6.436,15.056,33.976,74.748,162.224],
        [6.315,16.22,40.024,95.581,223.559]
    ]


    area_overhead= np.log2(area_overhead)

    color_list = ["#6C8EBF", "#82B366", "#D79B00",  "#B85450", "#9673A6", "#B46504", "#D6B656", "#23445D", "#56517E"]
    shape_list = ["o", "*", "X", "p", ">", "d", "s", "H", "<"]
    linestyle_list = ["solid", "dotted", "dashed", "dashdot"]
    patterns = [ "//" , "\\\\" , "oo" , "*" , "o-" , "xx", "o", "O", ".", " " ]

    x = [i for i in range(len(area_overhead[0]))]
    plt.figure(figsize=[7,5.5])
    ax = plt.subplot(111)

    barWidth = 0.5
    plt_handler = []

    for case_id in range(len(area_overhead)):
        plt_handler.append(plt.plot(x, area_overhead[case_id], marker=shape_list[case_id], markersize=10, color=color_list[case_id], linewidth=2, linestyle=linestyle_list[case_id])[0])

    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 22

    plt.ylabel(r"Area (Line) $Log_2(um^2)$", fontsize=BIGGER_SIZE)
    plt.xlabel(r"Number of Reduction Data", fontsize=BIGGER_SIZE)

    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.xticks(x,["16","32","64","128","256"], fontsize=MEDIUM_SIZE)


    ax2 = ax.twinx()
    shape_list = ["o", "*", "X", "p", ">", "d", "s", "H", "<"]
    legend_list = ["A", "B", "C", "D", "E", "F", "G", "H"]
    color_list = ["#B46504", "#23445D", "#D79B00", "#D6B656", "#9673A6", "#B85450", "#6C8EBF", "#82B366"]

    barWidth=0.3
    # multiple line plots
    for i in range(len(power_overhead[0])):
        if(i==0):
            plt_handler.append(ax2.bar(x[i]-2*barWidth/3, power_overhead[0][i], width=2/3*barWidth, bottom=0, color=color_list[0], edgecolor="white", hatch=patterns[0]))
            plt_handler.append(ax2.bar(x[i], power_overhead[1][i], width=2/3*barWidth, bottom=0, color=color_list[1], edgecolor="white", hatch=patterns[1]))
            plt_handler.append(ax2.bar(x[i]+2*barWidth/3, power_overhead[2][i], width=2/3*barWidth, bottom=0, color=color_list[2], edgecolor="white", hatch=patterns[2]))
        else:
            ax2.bar(x[i]-2*barWidth/3, power_overhead[0][i], width=2/3*barWidth, bottom=0, color=color_list[0], edgecolor="white", hatch=patterns[0])
            ax2.bar(x[i], power_overhead[1][i], width=2/3*barWidth, bottom=0, color=color_list[1], edgecolor="white", hatch=patterns[1])
            ax2.bar(x[i]+2*barWidth/3, power_overhead[2][i], width=2/3*barWidth, bottom=0, color=color_list[2], edgecolor="white", hatch=patterns[2])
    ax2.set_ylabel("Power (bar, mW)", fontsize=BIGGER_SIZE)
    plt.legend(plt_handler, ["ART(MAERI)", "FAN(SIGMA)", "BIRRD(FEATHER)","ART(MAERI)", "FAN(SIGMA)", "BIRRD(FEATHER)"], loc='best', ncol=1, fontsize=15)
    plt.savefig('figure14_a.pdf', bbox_inches="tight", transparent=True) 
    plt.show()


def figure_14_b():
    import matplotlib.pyplot as plt
    import numpy as np

    import matplotlib.patches as patches
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    import numpy as np
    xy = [5, 10]

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
    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.ylim([0, max_range*1.05])
    plt.xticks([0,1,2],[ "SIMBA-like-256", "SIGMA-256","FEATHER-256"], fontsize=19)

    space = 0
    plt.legend(plt_handler, ["Comp. NoC", "Redn. NoC", "Dist. NoC", "Controller", "local mem.", "MAC" ], loc='upper center', ncol=3, bbox_to_anchor=(0.4, 1.25), columnspacing=0.5, labelspacing=space, fontsize=SMALL_SIZE)
    # Conv->BN->ReLU->MaxPooling 
    style = "Simple, tail_width=2, head_width=8, head_length=8"

    kw = dict(arrowstyle=style, color="k")
    kw1 = dict(arrowstyle=style, color="red")
    plt.gca().add_patch(patches.FancyArrowPatch((0, max_range1), (1, max_range_2), connectionstyle="arc3,rad=-.0", **kw1))
    kw2 = dict(arrowstyle=style, color="green")
    plt.gca().add_patch(patches.FancyArrowPatch((1, max_range_2), (2, max_range_3), connectionstyle="arc3,rad=+.01", **kw2))
    plt.text(1.3,max_range_3*1.1, f"only {(max_range_3)/max_range_2*100:0.0f}% area", color='green', fontsize=MEDIUM_SIZE)
    plt.text(-0.18,max_range_3*1.25, f"{max_range_2/max_range1:0.2f}X area", color='red', fontsize=MEDIUM_SIZE)
    plt.text(1.57,0.6, f" Die Photo", color='blue', fontsize=MEDIUM_SIZE)

    kw3 = dict(arrowstyle=style, color="purple")
    plt.gca().add_patch(patches.FancyArrowPatch((0, max_range1), (2, max_range_3), connectionstyle="arc3,rad=-.06", **kw3))
    plt.text(-0.13,max_range_3*1.05, f"{max_range_3/max_range1:0.2f}X area", color='purple', fontsize=MEDIUM_SIZE)

    plt.text(-0.6,1.05, f"BIRRD (4% area and 3.3% power of entire die)", color='blue', fontsize=SMALL_SIZE)

    plt.savefig('figure14_b.pdf', bbox_inches="tight", transparent=True) 

if __name__ == "__main__":
    figure_2()
    figure_12()
    figure_13()
    figure_14_a()
    figure_14_b()
