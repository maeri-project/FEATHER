
# ====================================================================
#  You can add more layer shapes by creating new cnn_layers variables
# ====================================================================
# W, H, C, N, M, S, R, Wpad, Hpad, Wstride, Hstride



# ---------- BELOW ARE THE PROVIDED SHAPES (CONV LAYERS of 3 DNN Models) --------------------
# Alex Net w/o grouping specified in http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture9.pdf
# W, H, C, N, M, S, R, Wpad, Hpad, Wstride, Hstride
# cnn_layers = [
#     (227, 227, 3, 1, 96, 11, 11, 1, 1, 4, 4),
#     (27, 27, 96, 1, 256, 5, 5, 2, 2, 1, 1),
#     (13, 13, 256, 1, 384, 3, 3, 1, 1, 1, 1),
#     (13, 13, 384, 1, 384, 3, 3, 1, 1, 1, 1),
#     (13, 13, 384, 1, 256, 3, 3, 1, 1, 1, 1),
#     ]

# ResNet50 Net Specified in []
# M, N, K
bert = [
    (512, 768, 768),
    (512, 768, 3072),
    (512, 3072, 768)]

# net_dim_list = [resnet50, mobv3, vgg1, vgg2, testnet]
net_dim_list = [bert]
net_name_list = ["bert"]