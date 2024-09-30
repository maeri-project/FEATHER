/******************************************************************************
Copyright (c) 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*******************************************************************************/
/*
    Top Module:  Feather Top module
    Data:        Only data width matters.
    Format:      keeping the input format unchanged
    Timing:      Sequential Logic
    Reset:       Asynchronized Reset [Low Reset]

    Function:    Contains all the BIRRD, NEST, SRAMs and Feather controller instantiations
    Note:        This template is designed for FEATHER with DPE_COL_NUM>4. 
                 When DPE_COL_NUM=DPE_ROW_NUM=4, BIRRD only has "2log(DPE_COL_NUM) - 1" stages instead of "2log(DPE_COL_NUM)" stage.
*/

`timescale 1ns / 1ps

module feather_top #(
//    parameter DPE_COL_NUM                       =   64,
//    parameter DPE_ROW_NUM                       =   64,

   parameter DPE_COL_NUM                       =   16,
   parameter DPE_ROW_NUM                       =   16,

//    parameter DPE_COL_NUM                       =   32,
//    parameter DPE_ROW_NUM                       =   32,
    // parameter DPE_COL_NUM                       =   8,
    // parameter DPE_ROW_NUM                       =   8,
    parameter LOG2_DPE_COL_NUM                  =   $clog2(DPE_COL_NUM),
    parameter LOG2_DPE_ROW_NUM                  =   $clog2(DPE_ROW_NUM),

    parameter WEIGHTS_DATA_WIDTH                =   8*DPE_COL_NUM,                                  //
    parameter WEIGHTS_NUM_BANKS                 =   1,                                              //
    parameter WEIGHTS_SRAM_DATA_WIDTH           =   WEIGHTS_NUM_BANKS*WEIGHTS_DATA_WIDTH,
    parameter WEIGHTS_SRAM_BANK_ADDR_WIDTH      =   2,                                              //
    parameter WEIGHTS_SRAM_ADDR_WIDTH           =   WEIGHTS_NUM_BANKS*WEIGHTS_SRAM_BANK_ADDR_WIDTH, //

    parameter IACTS_DATA_WIDTH                  =   8,                                              //
    parameter IACTS_NUM_BANKS                   =   DPE_COL_NUM,                                    //
    parameter IACTS_SRAM_DATA_WIDTH             =   IACTS_NUM_BANKS*IACTS_DATA_WIDTH,
    parameter IACTS_SRAM_BANK_ADDR_WIDTH        =   2,                                              //
    parameter IACTS_SRAM_ADDR_WIDTH             =   IACTS_NUM_BANKS*IACTS_SRAM_BANK_ADDR_WIDTH,     //

    parameter SCALE_VALUE_WIDTH                 =   32,                                             //

    parameter PE_OUTPUT_WIDTH                   =   32,                                             //

    parameter OACTS_DATA_WIDTH                  =   8,                                              //

    parameter OUTBUF_DATA_WIDTH                 =   PE_OUTPUT_WIDTH,                                //
    parameter OUTBUF_NUM_BANKS                  =   DPE_COL_NUM,                                    //
    parameter OUTBUF_SRAM_DATA_WIDTH            =   OUTBUF_NUM_BANKS*OUTBUF_DATA_WIDTH,             //
    parameter OUTBUF_SRAM_BANK_ADDR_WIDTH       =   2,                                              //
    parameter OUTBUF_SRAM_ADDR_WIDTH            =   OUTBUF_NUM_BANKS*OUTBUF_SRAM_BANK_ADDR_WIDTH,   //

    parameter INSTR_SRAM_BANK_ADDR_WIDTH        =   2                                               //
)(

    clk                                     ,
    rst_n                                   ,

    i_feather_top_en                        ,

    // input data
    // ZP data
    i_iacts_zp                              ,
    i_iacts_zp_valid                        ,
    i_weights_zp                            ,
    i_weights_zp_valid                      ,

    //Weights
    i_weights_write_valid                   ,
    i_weights_write_data                    ,
    i_weights_write_addr                    ,
    i_weights_write_addr_end                ,

    //iActs
    i_iacts_write_valid                     ,
    i_iacts_write_data                      ,
    i_iacts_write_addr                      ,
    i_iacts_write_addr_end                  ,

    //instruction data
    i_instr_write_valid                     ,
    i_instr_write_data                      ,
    i_instr_write_addr                      ,

    i_all_buf_pingpong_config               ,
    o_outbuf_data_wr_rdy                    ,
    i_outbuf_wr_instr                       ,

    //Quantization value                    ,
    i_scale_val                             ,

    // output related
    i_oacts_read_valid                      ,
    o_oacts_read_data                       ,
    i_oacts_read_addr                       ,
    i_oacts_read_addr_end

);

    localparam NUM_STAGE                        =   2*(LOG2_DPE_COL_NUM);                               // 2_BIT_CMD * (Number of stages-1) *(birrd_INPUTS/2)
    localparam birrd_COMMAND_WIDTH_PER_ROW      =   2*NUM_STAGE ;                                       // 2_BIT_CMD * (Number of stages-1)
    localparam birrd_IN_COMMAND_WIDTH           =   birrd_COMMAND_WIDTH_PER_ROW * DPE_COL_NUM >> 1 ;    // 2_BIT_CMD * (Number of stages-1) *(birrd_INPUTS/2)
    localparam INSTR_SRAM_BANK_DATA_WIDTH       =   birrd_IN_COMMAND_WIDTH;                             //  instruction is only birrd instruction here
    localparam WEIGHTS_DEPTH                    =   DPE_ROW_NUM;                                        //
    localparam LOG2_WEIGHTS_DEPTH               =   LOG2_DPE_ROW_NUM;                                   //
    localparam PE_SEL_WIDTH                     =   LOG2_DPE_COL_NUM + LOG2_WEIGHTS_DEPTH;              //

    localparam IACTS_PINGPONG_CONFIG_WIDTH      =   4;
    localparam WEIGHTS_PINGPONG_CONFIG_WIDTH    =   4;
    localparam ALL_BUF_CONFIG_WIDTH             =   IACTS_PINGPONG_CONFIG_WIDTH  + WEIGHTS_PINGPONG_CONFIG_WIDTH;
    localparam OUTBUF_SRAM_INSTR_WIDTH          =   OUTBUF_SRAM_ADDR_WIDTH + (2*OUTBUF_NUM_BANKS); //  OUTBUF_NUM_BANKS*ADDR + (OUTBUF_NUM_BANKS ---for r_bypass_to_scale) + (OUTBUF_NUM_BANKS--- for r_mul_with_scale)


    /*
        ports
    */
    input                                               clk                         ;
    input                                               rst_n                       ;

    input                                               i_feather_top_en             ;

    input       [IACTS_DATA_WIDTH               -1: 0]  i_iacts_zp                  ;
    input                                               i_iacts_zp_valid            ;
    input       [WEIGHTS_DATA_WIDTH/DPE_COL_NUM -1: 0]  i_weights_zp                ;
    input                                               i_weights_zp_valid          ;

    input                                               i_weights_write_valid       ;
    input       [WEIGHTS_SRAM_DATA_WIDTH        -1: 0]  i_weights_write_data        ;
    input       [WEIGHTS_SRAM_BANK_ADDR_WIDTH   -1: 0]  i_weights_write_addr        ;
    input       [WEIGHTS_SRAM_BANK_ADDR_WIDTH   -1: 0]  i_weights_write_addr_end    ;

    input                                               i_iacts_write_valid         ;
    input       [IACTS_SRAM_DATA_WIDTH          -1: 0]  i_iacts_write_data          ;
    input       [IACTS_SRAM_BANK_ADDR_WIDTH     -1: 0]  i_iacts_write_addr          ;
    input       [IACTS_SRAM_BANK_ADDR_WIDTH     -1: 0]  i_iacts_write_addr_end      ;

    input                                               i_instr_write_valid         ;
    input       [INSTR_SRAM_BANK_DATA_WIDTH     -1: 0]  i_instr_write_data          ;
    input       [INSTR_SRAM_BANK_ADDR_WIDTH     -1: 0]  i_instr_write_addr          ;

    input       [ALL_BUF_CONFIG_WIDTH           -1: 0]  i_all_buf_pingpong_config   ;
    input       [OUTBUF_SRAM_INSTR_WIDTH        -1: 0]  i_outbuf_wr_instr           ;
    output      [OUTBUF_NUM_BANKS               -1: 0]  o_outbuf_data_wr_rdy        ;

    input       [SCALE_VALUE_WIDTH              -1: 0]  i_scale_val                 ;

    input                                               i_oacts_read_valid          ;
    output      [IACTS_SRAM_DATA_WIDTH          -1: 0]  o_oacts_read_data           ;
    input       [IACTS_SRAM_ADDR_WIDTH          -1: 0]  i_oacts_read_addr           ;
    input       [IACTS_SRAM_ADDR_WIDTH          -1: 0]  i_oacts_read_addr_end       ;

    /*
        feather Contoller signals
    */
    wire    [IACTS_SRAM_DATA_WIDTH              -1: 0]  w_iacts_from_ctrl_to_dpe                ;
    wire    [IACTS_NUM_BANKS                    -1: 0]  w_iacts_valid_from_ctrl_to_dpe          ;
    wire    [DPE_COL_NUM                        -1: 0]  w_weights_valid_from_ctrl_to_dpe        ;
    wire    [WEIGHTS_SRAM_DATA_WIDTH            -1: 0]  w_weights_from_ctrl_to_dpe              ;
    wire    [IACTS_DATA_WIDTH                   -1: 0]  w_iacts_zp_from_ctrl_to_dpe             ;
    wire    [WEIGHTS_DATA_WIDTH/DPE_COL_NUM     -1: 0]  w_weights_zp_from_ctrl_to_dpe           ;
    wire                                                w_weights_ping_pong_sel                 ;
    wire    [PE_SEL_WIDTH                       -1: 0]  w_pe_sel                                ;
    wire    [LOG2_WEIGHTS_DEPTH                 -1: 0]  w_weights_to_use                        ;
    wire    [birrd_IN_COMMAND_WIDTH             -1: 0]  w_birrd_instr                           ;

    /*
        feather Controller + Buffers' signals
    */
    wire    [IACTS_SRAM_DATA_WIDTH              -1: 0]  w_iacts_sram_a_wr_data_ping             ;
    wire    [IACTS_SRAM_ADDR_WIDTH              -1: 0]  w_iacts_sram_a_wr_addr_ping             ;
    wire    [IACTS_NUM_BANKS                    -1: 0]  w_iacts_sram_a_wr_en_ping               ;
    wire    [IACTS_SRAM_ADDR_WIDTH              -1: 0]  w_iacts_sram_b_rd_addr_ping             ;
    wire    [IACTS_NUM_BANKS                    -1: 0]  w_iacts_sram_b_rd_en_ping               ;
    wire    [IACTS_SRAM_DATA_WIDTH              -1: 0]  w_iacts_sram_b_rd_data_ping             ;
    wire    [IACTS_SRAM_DATA_WIDTH              -1: 0]  w_iacts_sram_a_wr_data_pong             ;
    wire    [IACTS_SRAM_ADDR_WIDTH              -1: 0]  w_iacts_sram_a_wr_addr_pong             ;
    wire    [IACTS_NUM_BANKS                    -1: 0]  w_iacts_sram_a_wr_en_pong               ;
    wire    [IACTS_SRAM_ADDR_WIDTH              -1: 0]  w_iacts_sram_b_rd_addr_pong             ;
    wire    [IACTS_NUM_BANKS                    -1: 0]  w_iacts_sram_b_rd_en_pong               ;
    wire    [IACTS_SRAM_DATA_WIDTH              -1: 0]  w_iacts_sram_b_rd_data_pong             ;
    wire    [WEIGHTS_SRAM_DATA_WIDTH            -1: 0]  w_weights_sram_a_wr_data_ping           ;
    wire    [WEIGHTS_SRAM_ADDR_WIDTH            -1: 0]  w_weights_sram_a_wr_addr_ping           ;
    wire    [WEIGHTS_NUM_BANKS                  -1: 0]  w_weights_sram_a_wr_en_ping             ;
    wire    [WEIGHTS_SRAM_ADDR_WIDTH            -1: 0]  w_weights_sram_b_rd_addr_ping           ;
    wire    [WEIGHTS_NUM_BANKS                  -1: 0]  w_weights_sram_b_rd_en_ping             ;
    wire    [WEIGHTS_SRAM_DATA_WIDTH            -1: 0]  w_weights_sram_b_rd_data_ping           ;
    wire    [WEIGHTS_SRAM_DATA_WIDTH            -1: 0]  w_weights_sram_a_wr_data_pong           ;
    wire    [WEIGHTS_SRAM_ADDR_WIDTH            -1: 0]  w_weights_sram_a_wr_addr_pong           ;
    wire    [WEIGHTS_NUM_BANKS                  -1: 0]  w_weights_sram_a_wr_en_pong             ;
    wire    [WEIGHTS_SRAM_ADDR_WIDTH            -1: 0]  w_weights_sram_b_rd_addr_pong           ;
    wire    [WEIGHTS_NUM_BANKS                  -1: 0]  w_weights_sram_b_rd_en_pong             ;
    wire    [WEIGHTS_SRAM_DATA_WIDTH            -1: 0]  w_weights_sram_b_rd_data_pong           ;
    wire    [OUTBUF_SRAM_DATA_WIDTH             -1: 0]  w_outbuf_sram_a_wr_data                 ;
    wire    [OUTBUF_SRAM_ADDR_WIDTH             -1: 0]  w_outbuf_sram_a_wr_addr                 ;
    wire    [OUTBUF_NUM_BANKS                   -1: 0]  w_outbuf_sram_a_wr_en                   ;
    wire    [OUTBUF_SRAM_ADDR_WIDTH             -1: 0]  w_outbuf_sram_b_rd_addr                 ;
    wire    [OUTBUF_NUM_BANKS                   -1: 0]  w_outbuf_sram_b_rd_en                   ;
    wire    [OUTBUF_SRAM_DATA_WIDTH             -1: 0]  w_outbuf_sram_b_rd_data                 ;
    wire    [INSTR_SRAM_BANK_DATA_WIDTH         -1: 0]  w_instr_sram_a_wr_data                  ;
    wire    [INSTR_SRAM_BANK_ADDR_WIDTH         -1: 0]  w_instr_sram_a_wr_addr                  ;
    wire                                                w_instr_sram_a_wr_en                    ;
    wire    [INSTR_SRAM_BANK_ADDR_WIDTH         -1: 0]  w_instr_sram_b_rd_addr                  ;
    wire                                                w_instr_sram_b_rd_en                    ;
    wire    [INSTR_SRAM_BANK_DATA_WIDTH         -1: 0]  w_instr_sram_b_rd_data                  ;

    /*
        birrd signals
    */
    wire    [IACTS_DATA_WIDTH                   -1: 0]  w_dpe_iacts_zp                  [0 : DPE_COL_NUM-1];
    wire    [WEIGHTS_DATA_WIDTH/DPE_COL_NUM     -1: 0]  w_dpe_weights_zp                [0 : DPE_COL_NUM-1];
    wire                                                w_dpe_weights_ping_pong_sel     [0 : DPE_ROW_NUM  ][0 : DPE_COL_NUM-1];
    wire    [PE_SEL_WIDTH                       -1: 0]  w_dpe_pe_sel                    [0 : DPE_ROW_NUM  ][0 : DPE_COL_NUM-1];
    wire    [LOG2_WEIGHTS_DEPTH                 -1: 0]  w_dpe_weights_to_use            [0 : DPE_ROW_NUM  ][0 : DPE_COL_NUM-1];
    wire    [IACTS_DATA_WIDTH                   -1: 0]  w_dpe_iacts                     [0 : DPE_ROW_NUM  ][0 : DPE_COL_NUM-1];
    wire                                                w_dpe_iacts_valid               [0 : DPE_ROW_NUM  ][0 : DPE_COL_NUM-1];
    wire    [WEIGHTS_DATA_WIDTH/DPE_COL_NUM     -1: 0]  w_dpe_weights                   [0 : DPE_ROW_NUM  ][0 : DPE_COL_NUM-1];
    wire                                                w_dpe_weights_valid             [0 : DPE_ROW_NUM  ][0 : DPE_COL_NUM-1];
    wire    [PE_OUTPUT_WIDTH                    -1: 0]  w_dpe_pe_out_data_arr           [0 : DPE_ROW_NUM-1][0 : DPE_COL_NUM-1];
    wire                                                w_dpe_pe_out_data_arr_valid     [0 : DPE_ROW_NUM-1][0 : DPE_COL_NUM-1];

    wire    [(PE_OUTPUT_WIDTH*DPE_ROW_NUM)      -1: 0]  w_dpe_out_to_mux_data_bus       [0 : DPE_COL_NUM-1];
    wire    [(                DPE_ROW_NUM)      -1: 0]  w_dpe_out_to_mux_data_bus_valid [0 : DPE_COL_NUM-1];

    wire    [PE_OUTPUT_WIDTH                    -1: 0]  w_o_col_mux_data                [0 : DPE_COL_NUM-1];
    wire                                                w_o_col_mux_data_valid          [0 : DPE_COL_NUM-1];
   
    wire    [(PE_OUTPUT_WIDTH*DPE_COL_NUM)      -1: 0]  w_i_birrd_data_bus               ;
    wire    [(                DPE_COL_NUM)      -1: 0]  w_i_birrd_data_bus_valid         ;
    wire    [(PE_OUTPUT_WIDTH*DPE_COL_NUM)      -1: 0]  w_o_birrd_data_bus               ;
    wire    [(                DPE_COL_NUM)      -1: 0]  w_o_birrd_data_bus_valid         ;

/*
____________________________________________________________________________________________________________________________
Weights PING BUFFER

PORT A - Write Port
PORT B - Read  Port
____________________________________________________________________________________________________________________________
*/
    sram_sp_2d_array#(
        .SRAM_BANK_DATA_WIDTH   (WEIGHTS_DATA_WIDTH                 ),
        .SRAM_BANK_ADDR_WIDTH   (WEIGHTS_SRAM_BANK_ADDR_WIDTH       ),
        .SRAM_BANK_DEPTH        (2**WEIGHTS_SRAM_BANK_ADDR_WIDTH    ),
        .NUM_BANK               (WEIGHTS_NUM_BANKS                  )
    ) WEIGHTS_PING_BUFFER(
        .clk                    (clk                                ),
        .rst_n                  (rst_n                              ),
        .i_sram_a_wr_data       (w_weights_sram_a_wr_data_ping      ),
        .i_sram_a_wr_addr       (w_weights_sram_a_wr_addr_ping      ),
        .i_sram_a_wr_en         (w_weights_sram_a_wr_en_ping        ),
        .i_sram_b_rd_addr       (w_weights_sram_b_rd_addr_ping      ),
        .i_sram_b_rd_en         (w_weights_sram_b_rd_en_ping        ),
        .o_sram_b_rd_data       (w_weights_sram_b_rd_data_ping      )
    );

/*
____________________________________________________________________________________________________________________________
Weights PONG BUFFER

PORT A - Write Port
PORT B - Read  Port
____________________________________________________________________________________________________________________________
*/
    sram_sp_2d_array#(
        .SRAM_BANK_DATA_WIDTH   (WEIGHTS_DATA_WIDTH                 ),
        .SRAM_BANK_ADDR_WIDTH   (WEIGHTS_SRAM_BANK_ADDR_WIDTH       ),
        .SRAM_BANK_DEPTH        (2**WEIGHTS_SRAM_BANK_ADDR_WIDTH    ),
        .NUM_BANK               (WEIGHTS_NUM_BANKS                  )
    ) WEIGHTS_PONG_BUFFER(
        .clk                    (clk                                ),
        .rst_n                  (rst_n                              ),
        .i_sram_a_wr_data       (w_weights_sram_a_wr_data_pong      ),
        .i_sram_a_wr_addr       (w_weights_sram_a_wr_addr_pong      ),
        .i_sram_a_wr_en         (w_weights_sram_a_wr_en_pong        ),
        .i_sram_b_rd_addr       (w_weights_sram_b_rd_addr_pong      ),
        .i_sram_b_rd_en         (w_weights_sram_b_rd_en_pong        ),
        .o_sram_b_rd_data       (w_weights_sram_b_rd_data_pong      )
    );

/*
____________________________________________________________________________________________________________________________
IACTS PING BUFFER

PORT A - Write Port
PORT B - Read  Port
____________________________________________________________________________________________________________________________
*/
    sram_sp_2d_array#(
        .SRAM_BANK_DATA_WIDTH   (IACTS_DATA_WIDTH               ),
        .SRAM_BANK_ADDR_WIDTH   (IACTS_SRAM_BANK_ADDR_WIDTH     ),
        .SRAM_BANK_DEPTH        (2**IACTS_SRAM_BANK_ADDR_WIDTH  ),
        .NUM_BANK               (IACTS_NUM_BANKS                )
    ) IACTS_PING_SRAM(
        .clk                    (clk                            ),
        .rst_n                  (rst_n                          ),
        .i_sram_a_wr_data       (w_iacts_sram_a_wr_data_ping    ),
        .i_sram_a_wr_addr       (w_iacts_sram_a_wr_addr_ping    ),
        .i_sram_a_wr_en         (w_iacts_sram_a_wr_en_ping      ),
        .i_sram_b_rd_addr       (w_iacts_sram_b_rd_addr_ping    ),
        .i_sram_b_rd_en         (w_iacts_sram_b_rd_en_ping      ),
        .o_sram_b_rd_data       (w_iacts_sram_b_rd_data_ping    )
    );
/*
____________________________________________________________________________________________________________________________
IACTS PONG BUFFER

PORT A - Write Port
PORT B - Read  Port
____________________________________________________________________________________________________________________________
*/
    sram_sp_2d_array#(
        .SRAM_BANK_DATA_WIDTH   (IACTS_DATA_WIDTH               ),
        .SRAM_BANK_ADDR_WIDTH   (IACTS_SRAM_BANK_ADDR_WIDTH     ),
        .SRAM_BANK_DEPTH        (2**IACTS_SRAM_BANK_ADDR_WIDTH  ),
        .NUM_BANK               (IACTS_NUM_BANKS                )
    ) IACTS_PONG_SRAM(
        .clk                    (clk                            ),
        .rst_n                  (rst_n                          ),
        .i_sram_a_wr_data       (w_iacts_sram_a_wr_data_pong    ),
        .i_sram_a_wr_addr       (w_iacts_sram_a_wr_addr_pong    ),
        .i_sram_a_wr_en         (w_iacts_sram_a_wr_en_pong      ),
        .i_sram_b_rd_addr       (w_iacts_sram_b_rd_addr_pong    ),
        .i_sram_b_rd_en         (w_iacts_sram_b_rd_en_pong      ),
        .o_sram_b_rd_data       (w_iacts_sram_b_rd_data_pong    )
    );

/*
____________________________________________________________________________________________________________________________
INSTRUCTION BUFFER

PORT A - Write Port
PORT B - Read  Port
____________________________________________________________________________________________________________________________
*/
    sram_sp_2d_array#(
        .SRAM_BANK_DATA_WIDTH   (INSTR_SRAM_BANK_DATA_WIDTH     ),
        .SRAM_BANK_ADDR_WIDTH   (INSTR_SRAM_BANK_ADDR_WIDTH     ),
        .SRAM_BANK_DEPTH        (2**INSTR_SRAM_BANK_ADDR_WIDTH  ),
        .NUM_BANK               (1                              )
    ) INSTR_SRAM(
        .clk                    (clk                            ),
        .rst_n                  (rst_n                          ),
        .i_sram_a_wr_data       (w_instr_sram_a_wr_data         ),
        .i_sram_a_wr_addr       (w_instr_sram_a_wr_addr         ),
        .i_sram_a_wr_en         (w_instr_sram_a_wr_en           ),
        .i_sram_b_rd_addr       (w_instr_sram_b_rd_addr         ),
        .i_sram_b_rd_en         (w_instr_sram_b_rd_en           ),
        .o_sram_b_rd_data       (w_instr_sram_b_rd_data         )
    );

/*
____________________________________________________________________________________________________________________________
birrd OUTPUT BUFFER

PORT A - Write Port
PORT B - Read  Port
____________________________________________________________________________________________________________________________
*/
    sram_sp_2d_array#(
        .SRAM_BANK_DATA_WIDTH   (OUTBUF_DATA_WIDTH              ),
        .SRAM_BANK_ADDR_WIDTH   (OUTBUF_SRAM_BANK_ADDR_WIDTH    ),
        .SRAM_BANK_DEPTH        (2**OUTBUF_SRAM_BANK_ADDR_WIDTH ),
        .NUM_BANK               (DPE_COL_NUM                    )
    ) OUTPUT_BUFFER_SRAM(
        .clk                    (clk                            ),
        .rst_n                  (rst_n                          ),
        .i_sram_a_wr_data       (w_outbuf_sram_a_wr_data        ),
        .i_sram_a_wr_addr       (w_outbuf_sram_a_wr_addr        ),
        .i_sram_a_wr_en         (w_outbuf_sram_a_wr_en          ),
        .i_sram_b_rd_addr       (w_outbuf_sram_b_rd_addr        ),
        .i_sram_b_rd_en         (w_outbuf_sram_b_rd_en          ),
        .o_sram_b_rd_data       (w_outbuf_sram_b_rd_data        )
    );
//_________________________________________________________________________________________________________________________//



/*
____________________________________________________________________________________________________________________________
feather Controller
____________________________________________________________________________________________________________________________
*/
    feather_controller #(
        .WEIGHTS_DATA_WIDTH                 (WEIGHTS_DATA_WIDTH                 ),
        .WEIGHTS_NUM_BANKS                  (WEIGHTS_NUM_BANKS                  ),
        .WEIGHTS_SRAM_DATA_WIDTH            (WEIGHTS_SRAM_DATA_WIDTH            ),
        .WEIGHTS_SRAM_BANK_ADDR_WIDTH       (WEIGHTS_SRAM_BANK_ADDR_WIDTH       ),
        .WEIGHTS_SRAM_ADDR_WIDTH            (WEIGHTS_SRAM_ADDR_WIDTH            ),
        .IACTS_DATA_WIDTH                   (IACTS_DATA_WIDTH                   ),
        .IACTS_NUM_BANKS                    (IACTS_NUM_BANKS                    ),
        .IACTS_SRAM_DATA_WIDTH              (IACTS_SRAM_DATA_WIDTH              ),
        .IACTS_SRAM_BANK_ADDR_WIDTH         (IACTS_SRAM_BANK_ADDR_WIDTH         ),
        .IACTS_SRAM_ADDR_WIDTH              (IACTS_SRAM_ADDR_WIDTH              ),
        .SCALE_VALUE_WIDTH                  (SCALE_VALUE_WIDTH                  ),
        .PE_OUTPUT_WIDTH                    (PE_OUTPUT_WIDTH                    ),
        .DPE_COL_NUM                        (DPE_COL_NUM                        ),
        .DPE_ROW_NUM                        (DPE_ROW_NUM                        ),
        .LOG2_DPE_COL_NUM                   (LOG2_DPE_COL_NUM                   ),
        .LOG2_DPE_ROW_NUM                   (LOG2_DPE_ROW_NUM                   ),
        .OACTS_DATA_WIDTH                   (OACTS_DATA_WIDTH                   ),
        .OUTBUF_DATA_WIDTH                  (OUTBUF_DATA_WIDTH                  ),
        .OUTBUF_NUM_BANKS                   (OUTBUF_NUM_BANKS                   ),
        .OUTBUF_SRAM_DATA_WIDTH             (OUTBUF_SRAM_DATA_WIDTH             ),
        .OUTBUF_SRAM_BANK_ADDR_WIDTH        (OUTBUF_SRAM_BANK_ADDR_WIDTH        ),
        .OUTBUF_SRAM_ADDR_WIDTH             (OUTBUF_SRAM_ADDR_WIDTH             ),
        .INSTR_SRAM_BANK_ADDR_WIDTH         (INSTR_SRAM_BANK_ADDR_WIDTH         )
    )feather_CONTROLLER_INST(
        .clk                                (clk                                ),  // inputs from top
        .rst_n                              (rst_n                              ),
        .i_feather_top_en                   (i_feather_top_en                   ),
        .i_iacts_zp                         (i_iacts_zp                         ),
        .i_iacts_zp_valid                   (i_iacts_zp_valid                   ),
        .i_weights_zp                       (i_weights_zp                       ),
        .i_weights_zp_valid                 (i_weights_zp_valid                 ),
        .i_scale_val                        (i_scale_val                        ),
        .i_weights_write_valid              (i_weights_write_valid              ),
        .i_weights_write_data               (i_weights_write_data               ),
        .i_weights_write_addr               (i_weights_write_addr               ),
        .i_weights_write_addr_end           (i_weights_write_addr_end           ),
        .i_iacts_write_valid                (i_iacts_write_valid                ),
        .i_iacts_write_data                 (i_iacts_write_data                 ),
        .i_iacts_write_addr                 (i_iacts_write_addr                 ),
        .i_iacts_write_addr_end             (i_iacts_write_addr_end             ),
        .i_instr_write_valid                (i_instr_write_valid                ),
        .i_instr_write_data                 (i_instr_write_data                 ),
        .i_instr_write_addr                 (i_instr_write_addr                 ),
        .i_oacts_read_valid                 (i_oacts_read_valid                 ),
        .i_oacts_read_addr                  (i_oacts_read_addr                  ),
        .i_oacts_read_addr_end              (i_oacts_read_addr_end              ),
        .o_oacts_read_data                  (o_oacts_read_data                  ),
        .i_all_buf_pingpong_config          (i_all_buf_pingpong_config          ),
        .o_outbuf_data_wr_rdy               (o_outbuf_data_wr_rdy               ),
        .i_outbuf_wr_instr                  (i_outbuf_wr_instr                  ),
        .o_iacts_from_ctrl_to_dpe           (w_iacts_from_ctrl_to_dpe           ),  // Outputs to birrd
        .o_iacts_valid_from_ctrl_to_dpe     (w_iacts_valid_from_ctrl_to_dpe     ),
        .o_weights_from_ctrl_to_dpe         (w_weights_from_ctrl_to_dpe         ),
        .o_weights_valid_from_ctrl_to_dpe   (w_weights_valid_from_ctrl_to_dpe   ),
        .o_iacts_zp_from_ctrl_to_dpe        (w_iacts_zp_from_ctrl_to_dpe        ),
        .o_weights_zp_from_ctrl_to_dpe      (w_weights_zp_from_ctrl_to_dpe      ),
        .o_weights_ping_pong_sel            (w_weights_ping_pong_sel            ),
        .o_pe_sel                           (w_pe_sel                           ),
        .o_weights_to_use                   (w_weights_to_use                   ),
        .o_birrd_instr                      (w_birrd_instr                      ),
        .i_data_bus_from_birrd              (w_o_birrd_data_bus                 ),  // inputs from birrd
        .i_data_bus_from_birrd_valid        (w_o_birrd_data_bus_valid           ),
        .o_iacts_sram_a_wr_data_ping        (w_iacts_sram_a_wr_data_ping        ),  // I-O from-to iActs Ping
        .o_iacts_sram_a_wr_addr_ping        (w_iacts_sram_a_wr_addr_ping        ),
        .o_iacts_sram_a_wr_en_ping          (w_iacts_sram_a_wr_en_ping          ),
        .o_iacts_sram_b_rd_addr_ping        (w_iacts_sram_b_rd_addr_ping        ),
        .o_iacts_sram_b_rd_en_ping          (w_iacts_sram_b_rd_en_ping          ),
        .i_iacts_sram_b_rd_data_ping        (w_iacts_sram_b_rd_data_ping        ),
        .o_iacts_sram_a_wr_data_pong        (w_iacts_sram_a_wr_data_pong        ),  // I-O from-to iActs Pong
        .o_iacts_sram_a_wr_addr_pong        (w_iacts_sram_a_wr_addr_pong        ),
        .o_iacts_sram_a_wr_en_pong          (w_iacts_sram_a_wr_en_pong          ),
        .o_iacts_sram_b_rd_addr_pong        (w_iacts_sram_b_rd_addr_pong        ),
        .o_iacts_sram_b_rd_en_pong          (w_iacts_sram_b_rd_en_pong          ),
        .i_iacts_sram_b_rd_data_pong        (w_iacts_sram_b_rd_data_pong        ),
        .o_weights_sram_a_wr_data_ping      (w_weights_sram_a_wr_data_ping      ),  // I-O from-to Weights Ping
        .o_weights_sram_a_wr_addr_ping      (w_weights_sram_a_wr_addr_ping      ),
        .o_weights_sram_a_wr_en_ping        (w_weights_sram_a_wr_en_ping        ),
        .o_weights_sram_b_rd_addr_ping      (w_weights_sram_b_rd_addr_ping      ),
        .o_weights_sram_b_rd_en_ping        (w_weights_sram_b_rd_en_ping        ),
        .i_weights_sram_b_rd_data_ping      (w_weights_sram_b_rd_data_ping      ),
        .o_weights_sram_a_wr_data_pong      (w_weights_sram_a_wr_data_pong      ),  // I-O from-to Weights Pong
        .o_weights_sram_a_wr_addr_pong      (w_weights_sram_a_wr_addr_pong      ),
        .o_weights_sram_a_wr_en_pong        (w_weights_sram_a_wr_en_pong        ),
        .o_weights_sram_b_rd_addr_pong      (w_weights_sram_b_rd_addr_pong      ),
        .o_weights_sram_b_rd_en_pong        (w_weights_sram_b_rd_en_pong        ),
        .i_weights_sram_b_rd_data_pong      (w_weights_sram_b_rd_data_pong      ),
        .o_outbuf_sram_a_wr_data            (w_outbuf_sram_a_wr_data            ),  // I-O from-to birrd OUTPUT Buffer
        .o_outbuf_sram_a_wr_addr            (w_outbuf_sram_a_wr_addr            ),
        .o_outbuf_sram_a_wr_en              (w_outbuf_sram_a_wr_en              ),
        .o_outbuf_sram_b_rd_addr            (w_outbuf_sram_b_rd_addr            ),
        .o_outbuf_sram_b_rd_en              (w_outbuf_sram_b_rd_en              ),
        .i_outbuf_sram_b_rd_data            (w_outbuf_sram_b_rd_data            ),
        .o_instr_sram_a_wr_data             (w_instr_sram_a_wr_data             ),  // I-O from-to INSTR Buffer
        .o_instr_sram_a_wr_addr             (w_instr_sram_a_wr_addr             ),
        .o_instr_sram_a_wr_en               (w_instr_sram_a_wr_en               ),
        .o_instr_sram_b_rd_addr             (w_instr_sram_b_rd_addr             ),
        .o_instr_sram_b_rd_en               (w_instr_sram_b_rd_en               ),
        .i_instr_sram_b_rd_data             (w_instr_sram_b_rd_data             )
    );



//_________________________________________________________________________________________________________________________//



//_________________________________________________________________________________________________________________________
// connecting data and control to birrd input
    genvar GENVAR_VEC_TO_ARR_COL_ITER;

    generate
        for(GENVAR_VEC_TO_ARR_COL_ITER=0; GENVAR_VEC_TO_ARR_COL_ITER < DPE_COL_NUM; GENVAR_VEC_TO_ARR_COL_ITER=GENVAR_VEC_TO_ARR_COL_ITER+1)
        begin:feather_SRAM_BANK_TO_DPE_COL

            assign w_dpe_iacts[0][GENVAR_VEC_TO_ARR_COL_ITER]                     =   w_iacts_from_ctrl_to_dpe           [(GENVAR_VEC_TO_ARR_COL_ITER*IACTS_DATA_WIDTH)      +:  IACTS_DATA_WIDTH];
            assign w_dpe_iacts_valid[0][GENVAR_VEC_TO_ARR_COL_ITER]               =   w_iacts_valid_from_ctrl_to_dpe     [GENVAR_VEC_TO_ARR_COL_ITER];

            assign w_dpe_weights[0][GENVAR_VEC_TO_ARR_COL_ITER]                   =   w_weights_from_ctrl_to_dpe         [(GENVAR_VEC_TO_ARR_COL_ITER*(WEIGHTS_DATA_WIDTH/DPE_COL_NUM))    +:  WEIGHTS_DATA_WIDTH/DPE_COL_NUM];
            assign w_dpe_weights_valid[0][GENVAR_VEC_TO_ARR_COL_ITER]             =   w_weights_valid_from_ctrl_to_dpe   [GENVAR_VEC_TO_ARR_COL_ITER];

            assign w_dpe_iacts_zp[GENVAR_VEC_TO_ARR_COL_ITER]                     =   w_iacts_zp_from_ctrl_to_dpe;

            assign w_dpe_weights_zp[GENVAR_VEC_TO_ARR_COL_ITER]                   =   w_weights_zp_from_ctrl_to_dpe;

            assign w_dpe_weights_ping_pong_sel[0][GENVAR_VEC_TO_ARR_COL_ITER]     =   w_weights_ping_pong_sel;
            assign w_dpe_pe_sel[0][GENVAR_VEC_TO_ARR_COL_ITER]                    =   w_pe_sel;
            assign w_dpe_weights_to_use[0][GENVAR_VEC_TO_ARR_COL_ITER]            =   w_weights_to_use;
        end

    endgenerate




//===============================================================================================//
// DPE ARRAY

    genvar  GENVAR_DPE_INST_ROW_ITER, GENVAR_DPE_INST_COL_ITER;
    genvar  GENVAR_birrd_INBUS_ITER;

    generate
        for(GENVAR_DPE_INST_COL_ITER=0; GENVAR_DPE_INST_COL_ITER < DPE_COL_NUM; GENVAR_DPE_INST_COL_ITER=GENVAR_DPE_INST_COL_ITER+1)
        begin:feather_GENVAR_DPE_INST_COL_ITER

            for(GENVAR_DPE_INST_ROW_ITER=0; GENVAR_DPE_INST_ROW_ITER < DPE_ROW_NUM; GENVAR_DPE_INST_ROW_ITER=GENVAR_DPE_INST_ROW_ITER+1)
            begin: feather_GENVAR_DPE_INST_ROW_ITER
                feather_pe#(
                    .THIS_PE_ID                 ((DPE_ROW_NUM*GENVAR_DPE_INST_COL_ITER) + GENVAR_DPE_INST_ROW_ITER),
                    .IACTS_DATA_WIDTH           (IACTS_DATA_WIDTH   ),
                    .WEIGHTS_DATA_WIDTH         (WEIGHTS_DATA_WIDTH/DPE_COL_NUM ),
                    .WEIGHTS_DEPTH              (DPE_ROW_NUM        ),
                    .LOG2_WEIGHTS_DEPTH         (LOG2_WEIGHTS_DEPTH ),
                    .PE_SEL_WIDTH               (PE_SEL_WIDTH       ),
                    .PE_OUTPUT_WIDTH            (PE_OUTPUT_WIDTH    )
                ) feather_PE_OTHER_ROWS(
                    .clk                        (clk                                                                                ),
                    .rst_n                      (rst_n                                                                              ),
                    .i_iacts                    (w_dpe_iacts                [GENVAR_DPE_INST_ROW_ITER  ][GENVAR_DPE_INST_COL_ITER]  ),
                    .i_iacts_valid              (w_dpe_iacts_valid          [GENVAR_DPE_INST_ROW_ITER  ][GENVAR_DPE_INST_COL_ITER]  ),
                    .i_weights                  (w_dpe_weights              [GENVAR_DPE_INST_ROW_ITER  ][GENVAR_DPE_INST_COL_ITER]  ),
                    .i_weights_valid            (w_dpe_weights_valid        [GENVAR_DPE_INST_ROW_ITER  ][GENVAR_DPE_INST_COL_ITER]  ),
                    .i_iacts_zp                 (w_dpe_iacts_zp             [GENVAR_DPE_INST_COL_ITER  ]                            ),
                    .i_weights_zp               (w_dpe_weights_zp           [GENVAR_DPE_INST_COL_ITER  ]                            ),
                    .i_weights_ping_pong_sel    (w_dpe_weights_ping_pong_sel[GENVAR_DPE_INST_ROW_ITER  ][GENVAR_DPE_INST_COL_ITER]  ),
                    .i_pe_sel                   (w_dpe_pe_sel               [GENVAR_DPE_INST_ROW_ITER  ][GENVAR_DPE_INST_COL_ITER]  ),
                    .i_weights_to_use           (w_dpe_weights_to_use       [GENVAR_DPE_INST_ROW_ITER  ][GENVAR_DPE_INST_COL_ITER]  ),
                    .o_weights_ping_pong_sel    (w_dpe_weights_ping_pong_sel[GENVAR_DPE_INST_ROW_ITER+1][GENVAR_DPE_INST_COL_ITER]  ),
                    .o_pe_sel                   (w_dpe_pe_sel               [GENVAR_DPE_INST_ROW_ITER+1][GENVAR_DPE_INST_COL_ITER]  ),
                    .o_weights_to_use           (w_dpe_weights_to_use       [GENVAR_DPE_INST_ROW_ITER+1][GENVAR_DPE_INST_COL_ITER]  ),
                    .o_iacts                    (w_dpe_iacts                [GENVAR_DPE_INST_ROW_ITER+1][GENVAR_DPE_INST_COL_ITER]  ),
                    .o_iacts_valid              (w_dpe_iacts_valid          [GENVAR_DPE_INST_ROW_ITER+1][GENVAR_DPE_INST_COL_ITER]  ),
                    .o_weights                  (w_dpe_weights              [GENVAR_DPE_INST_ROW_ITER+1][GENVAR_DPE_INST_COL_ITER]  ),
                    .o_weights_valid            (w_dpe_weights_valid        [GENVAR_DPE_INST_ROW_ITER+1][GENVAR_DPE_INST_COL_ITER]  ),
                    .o_out_data                 (w_dpe_pe_out_data_arr      [GENVAR_DPE_INST_ROW_ITER  ][GENVAR_DPE_INST_COL_ITER]  ),
                    .o_out_data_valid           (w_dpe_pe_out_data_arr_valid[GENVAR_DPE_INST_ROW_ITER  ][GENVAR_DPE_INST_COL_ITER]  )
                );
            end

            for(GENVAR_birrd_INBUS_ITER=0; GENVAR_birrd_INBUS_ITER < DPE_ROW_NUM; GENVAR_birrd_INBUS_ITER=GENVAR_birrd_INBUS_ITER+1)
            begin:DPE_OUT_ARRAY_TO_BUS
                assign  w_dpe_out_to_mux_data_bus      [GENVAR_DPE_INST_COL_ITER][(GENVAR_birrd_INBUS_ITER*PE_OUTPUT_WIDTH)  +:  PE_OUTPUT_WIDTH]    = w_dpe_pe_out_data_arr      [GENVAR_birrd_INBUS_ITER][GENVAR_DPE_INST_COL_ITER];
                assign  w_dpe_out_to_mux_data_bus_valid[GENVAR_DPE_INST_COL_ITER][GENVAR_birrd_INBUS_ITER]                                           = w_dpe_pe_out_data_arr_valid[GENVAR_birrd_INBUS_ITER][GENVAR_DPE_INST_COL_ITER];
            end
            //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
            //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
            //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
            //// Mult-Stage muxing tree [Per Column]
            o_bus_autopick_seq#(
                .NUM_INPUT_DATA (DPE_ROW_NUM                                                    ),
                .DATA_WIDTH     (PE_OUTPUT_WIDTH                                                )
            )COL_WISE_O_DATA_BUS(
                .clk            (clk                                                            ),
                .rst_n          (rst_n                                                          ),
                .i_valid        (w_dpe_out_to_mux_data_bus_valid    [GENVAR_DPE_INST_COL_ITER]  ),
                .i_data_bus     (w_dpe_out_to_mux_data_bus          [GENVAR_DPE_INST_COL_ITER]  ),
                .o_valid        (w_o_col_mux_data_valid             [GENVAR_DPE_INST_COL_ITER]  ),
                .o_data_bus     (w_o_col_mux_data                   [GENVAR_DPE_INST_COL_ITER]  ),
                .i_en           (1'b1                                                           )
            );
            //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
            //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
            //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

            assign  w_i_birrd_data_bus       [(GENVAR_DPE_INST_COL_ITER*PE_OUTPUT_WIDTH) +:  PE_OUTPUT_WIDTH]    = w_o_col_mux_data      [GENVAR_DPE_INST_COL_ITER];
            assign  w_i_birrd_data_bus_valid [GENVAR_DPE_INST_COL_ITER]                                          = w_o_col_mux_data_valid[GENVAR_DPE_INST_COL_ITER];
        end
    endgenerate

//===============================================================================================//

//###############################################################################################//
//// birrd
    birrd_simple_cmd_flow_seq #(
        .COMMAND_WIDTH      (2                          ),
        .DATA_WIDTH         (PE_OUTPUT_WIDTH            ),
        .NUM_INPUT_DATA     (DPE_COL_NUM                ),
        .IN_COMMAND_WIDTH   (birrd_COMMAND_WIDTH_PER_ROW )
    )   birrd_INST(
        .clk                (clk                        ),
        .rst_n              (rst_n                      ),
        .i_valid            (w_i_birrd_data_bus_valid   ),
        .i_data_bus         (w_i_birrd_data_bus         ),
        .o_valid            (w_o_birrd_data_bus_valid   ),
        .o_data_bus         (w_o_birrd_data_bus         ),
        .i_en               (1'b1                       ),
        .i_cmd              (w_birrd_instr              )
    );
//###############################################################################################//

endmodule
