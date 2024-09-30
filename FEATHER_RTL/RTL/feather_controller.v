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
    Top Module:  Feather Controller
    Data:        Only data width matters.
    Format:      keeping the input format unchanged
    Timing:      Sequential Logic
    Reset:       Asynchronized Reset [Low Reset]

    Function:    Interconnects all the FEATHER modules. Performs the SRAM RD/WR control and PING/PONG muxing logic
    Note:        This template is designed for FEATHER with DPE_COL_NUM>4. 
                 When DPE_COL_NUM=DPE_ROW_NUM=4, BIRRD only has "2log(DPE_COL_NUM) - 1" stages instead of "2log(DPE_COL_NUM)" stage.
*/

`timescale 1ns / 1ps

module feather_controller #(
    parameter WEIGHTS_DATA_WIDTH                =   8,                                              //
    parameter WEIGHTS_NUM_BANKS                 =   4,                                              //
    parameter WEIGHTS_SRAM_DATA_WIDTH           =   WEIGHTS_NUM_BANKS*WEIGHTS_DATA_WIDTH,           //
    parameter WEIGHTS_SRAM_BANK_ADDR_WIDTH      =   10,                                             //
    parameter WEIGHTS_SRAM_ADDR_WIDTH           =   WEIGHTS_NUM_BANKS*WEIGHTS_SRAM_BANK_ADDR_WIDTH, //

    parameter IACTS_DATA_WIDTH                  =   8,                                              //
    parameter IACTS_NUM_BANKS                   =   4,                                              //
    parameter IACTS_SRAM_DATA_WIDTH             =   IACTS_NUM_BANKS*IACTS_DATA_WIDTH,               //
    parameter IACTS_SRAM_BANK_ADDR_WIDTH        =   10,                                             //
    parameter IACTS_SRAM_ADDR_WIDTH             =   IACTS_NUM_BANKS*IACTS_SRAM_BANK_ADDR_WIDTH,     //

    parameter SCALE_VALUE_WIDTH                 =   32,                                             //

    parameter PE_OUTPUT_WIDTH                   =   32,                                             //

    parameter DPE_COL_NUM                       =   8,                                              //
    parameter DPE_ROW_NUM                       =   4,                                              //
    parameter LOG2_DPE_COL_NUM                  =   3,                                              //
    parameter LOG2_DPE_ROW_NUM                  =   2,                                              //

    parameter OACTS_DATA_WIDTH                  =   8,                                              //

    parameter OUTBUF_DATA_WIDTH                 =   PE_OUTPUT_WIDTH,                                //
    parameter OUTBUF_NUM_BANKS                  =   DPE_COL_NUM,                                    //
    parameter OUTBUF_SRAM_DATA_WIDTH            =   OUTBUF_NUM_BANKS*OUTBUF_DATA_WIDTH,             //
    parameter OUTBUF_SRAM_BANK_ADDR_WIDTH       =   10,                                             //
    parameter OUTBUF_SRAM_ADDR_WIDTH            =   OUTBUF_NUM_BANKS*OUTBUF_SRAM_BANK_ADDR_WIDTH,   //

    parameter INSTR_SRAM_BANK_ADDR_WIDTH        =   10                                              //
    )(

    // signals from feather TOP
    clk                                 ,
    rst_n                               ,
    i_feather_top_en                     ,
    i_iacts_zp                          ,
    i_iacts_zp_valid                    ,
    i_weights_zp                        ,
    i_weights_zp_valid                  ,
    i_scale_val                         ,
    i_weights_write_valid               ,
    i_weights_write_data                ,
    i_weights_write_addr                ,
    i_weights_write_addr_end            ,
    i_iacts_write_valid                 ,
    i_iacts_write_data                  ,
    i_iacts_write_addr                  ,
    i_iacts_write_addr_end              ,
    i_instr_write_valid                 ,
    i_instr_write_data                  ,
    i_instr_write_addr                  ,
    i_oacts_read_valid                  ,
    o_oacts_read_data                   ,
    i_oacts_read_addr                   ,
    i_oacts_read_addr_end               ,
    i_all_buf_pingpong_config           ,
    o_outbuf_data_wr_rdy                ,
    i_outbuf_wr_instr                   ,

    // Outputs to birrd
    o_iacts_from_ctrl_to_dpe            ,
    o_iacts_valid_from_ctrl_to_dpe      ,
    o_weights_from_ctrl_to_dpe          ,
    o_weights_valid_from_ctrl_to_dpe    ,
    o_iacts_zp_from_ctrl_to_dpe         ,
    o_weights_zp_from_ctrl_to_dpe       ,
    o_weights_ping_pong_sel             ,
    o_pe_sel                            ,
    o_weights_to_use                    ,
    o_birrd_instr                        ,

    // inputs from birrd
    i_data_bus_from_birrd                ,
    i_data_bus_from_birrd_valid          ,

    // I-O from-to iActs Ping
    o_iacts_sram_a_wr_data_ping         ,
    o_iacts_sram_a_wr_addr_ping         ,
    o_iacts_sram_a_wr_en_ping           ,
    o_iacts_sram_b_rd_addr_ping         ,
    o_iacts_sram_b_rd_en_ping           ,
    i_iacts_sram_b_rd_data_ping         ,

    // I-O from-to iActs Pong
    o_iacts_sram_a_wr_data_pong         ,
    o_iacts_sram_a_wr_addr_pong         ,
    o_iacts_sram_a_wr_en_pong           ,
    o_iacts_sram_b_rd_addr_pong         ,
    o_iacts_sram_b_rd_en_pong           ,
    i_iacts_sram_b_rd_data_pong         ,

    // I-O from-to Weights Ping
    o_weights_sram_a_wr_data_ping       ,
    o_weights_sram_a_wr_addr_ping       ,
    o_weights_sram_a_wr_en_ping         ,
    o_weights_sram_b_rd_addr_ping       ,
    o_weights_sram_b_rd_en_ping         ,
    i_weights_sram_b_rd_data_ping       ,

    // I-O from-to Weights Pong
    o_weights_sram_a_wr_data_pong       ,
    o_weights_sram_a_wr_addr_pong       ,
    o_weights_sram_a_wr_en_pong         ,
    o_weights_sram_b_rd_addr_pong       ,
    o_weights_sram_b_rd_en_pong         ,
    i_weights_sram_b_rd_data_pong       ,

    // I-O from-to birrd OUTPUT Buffer
    o_outbuf_sram_a_wr_data             ,
    o_outbuf_sram_a_wr_addr             ,
    o_outbuf_sram_a_wr_en               ,
    o_outbuf_sram_b_rd_addr             ,
    o_outbuf_sram_b_rd_en               ,
    i_outbuf_sram_b_rd_data             ,

    // I-O from-to INSTR Buffer
    o_instr_sram_a_wr_data              ,
    o_instr_sram_a_wr_addr              ,
    o_instr_sram_a_wr_en                ,
    o_instr_sram_b_rd_addr              ,
    o_instr_sram_b_rd_en                ,
    i_instr_sram_b_rd_data

);

    localparam NUM_STAGE                            =   2*(LOG2_DPE_COL_NUM);                               // 2_BIT_CMD * (Number of stages-1) *(birrd_INPUTS/2)
    localparam birrd_COMMAND_WIDTH_PER_ROW          =   2*NUM_STAGE ;                                       // 2_BIT_CMD * (Number of stages-1)
    localparam birrd_IN_COMMAND_WIDTH               =   birrd_COMMAND_WIDTH_PER_ROW * DPE_COL_NUM >> 1 ;    // 2_BIT_CMD * (Number of stages-1) *(birrd_INPUTS/2)
    localparam INSTR_SRAM_BANK_DATA_WIDTH           =   birrd_IN_COMMAND_WIDTH;                             //  instruction is only birrd instruction here
    localparam WEIGHTS_DEPTH                        =   DPE_ROW_NUM;                                        //
    localparam LOG2_WEIGHTS_DEPTH                   =   LOG2_DPE_ROW_NUM;                                   //
    localparam PE_SEL_WIDTH                         =   LOG2_DPE_COL_NUM + LOG2_WEIGHTS_DEPTH;              //

    localparam IACTS_PINGPONG_CONFIG_WIDTH          =   4;
    localparam WEIGHTS_PINGPONG_CONFIG_WIDTH        =   4;
    localparam ALL_BUF_CONFIG_WIDTH                 =   IACTS_PINGPONG_CONFIG_WIDTH + WEIGHTS_PINGPONG_CONFIG_WIDTH;
    localparam OUTBUF_SRAM_INSTR_WIDTH              =   OUTBUF_SRAM_ADDR_WIDTH + (2*OUTBUF_NUM_BANKS); //  OUTBUF_NUM_BANKS*ADDR + (OUTBUF_NUM_BANKS ---for r_bypass_to_scale) + (OUTBUF_NUM_BANKS--- for r_mul_with_scale)

    localparam [IACTS_PINGPONG_CONFIG_WIDTH     -1: 0]  IACTS_PINGPONG_IDLE                      =   0;
    localparam [IACTS_PINGPONG_CONFIG_WIDTH     -1: 0]  IACTS_PINGPONG_FILL_PING                 =   1;
    localparam [IACTS_PINGPONG_CONFIG_WIDTH     -1: 0]  IACTS_PINGPONG_PING_FEED_DPE_FILL_PONG   =   2;
    localparam [IACTS_PINGPONG_CONFIG_WIDTH     -1: 0]  IACTS_PINGPONG_PONG_FEED_DPE_FILL_PING   =   3;
    localparam [IACTS_PINGPONG_CONFIG_WIDTH     -1: 0]  IACTS_PINGPONG_DRAIN_PONG                =   4;
    localparam [IACTS_PINGPONG_CONFIG_WIDTH     -1: 0]  IACTS_PINGPONG_DRAIN_PING                =   5;
    localparam [IACTS_PINGPONG_CONFIG_WIDTH     -1: 0]  IACTS_PINGPONG_FILL_PONG                 =   6;

    localparam [WEIGHTS_PINGPONG_CONFIG_WIDTH   -1: 0]  WEIGHTS_PINGPONG_IDLE                    =   0;
    localparam [WEIGHTS_PINGPONG_CONFIG_WIDTH   -1: 0]  WEIGHTS_PINGPONG_FILL_PING               =   1;
    localparam [WEIGHTS_PINGPONG_CONFIG_WIDTH   -1: 0]  WEIGHTS_PINGPONG_PING_FEED_DPE           =   2;
    localparam [WEIGHTS_PINGPONG_CONFIG_WIDTH   -1: 0]  WEIGHTS_PINGPONG_FILL_PONG               =   3;
    localparam [WEIGHTS_PINGPONG_CONFIG_WIDTH   -1: 0]  WEIGHTS_PINGPONG_PONG_FEED_DPE           =   4;


    /*
        ports
    */
    input                                               clk                                     ;
    input                                               rst_n                                   ;
    input                                               i_feather_top_en                        ;
    input       [IACTS_DATA_WIDTH               -1: 0]  i_iacts_zp                              ;
    input                                               i_iacts_zp_valid                        ;
    input       [WEIGHTS_DATA_WIDTH/DPE_COL_NUM -1: 0]  i_weights_zp                            ;
    input                                               i_weights_zp_valid                      ;
    input       [SCALE_VALUE_WIDTH              -1: 0]  i_scale_val                             ;
    input                                               i_iacts_write_valid                     ;
    input       [IACTS_SRAM_DATA_WIDTH          -1: 0]  i_iacts_write_data                      ;
    input       [IACTS_SRAM_BANK_ADDR_WIDTH     -1: 0]  i_iacts_write_addr                      ;
    input       [IACTS_SRAM_BANK_ADDR_WIDTH     -1: 0]  i_iacts_write_addr_end                  ;
    input                                               i_weights_write_valid                   ;
    input       [WEIGHTS_SRAM_DATA_WIDTH        -1: 0]  i_weights_write_data                    ;
    input       [WEIGHTS_SRAM_BANK_ADDR_WIDTH   -1: 0]  i_weights_write_addr                    ;
    input       [WEIGHTS_SRAM_BANK_ADDR_WIDTH   -1: 0]  i_weights_write_addr_end                ;
    input                                               i_instr_write_valid                     ;
    input       [INSTR_SRAM_BANK_DATA_WIDTH     -1: 0]  i_instr_write_data                      ;
    input       [INSTR_SRAM_BANK_ADDR_WIDTH     -1: 0]  i_instr_write_addr                      ;
    input                                               i_oacts_read_valid                      ;
    output      [IACTS_SRAM_DATA_WIDTH          -1: 0]  o_oacts_read_data                       ;
    input       [IACTS_SRAM_ADDR_WIDTH          -1: 0]  i_oacts_read_addr                       ;
    input       [IACTS_SRAM_ADDR_WIDTH          -1: 0]  i_oacts_read_addr_end                   ;   // check this --- because differet address in different bank might be read? or alwayws 1 addr for all banks?
    input       [ALL_BUF_CONFIG_WIDTH           -1: 0]  i_all_buf_pingpong_config               ;
    output      [OUTBUF_NUM_BANKS               -1: 0]  o_outbuf_data_wr_rdy                    ;
    input       [OUTBUF_SRAM_INSTR_WIDTH        -1: 0]  i_outbuf_wr_instr                       ;

    // Outputs to birrd
    output      [IACTS_SRAM_DATA_WIDTH          -1: 0]  o_iacts_from_ctrl_to_dpe                ;
    output      [IACTS_NUM_BANKS                -1: 0]  o_iacts_valid_from_ctrl_to_dpe          ;
    output      [DPE_COL_NUM                    -1: 0]  o_weights_valid_from_ctrl_to_dpe        ;
    output      [WEIGHTS_SRAM_DATA_WIDTH        -1: 0]  o_weights_from_ctrl_to_dpe              ;
    output      [IACTS_DATA_WIDTH               -1: 0]  o_iacts_zp_from_ctrl_to_dpe             ;
    output      [WEIGHTS_DATA_WIDTH/DPE_COL_NUM -1: 0]  o_weights_zp_from_ctrl_to_dpe           ;
    output                                              o_weights_ping_pong_sel                 ;
    output      [PE_SEL_WIDTH                   -1: 0]  o_pe_sel                                ;
    output      [LOG2_WEIGHTS_DEPTH             -1: 0]  o_weights_to_use                        ;
    output      [birrd_IN_COMMAND_WIDTH         -1: 0]  o_birrd_instr                           ;

    // inputs from birrd
    input       [OUTBUF_SRAM_DATA_WIDTH         -1: 0]  i_data_bus_from_birrd                   ;
    input       [OUTBUF_NUM_BANKS               -1: 0]  i_data_bus_from_birrd_valid             ;

    // I-O from-to iActs Ping
    output      [IACTS_SRAM_DATA_WIDTH          -1: 0]  o_iacts_sram_a_wr_data_ping             ;
    output      [IACTS_SRAM_ADDR_WIDTH          -1: 0]  o_iacts_sram_a_wr_addr_ping             ;
    output      [IACTS_NUM_BANKS                -1: 0]  o_iacts_sram_a_wr_en_ping               ;
    output      [IACTS_SRAM_ADDR_WIDTH          -1: 0]  o_iacts_sram_b_rd_addr_ping             ;
    output      [IACTS_NUM_BANKS                -1: 0]  o_iacts_sram_b_rd_en_ping               ;
    input       [IACTS_SRAM_DATA_WIDTH          -1: 0]  i_iacts_sram_b_rd_data_ping             ;

    // I-O from-to iActs Pong
    output      [IACTS_SRAM_DATA_WIDTH          -1: 0]  o_iacts_sram_a_wr_data_pong             ;
    output      [IACTS_SRAM_ADDR_WIDTH          -1: 0]  o_iacts_sram_a_wr_addr_pong             ;
    output      [IACTS_NUM_BANKS                -1: 0]  o_iacts_sram_a_wr_en_pong               ;
    output      [IACTS_SRAM_ADDR_WIDTH          -1: 0]  o_iacts_sram_b_rd_addr_pong             ;
    output      [IACTS_NUM_BANKS                -1: 0]  o_iacts_sram_b_rd_en_pong               ;
    input       [IACTS_SRAM_DATA_WIDTH          -1: 0]  i_iacts_sram_b_rd_data_pong             ;

    // I-O from-to Weights Ping
    output      [WEIGHTS_SRAM_DATA_WIDTH        -1: 0]  o_weights_sram_a_wr_data_ping           ;
    output      [WEIGHTS_SRAM_ADDR_WIDTH        -1: 0]  o_weights_sram_a_wr_addr_ping           ;
    output      [WEIGHTS_NUM_BANKS              -1: 0]  o_weights_sram_a_wr_en_ping             ;
    output      [WEIGHTS_SRAM_ADDR_WIDTH        -1: 0]  o_weights_sram_b_rd_addr_ping           ;
    output      [WEIGHTS_NUM_BANKS              -1: 0]  o_weights_sram_b_rd_en_ping             ;
    input       [WEIGHTS_SRAM_DATA_WIDTH        -1: 0]  i_weights_sram_b_rd_data_ping           ;

    // I-O from-to Weights Pong
    output      [WEIGHTS_SRAM_DATA_WIDTH        -1: 0]  o_weights_sram_a_wr_data_pong           ;
    output      [WEIGHTS_SRAM_ADDR_WIDTH        -1: 0]  o_weights_sram_a_wr_addr_pong           ;
    output      [WEIGHTS_NUM_BANKS              -1: 0]  o_weights_sram_a_wr_en_pong             ;
    output      [WEIGHTS_SRAM_ADDR_WIDTH        -1: 0]  o_weights_sram_b_rd_addr_pong           ;
    output      [WEIGHTS_NUM_BANKS              -1: 0]  o_weights_sram_b_rd_en_pong             ;
    input       [WEIGHTS_SRAM_DATA_WIDTH        -1: 0]  i_weights_sram_b_rd_data_pong           ;

    // I-O from-to birrd OUTPUT Buffer
    output      [OUTBUF_SRAM_DATA_WIDTH         -1: 0]  o_outbuf_sram_a_wr_data                 ;
    output      [OUTBUF_SRAM_ADDR_WIDTH         -1: 0]  o_outbuf_sram_a_wr_addr                 ;
    output      [OUTBUF_NUM_BANKS               -1: 0]  o_outbuf_sram_a_wr_en                   ;
    output      [OUTBUF_SRAM_ADDR_WIDTH         -1: 0]  o_outbuf_sram_b_rd_addr                 ;
    output      [OUTBUF_NUM_BANKS               -1: 0]  o_outbuf_sram_b_rd_en                   ;
    input       [OUTBUF_SRAM_DATA_WIDTH         -1: 0]  i_outbuf_sram_b_rd_data                 ;

    output      [INSTR_SRAM_BANK_DATA_WIDTH     -1: 0]  o_instr_sram_a_wr_data                  ;
    output      [INSTR_SRAM_BANK_ADDR_WIDTH     -1: 0]  o_instr_sram_a_wr_addr                  ;
    output                                              o_instr_sram_a_wr_en                    ;
    output      [INSTR_SRAM_BANK_ADDR_WIDTH     -1: 0]  o_instr_sram_b_rd_addr                  ;
    output                                              o_instr_sram_b_rd_en                    ;
    input       [INSTR_SRAM_BANK_DATA_WIDTH     -1: 0]  i_instr_sram_b_rd_data                  ;


    // internal signals

    reg     [IACTS_PINGPONG_CONFIG_WIDTH        -1: 0]  r_acts_buf_ping_pong_state          ;
    reg     [WEIGHTS_PINGPONG_CONFIG_WIDTH      -1: 0]  r_weights_buf_ping_pong_state       ;

    reg     [IACTS_DATA_WIDTH                   -1: 0]  r_iacts_zp                          ;
    reg     [WEIGHTS_DATA_WIDTH/DPE_COL_NUM     -1: 0]  r_weights_zp                        ;
    reg     [SCALE_VALUE_WIDTH                  -1: 0]  r_scale_val                         ;

    reg     [IACTS_SRAM_BANK_ADDR_WIDTH         -1: 0]  r_iacts_pingpong_rd_addr            ;
    reg     [WEIGHTS_SRAM_BANK_ADDR_WIDTH       -1: 0]  r_weights_pingpong_rd_addr          ;

    wire    [OUTBUF_NUM_BANKS                   -1: 0]  w_bypass_to_scale                   ;
    reg     [OUTBUF_NUM_BANKS                   -1: 0]  r_bypass_to_scale                   ;
    wire    [OUTBUF_NUM_BANKS                   -1: 0]  w_mul_with_scale                    ;
    reg     [OUTBUF_NUM_BANKS                   -1: 0]  r_mul_with_scale                    ;
    reg     [OUTBUF_NUM_BANKS                   -1: 0]  r_mul_with_scale_and_store_oact     ;

    wire    [IACTS_PINGPONG_CONFIG_WIDTH        -1: 0]  w_iacts_pingpong_config             ;
    wire    [WEIGHTS_PINGPONG_CONFIG_WIDTH      -1: 0]  w_weights_pingpong_config           ;
    reg     [birrd_IN_COMMAND_WIDTH             -1: 0]  r_birrd_instr                       ;


    reg                                                 r_weights_ping_pong_sel             ;
    reg     [PE_SEL_WIDTH                       -1: 0]  r_pe_sel                            ;
    reg     [LOG2_WEIGHTS_DEPTH                 -1: 0]  r_weights_to_use                    ;

    reg     [OUTBUF_NUM_BANKS                   -1: 0]  r_o_birrd_data_bus_valid            ;
    reg     [OUTBUF_NUM_BANKS                   -1: 0]  r_partial_sum_done                  ;


    // signals for birrd output bus
    reg     [OUTBUF_SRAM_DATA_WIDTH             -1: 0]  r_outbuf_sram_a_wr_data             ;
    reg     [OUTBUF_SRAM_ADDR_WIDTH             -1: 0]  r_outbuf_sram_a_wr_addr_temp        ;
    reg     [OUTBUF_SRAM_ADDR_WIDTH             -1: 0]  r_outbuf_sram_a_wr_addr             ;
    reg     [OUTBUF_SRAM_ADDR_WIDTH             -1: 0]  r_outbuf_sram_b_rd_addr             ;
    reg     [OUTBUF_SRAM_ADDR_WIDTH             -1: 0]  r_oacts_wr_addr_to_pingpong         ;   // should this be IACTS_SRAM_ADDR_WIDTH ??? check this
    wire    [PE_OUTPUT_WIDTH                    -1: 0]  w_o_birrd_data               [0 : DPE_COL_NUM-1];
    reg     [PE_OUTPUT_WIDTH                    -1: 0]  r_o_birrd_data_prev          [0 : DPE_COL_NUM-1];
    reg     [PE_OUTPUT_WIDTH                    -1: 0]  r_o_birrd_data               [0 : DPE_COL_NUM-1];
    reg     [PE_OUTPUT_WIDTH                    -1: 0]  r_o_birrd_data_for_mul       [0 : DPE_COL_NUM-1];
    wire    [PE_OUTPUT_WIDTH+SCALE_VALUE_WIDTH  -1: 0]  w_o_birrd_data_post_quant    [0 : DPE_COL_NUM-1];
    reg     [OACTS_DATA_WIDTH                   -1: 0]  r_oacts_post_quant          [0 : DPE_COL_NUM-1];
    wire    [IACTS_SRAM_DATA_WIDTH              -1: 0]  w_oacts_post_quant;

    wire     [IACTS_SRAM_DATA_WIDTH              -1: 0]  w_iacts_from_ctrl_to_dpe           ;
    wire     [IACTS_NUM_BANKS                    -1: 0]  w_iacts_valid_from_ctrl_to_dpe     ;
    wire     [DPE_COL_NUM                        -1: 0]  w_weights_valid_from_ctrl_to_dpe   ;
    wire     [WEIGHTS_SRAM_DATA_WIDTH            -1: 0]  w_weights_from_ctrl_to_dpe         ;
    wire     [IACTS_SRAM_DATA_WIDTH              -1: 0]  w_iacts_sram_a_wr_data_ping        ;
    wire     [IACTS_SRAM_ADDR_WIDTH              -1: 0]  w_iacts_sram_a_wr_addr_ping        ;
    wire     [IACTS_NUM_BANKS                    -1: 0]  w_iacts_sram_a_wr_en_ping          ;
    wire     [IACTS_SRAM_ADDR_WIDTH              -1: 0]  w_iacts_sram_b_rd_addr_ping        ;
    wire     [IACTS_NUM_BANKS                    -1: 0]  w_iacts_sram_b_rd_en_ping          ;
    wire     [IACTS_SRAM_DATA_WIDTH              -1: 0]  w_iacts_sram_a_wr_data_pong        ;
    wire     [IACTS_SRAM_ADDR_WIDTH              -1: 0]  w_iacts_sram_a_wr_addr_pong        ;
    wire     [IACTS_NUM_BANKS                    -1: 0]  w_iacts_sram_a_wr_en_pong          ;
    wire     [IACTS_SRAM_ADDR_WIDTH              -1: 0]  w_iacts_sram_b_rd_addr_pong        ;
    wire     [IACTS_NUM_BANKS                    -1: 0]  w_iacts_sram_b_rd_en_pong          ;
    wire     [WEIGHTS_SRAM_DATA_WIDTH            -1: 0]  w_weights_sram_a_wr_data_ping      ;
    wire     [WEIGHTS_SRAM_ADDR_WIDTH            -1: 0]  w_weights_sram_a_wr_addr_ping      ;
    wire     [WEIGHTS_NUM_BANKS                  -1: 0]  w_weights_sram_a_wr_en_ping        ;
    wire     [WEIGHTS_SRAM_ADDR_WIDTH            -1: 0]  w_weights_sram_b_rd_addr_ping      ;
    wire     [WEIGHTS_NUM_BANKS                  -1: 0]  w_weights_sram_b_rd_en_ping        ;
    wire     [WEIGHTS_SRAM_DATA_WIDTH            -1: 0]  w_weights_sram_a_wr_data_pong      ;
    wire     [WEIGHTS_SRAM_ADDR_WIDTH            -1: 0]  w_weights_sram_a_wr_addr_pong      ;
    wire     [WEIGHTS_NUM_BANKS                  -1: 0]  w_weights_sram_a_wr_en_pong        ;
    wire     [WEIGHTS_SRAM_ADDR_WIDTH            -1: 0]  w_weights_sram_b_rd_addr_pong      ;
    wire     [WEIGHTS_NUM_BANKS                  -1: 0]  w_weights_sram_b_rd_en_pong        ;
    wire     [IACTS_SRAM_DATA_WIDTH              -1: 0]  w_oacts_read_data                  ;




    // extracting ping pong FSM state instruction from i_all_buf_pingpong_config - feather top
    assign w_iacts_pingpong_config   = i_all_buf_pingpong_config[0     +:  IACTS_PINGPONG_CONFIG_WIDTH];
    assign w_weights_pingpong_config = i_all_buf_pingpong_config[IACTS_PINGPONG_CONFIG_WIDTH     +:  WEIGHTS_PINGPONG_CONFIG_WIDTH];

    /*
        Reading/Writing Weights, iActs and Instruction from SRAMs
    */
    always @(posedge clk or negedge rst_n)
    begin
        if(!rst_n)
        begin
            r_scale_val                     <=  0;
            r_weights_zp                    <=  0;
            r_iacts_zp                      <=  0;
            r_birrd_instr                   <=  0;
            r_iacts_pingpong_rd_addr        <=  0;
            r_weights_pingpong_rd_addr      <=  0;
            r_weights_ping_pong_sel         <=  0;
            r_pe_sel                        <=  0;
            r_weights_to_use                <=  0;
            r_acts_buf_ping_pong_state      <=  IACTS_PINGPONG_IDLE;
            r_weights_buf_ping_pong_state   <=  WEIGHTS_PINGPONG_IDLE;
        end
        else
        begin

        //________________________________________________________________________________________________________________________
            //  r_acts_buf_ping_pong_state  FSM
            case(r_acts_buf_ping_pong_state)
                IACTS_PINGPONG_IDLE:
                begin
                    //(w_iacts_pingpong_config ==  IACTS_PINGPONG_FILL_PING)
                    if(i_feather_top_en == 1)
                    begin
                        r_acts_buf_ping_pong_state  <=  IACTS_PINGPONG_FILL_PING;
                        r_iacts_pingpong_rd_addr    <=  0;
                    end
                end


                IACTS_PINGPONG_FILL_PING:
                begin
                    //(w_iacts_pingpong_config ==  IACTS_PINGPONG_PING_FEED_DPE_FILL_PONG)
                    if(i_iacts_write_addr == i_iacts_write_addr_end)
                    begin
                        r_acts_buf_ping_pong_state  <=  IACTS_PINGPONG_PING_FEED_DPE_FILL_PONG;
                        r_iacts_pingpong_rd_addr    <=  0;
                    end
                end


                IACTS_PINGPONG_PING_FEED_DPE_FILL_PONG:
                begin
                    if(r_iacts_pingpong_rd_addr ==  i_iacts_write_addr_end)
                    begin
                        if      (w_iacts_pingpong_config ==  IACTS_PINGPONG_DRAIN_PONG)
                        begin
                            r_acts_buf_ping_pong_state  <=  IACTS_PINGPONG_DRAIN_PONG;
                        end
                        else if (w_iacts_pingpong_config ==  IACTS_PINGPONG_PONG_FEED_DPE_FILL_PING)
                        begin
                            r_acts_buf_ping_pong_state  <=  IACTS_PINGPONG_PONG_FEED_DPE_FILL_PING;
                        end

                        r_iacts_pingpong_rd_addr        <=  0;
                    end
                    else
                    begin
                        r_iacts_pingpong_rd_addr        <=  r_iacts_pingpong_rd_addr + 1;
                    end
                end


                IACTS_PINGPONG_PONG_FEED_DPE_FILL_PING:
                begin
                    if(r_iacts_pingpong_rd_addr ==  i_iacts_write_addr_end)
                    begin
                        if      (w_iacts_pingpong_config ==  IACTS_PINGPONG_DRAIN_PING)
                        begin
                            r_acts_buf_ping_pong_state  <=  IACTS_PINGPONG_DRAIN_PING;
                        end
                        else if (w_iacts_pingpong_config ==  IACTS_PINGPONG_PING_FEED_DPE_FILL_PONG)
                        begin
                            r_acts_buf_ping_pong_state  <=  IACTS_PINGPONG_PING_FEED_DPE_FILL_PONG;
                        end
                        r_iacts_pingpong_rd_addr        <=  0;
                    end
                    else
                    begin
                        r_iacts_pingpong_rd_addr    <=  r_iacts_pingpong_rd_addr + 1;
                    end

                end


                IACTS_PINGPONG_DRAIN_PONG:
                begin
                    if(i_oacts_read_addr    ==  i_oacts_read_addr_end)
                    begin
                        if      (w_iacts_pingpong_config ==  IACTS_PINGPONG_FILL_PONG)
                        begin
                            r_acts_buf_ping_pong_state  <=  IACTS_PINGPONG_FILL_PONG;
                        end
                        else if (w_iacts_pingpong_config ==  IACTS_PINGPONG_FILL_PING)
                        begin
                            r_acts_buf_ping_pong_state  <=  IACTS_PINGPONG_FILL_PING;
                        end
                    end
                end

                IACTS_PINGPONG_DRAIN_PING:
                begin
                    if(i_oacts_read_addr    ==  i_oacts_read_addr_end)
                    begin
                        if      (w_iacts_pingpong_config ==  IACTS_PINGPONG_FILL_PONG)
                        begin
                            r_acts_buf_ping_pong_state  <=  IACTS_PINGPONG_FILL_PONG;
                        end
                        else if (w_iacts_pingpong_config ==  IACTS_PINGPONG_FILL_PING)
                        begin
                            r_acts_buf_ping_pong_state  <=  IACTS_PINGPONG_FILL_PING;
                        end
                    end
                end


                IACTS_PINGPONG_FILL_PONG:
                begin
                    if(i_iacts_write_addr == i_iacts_write_addr_end)
                    begin
                        r_acts_buf_ping_pong_state  <=  IACTS_PINGPONG_PONG_FEED_DPE_FILL_PING;
                    end
                end


                default:
                begin
                    r_acts_buf_ping_pong_state      <=  IACTS_PINGPONG_IDLE;
                    r_iacts_pingpong_rd_addr        <=  0;
                end
            endcase
        //________________________________________________________________________________________________________________________




        //________________________________________________________________________________________________________________________
            case(r_weights_buf_ping_pong_state)
                WEIGHTS_PINGPONG_IDLE:
                begin
                    if(i_feather_top_en == 1)
                    begin
                        r_weights_buf_ping_pong_state   <=  WEIGHTS_PINGPONG_FILL_PING;
                        r_weights_pingpong_rd_addr      <=  0;
                    end
                end


                WEIGHTS_PINGPONG_FILL_PING:
                begin
                    if(i_weights_write_addr ==  i_weights_write_addr_end)
                    begin
                        if(w_weights_pingpong_config ==  WEIGHTS_PINGPONG_PING_FEED_DPE)
                        begin
                            r_weights_buf_ping_pong_state   <=  WEIGHTS_PINGPONG_PING_FEED_DPE;
                            r_weights_pingpong_rd_addr      <=  0;
                        end
                    end
                end


                WEIGHTS_PINGPONG_PING_FEED_DPE:
                begin
                    if(r_weights_pingpong_rd_addr   ==  i_weights_write_addr_end)
                    begin
                        if(w_weights_pingpong_config ==  WEIGHTS_PINGPONG_FILL_PONG)
                        begin
                            r_weights_buf_ping_pong_state   <=  WEIGHTS_PINGPONG_FILL_PONG;
                        end
                        r_weights_pingpong_rd_addr          <=  0;
                        r_pe_sel                            <=  0;
                    end
                    begin
                        r_weights_pingpong_rd_addr          <=  r_weights_pingpong_rd_addr + 1;
                        r_weights_to_use                    <=  $unsigned(WEIGHTS_DEPTH)-1;
                        r_pe_sel                            <=  r_pe_sel + 1;
                    end
                end


                WEIGHTS_PINGPONG_FILL_PONG:
                begin
                    if(i_weights_write_addr == i_weights_write_addr_end)
                    begin
                        r_weights_buf_ping_pong_state   <=  WEIGHTS_PINGPONG_PONG_FEED_DPE;
                        r_weights_pingpong_rd_addr      <=  0;
                    end
                end


                WEIGHTS_PINGPONG_PONG_FEED_DPE:
                begin
                    if(r_weights_pingpong_rd_addr   ==  i_weights_write_addr_end)
                    begin
                        r_weights_buf_ping_pong_state   <=  WEIGHTS_PINGPONG_FILL_PING;
                        r_weights_pingpong_rd_addr      <=  0;
                    end
                    begin
                        r_weights_pingpong_rd_addr      <=  r_weights_pingpong_rd_addr + 1;
                    end
                end


                default:
                begin
                    r_weights_buf_ping_pong_state <=  0;
                end
            endcase
            //________________________________________________________________________________________________________________________
            //************************************************************************************************//
            if(r_pe_sel == $unsigned(DPE_COL_NUM*WEIGHTS_DEPTH)-1)
            begin
                r_weights_ping_pong_sel <=  ~r_weights_ping_pong_sel;
            end
            //************************************************************************************************//





        //************************************************************************************************//
        // Quantization and ZP data generation
        if(i_iacts_zp_valid)
        begin
            r_iacts_zp          <=  i_iacts_zp;
        end
        if(i_weights_zp_valid)
        begin
            r_weights_zp        <=  i_weights_zp;
            r_scale_val         <=  i_scale_val;
        end
        //************************************************************************************************//


        //************************************************************************************************//
        r_birrd_instr        <=  i_instr_sram_b_rd_data[birrd_IN_COMMAND_WIDTH-1 : 0];
        //************************************************************************************************//

        end
    end

//__________________________________________________________________________________________________________________________
//__________________________________________________________________________________________________________________________

    /*
            ping pong control generation

        + ---------------------------------+        + ---------------------------------+
        |  r_acts_buf_ping_pong_state = 1  |        |  r_acts_buf_ping_pong_state = 6  |
        +----------------------------------+        +----------------------------------+
        |    in                            |        |                            in    |
        |     \                            |        |                           /      |
        |      V                           |        |                          V       |
        |      +---+           +---+       |        |      +---+           +---+       |
        |      |   |           |   |       |        |      |   |           |   |       |
        |      | I |           | O |       |        |      | I |           | O |       |
        |      |   |           |   |       |        |      |   |           |   |       |
        |      +---+           +---+       |        |      +---+           +---+       |
        |                                  |        |                                  |
        |                                  |        |                                  |
        |                                  |        |                                  |
        +----------------------------------+        +----------------------------------+

        + ---------------------------------+        + ---------------------------------+
        |  r_acts_buf_ping_pong_state = 2  |        |  r_acts_buf_ping_pong_state = 4  |
        +----------------------------------+        +----------------------------------+
        |                                  |        |                          out     |
        |                                  |        |                           Λ      |
        |                                  |        |                          /       |
        |      +---+           +---+       |        |      +---+           +---+       |
        |      |   |           |   |       |        |      |   |           |   |       |
        |      | I |           | O |       |        |      | I |           | O |       |
        |      |   |           |   |       |        |      |   |           |   |       |
        |      +---+           +---+       |        |      +---+           +---+       |
        |           \          Λ           |        |                                  |
        |            V        /            |        |                                  |
        |             DPE+birrd            |        |             DPE+birrd             |
        +----------------------------------+        +----------------------------------+

        + ---------------------------------+        + ---------------------------------+
        |  r_acts_buf_ping_pong_state = 3  |        |  r_acts_buf_ping_pong_state = 5  |
        +----------------------------------+        +----------------------------------+
        |                                  |        |   out                            |
        |                                  |        |    Λ                             |
        |                                  |        |     \                            |
        |      +---+           +---+       |        |      +---+           +---+       |
        |      |   |           |   |       |        |      |   |           |   |       |
        |      | I |           | O |       |        |      | I |           | O |       |
        |      |   |           |   |       |        |      |   |           |   |       |
        |      +---+           +---+       |        |      +---+           +---+       |
        |          Λ           /           |        |                                  |
        |           \         V            |        |                                  |
        |             DPE+birrd            |        |             DPE+birrd             |
        +----------------------------------+        +----------------------------------+

            ping pong control generation - weights

        + ---------------------------------+        + ---------------------------------+
        | r_weights_buf_ping_pong_state= 1 |        | r_weights_buf_ping_pong_state= 3 |
        +----------------------------------+        +----------------------------------+
        |    in                            |        |                            in    |
        |     \                            |        |                           /      |
        |      V                           |        |                          V       |
        |      +---+           +---+       |        |      +---+           +---+       |
        |      |   |           |   |       |        |      |   |           |   |       |
        |      | I |           | O |       |        |      | I |           | O |       |
        |      |   |           |   |       |        |      |   |           |   |       |
        |      +---+           +---+       |        |      +---+           +---+       |
        |                                  |        |                                  |
        |                                  |        |                                  |
        |                                  |        |                                  |
        +----------------------------------+        +----------------------------------+

        + ---------------------------------+        + ---------------------------------+
        | r_weights_buf_ping_pong_state= 2 |        | r_weights_buf_ping_pong_state= 4 |
        +----------------------------------+        +----------------------------------+
        |                                  |        |                                  |
        |                                  |        |                                  |
        |                                  |        |                                  |
        |      +---+           +---+       |        |      +---+           +---+       |
        |      |   |           |   |       |        |      |   |           |   |       |
        |      | I |           | O |       |        |      | I |           | O |       |
        |      |   |           |   |       |        |      |   |           |   |       |
        |      +---+           +---+       |        |      +---+           +---+       |
        |           \                      |        |                      /           |
        |            V                     |        |                     V            |
        |             DPE+birrd             |        |             DPE+birrd             |
        +----------------------------------+        +----------------------------------+
    */

    // for iActs Ping, port A
    assign  w_iacts_sram_a_wr_data_ping         =   (r_acts_buf_ping_pong_state ==  IACTS_PINGPONG_FILL_PING)               ?   i_iacts_write_data                              :   (
                                                    (r_acts_buf_ping_pong_state ==  IACTS_PINGPONG_PONG_FEED_DPE_FILL_PING) ?   w_oacts_post_quant                              :
                                                                                                                                0   );
    assign  w_iacts_sram_a_wr_addr_ping         =   (r_acts_buf_ping_pong_state ==  IACTS_PINGPONG_FILL_PING)               ?   {IACTS_NUM_BANKS{i_iacts_write_addr}}           :   (
                                                    (r_acts_buf_ping_pong_state ==  IACTS_PINGPONG_PONG_FEED_DPE_FILL_PING) ?   r_oacts_wr_addr_to_pingpong                     :
                                                                                                                                0   );
    assign  w_iacts_sram_a_wr_en_ping           =   (r_acts_buf_ping_pong_state ==  IACTS_PINGPONG_FILL_PING)               ?   {IACTS_NUM_BANKS{i_iacts_write_valid}}          :   (
                                                    (r_acts_buf_ping_pong_state ==  IACTS_PINGPONG_PONG_FEED_DPE_FILL_PING) ?   r_mul_with_scale_and_store_oact                 :
                                                                                                                                0   );
    // for iActs Ping, port B   
    assign  w_iacts_sram_b_rd_addr_ping         =   (r_acts_buf_ping_pong_state ==  IACTS_PINGPONG_PING_FEED_DPE_FILL_PONG) ?   {IACTS_NUM_BANKS{r_iacts_pingpong_rd_addr}}     :   (
                                                    (r_acts_buf_ping_pong_state ==  IACTS_PINGPONG_DRAIN_PING)              ?   {IACTS_NUM_BANKS{i_oacts_read_addr}}            :
                                                                                                                                0   );
    assign  w_iacts_sram_b_rd_en_ping           =   (r_acts_buf_ping_pong_state ==  IACTS_PINGPONG_PING_FEED_DPE_FILL_PONG) ?   ~0                                              :   (
                                                    (r_acts_buf_ping_pong_state ==  IACTS_PINGPONG_DRAIN_PING)              ?   {IACTS_NUM_BANKS{i_oacts_read_valid}}           :
                                                                                                                                0   );

    // for iActs Pong, port A   
    assign  w_iacts_sram_a_wr_data_pong         =   (r_acts_buf_ping_pong_state ==  IACTS_PINGPONG_FILL_PONG)               ?   i_iacts_write_data                              :   (
                                                    (r_acts_buf_ping_pong_state ==  IACTS_PINGPONG_PING_FEED_DPE_FILL_PONG) ?   w_oacts_post_quant                              :
                                                                                                                                0   );
    assign  w_iacts_sram_a_wr_addr_pong         =   (r_acts_buf_ping_pong_state ==  IACTS_PINGPONG_FILL_PONG)               ?   {IACTS_NUM_BANKS{i_iacts_write_addr}}           :   (
                                                    (r_acts_buf_ping_pong_state ==  IACTS_PINGPONG_PING_FEED_DPE_FILL_PONG) ?   r_oacts_wr_addr_to_pingpong                     :
                                                                                                                                0   );
    assign  w_iacts_sram_a_wr_en_pong           =   (r_acts_buf_ping_pong_state ==  IACTS_PINGPONG_FILL_PONG)               ?   {IACTS_NUM_BANKS{i_iacts_write_valid}}          :   (
                                                    (r_acts_buf_ping_pong_state ==  IACTS_PINGPONG_PING_FEED_DPE_FILL_PONG) ?   r_mul_with_scale_and_store_oact                 :
                                                                                                                                0   );

    assign  w_iacts_sram_b_rd_addr_pong         =   (r_acts_buf_ping_pong_state ==  IACTS_PINGPONG_PONG_FEED_DPE_FILL_PING) ?   {IACTS_NUM_BANKS{r_iacts_pingpong_rd_addr}}     :   (
                                                    (r_acts_buf_ping_pong_state ==  IACTS_PINGPONG_DRAIN_PONG)              ?   {IACTS_NUM_BANKS{i_oacts_read_addr}}            :
                                                                                                                                0   );

    assign  w_iacts_sram_b_rd_en_pong           =   (r_acts_buf_ping_pong_state ==  IACTS_PINGPONG_PONG_FEED_DPE_FILL_PING) ?   ~0                                              :   (
                                                    (r_acts_buf_ping_pong_state ==  IACTS_PINGPONG_DRAIN_PONG)              ?   {IACTS_NUM_BANKS{i_oacts_read_valid}}           :
                                                                                                                                0   );


    assign  w_iacts_from_ctrl_to_dpe            =   (r_acts_buf_ping_pong_state ==  IACTS_PINGPONG_PING_FEED_DPE_FILL_PONG) ?   i_iacts_sram_b_rd_data_ping                     :   (
                                                    (r_acts_buf_ping_pong_state ==  IACTS_PINGPONG_PONG_FEED_DPE_FILL_PING) ?   i_iacts_sram_b_rd_data_pong                     :
                                                                                                                                0   );

    assign  w_iacts_valid_from_ctrl_to_dpe      =   (r_acts_buf_ping_pong_state ==  IACTS_PINGPONG_PING_FEED_DPE_FILL_PONG) ?   ~0                                              :   (
                                                    (r_acts_buf_ping_pong_state ==  IACTS_PINGPONG_PONG_FEED_DPE_FILL_PING) ?   ~0                                              :
                                                                                                                                0   );

    assign  w_oacts_read_data                   =   (r_acts_buf_ping_pong_state ==  IACTS_PINGPONG_DRAIN_PONG)              ?   i_iacts_sram_b_rd_data_pong                     :   (
                                                    (r_acts_buf_ping_pong_state ==  IACTS_PINGPONG_DRAIN_PING)              ?   i_iacts_sram_b_rd_data_ping                     :
                                                                                                                                0   );

        // Weigths Ping, A  
    assign  w_weights_sram_a_wr_data_ping       =   (r_weights_buf_ping_pong_state  == WEIGHTS_PINGPONG_FILL_PING)          ?   i_weights_write_data                            :
                                                                                                                                0;
    assign  w_weights_sram_a_wr_addr_ping       =   (r_weights_buf_ping_pong_state  == WEIGHTS_PINGPONG_FILL_PING)          ?   {WEIGHTS_NUM_BANKS{i_weights_write_addr}}       :
                                                                                                                                0;
    assign  w_weights_sram_a_wr_en_ping         =   (r_weights_buf_ping_pong_state  == WEIGHTS_PINGPONG_FILL_PING)          ?   {WEIGHTS_NUM_BANKS{i_weights_write_valid}}      :
                                                                                                                                0;

        // Weigths Ping, B  
    assign  w_weights_sram_b_rd_addr_ping       =   (r_weights_buf_ping_pong_state  == WEIGHTS_PINGPONG_PING_FEED_DPE)      ?   {WEIGHTS_NUM_BANKS{r_weights_pingpong_rd_addr}} :
                                                                                                                                0;
    assign  w_weights_sram_b_rd_en_ping         =   (r_weights_buf_ping_pong_state  == WEIGHTS_PINGPONG_PING_FEED_DPE)      ?   ~0                                              :
                                                                                                                                0;

        // Weigths Pong, A  
    assign  w_weights_sram_a_wr_data_pong       =   (r_weights_buf_ping_pong_state  == WEIGHTS_PINGPONG_FILL_PONG)          ?   i_weights_write_data                            :
                                                                                                                                0;
    assign  w_weights_sram_a_wr_addr_pong       =   (r_weights_buf_ping_pong_state  == WEIGHTS_PINGPONG_FILL_PONG)          ?   {WEIGHTS_NUM_BANKS{i_weights_write_addr}}       :
                                                                                                                                0;
    assign  w_weights_sram_a_wr_en_pong         =   (r_weights_buf_ping_pong_state  == WEIGHTS_PINGPONG_FILL_PONG)          ?   {WEIGHTS_NUM_BANKS{i_weights_write_valid}}      :
                                                                                                                                0;

        // Weigths Pong, B  
    assign  w_weights_sram_b_rd_addr_pong       =   (r_weights_buf_ping_pong_state  == WEIGHTS_PINGPONG_PING_FEED_DPE)      ?   {WEIGHTS_NUM_BANKS{r_weights_pingpong_rd_addr}} :
                                                                                                                                0;
    assign  w_weights_sram_b_rd_en_pong         =   (r_weights_buf_ping_pong_state  == WEIGHTS_PINGPONG_PING_FEED_DPE)      ?   ~0                                              :
                                                                                                                                0;

    assign  w_weights_from_ctrl_to_dpe          =   (r_weights_buf_ping_pong_state  == WEIGHTS_PINGPONG_PING_FEED_DPE)      ?   i_weights_sram_b_rd_data_ping                   :   (
                                                    (r_weights_buf_ping_pong_state  == WEIGHTS_PINGPONG_PONG_FEED_DPE)      ?   i_weights_sram_b_rd_data_pong                   :
                                                                                                                                0   );

    assign  w_weights_valid_from_ctrl_to_dpe    =   (r_weights_buf_ping_pong_state  == WEIGHTS_PINGPONG_PING_FEED_DPE)      ?   ~0                                              :   (
                                                    (r_weights_buf_ping_pong_state  == WEIGHTS_PINGPONG_PONG_FEED_DPE)      ?   ~0                                              :
                                                                                                                                0   );

//__________________________________________________________________________________________________________________________
//__________________________________________________________________________________________________________________________


// output of birrd processing and R/W generation for birrd OUTPUT BUFFER

    genvar GENVAR_birrd_OUTBUS_ITER;
    generate
        for(GENVAR_birrd_OUTBUS_ITER=0; GENVAR_birrd_OUTBUS_ITER < DPE_COL_NUM; GENVAR_birrd_OUTBUS_ITER=GENVAR_birrd_OUTBUS_ITER+1)
        begin:birrd_OUTPUT_BUS_TO_ARRAY
            assign  w_o_birrd_data       [GENVAR_birrd_OUTBUS_ITER]   =   i_data_bus_from_birrd[(GENVAR_birrd_OUTBUS_ITER*PE_OUTPUT_WIDTH)  +:  PE_OUTPUT_WIDTH];

            // extracting w_bypass_to_scale, w_mul_with_scale info from i_outbuf_wr_instr
            assign  w_bypass_to_scale   [GENVAR_birrd_OUTBUS_ITER]   =   i_outbuf_wr_instr[(OUTBUF_SRAM_ADDR_WIDTH + GENVAR_birrd_OUTBUS_ITER)  +: 1];
            assign  w_mul_with_scale    [GENVAR_birrd_OUTBUS_ITER]   =   i_outbuf_wr_instr[(OUTBUF_SRAM_ADDR_WIDTH + OUTBUF_NUM_BANKS + GENVAR_birrd_OUTBUS_ITER)  +: 1];
        end
    endgenerate


    /*
        Output of birrd goes to individual PE_OUTPUT_WIDTH-bits bank SRAMs
    */
    integer GENVAR_COL_SRAM_ITER0, GENVAR_COL_SRAM_ITER;
    genvar GENVAR_COL_SRAM_ITER2;

    always @(posedge clk or negedge rst_n)
    begin
        if(!rst_n)
        begin

            for(GENVAR_COL_SRAM_ITER0=0; GENVAR_COL_SRAM_ITER0<DPE_COL_NUM; GENVAR_COL_SRAM_ITER0=GENVAR_COL_SRAM_ITER0+1)
            begin
                r_o_birrd_data          [GENVAR_COL_SRAM_ITER0]    <=  0;
                r_o_birrd_data_prev     [GENVAR_COL_SRAM_ITER0]    <=  0;
                r_oacts_post_quant      [GENVAR_COL_SRAM_ITER0]    <=  0;
                r_o_birrd_data_for_mul  [GENVAR_COL_SRAM_ITER0]    <=  0;
            end
            r_outbuf_sram_a_wr_addr_temp        <=  0;
            r_outbuf_sram_a_wr_addr             <=  0;
            r_oacts_wr_addr_to_pingpong         <=  0;
            r_outbuf_sram_b_rd_addr             <=  0;
            r_bypass_to_scale                   <=  0;
            r_mul_with_scale                    <=  0;
            r_mul_with_scale_and_store_oact     <=  0;
            r_o_birrd_data_bus_valid            <=  0;
            r_partial_sum_done                  <=  0;
            r_outbuf_sram_a_wr_data             <=  0;
        end
        else
        begin
            //---------------------------------------------------------------------------------------------//


            // registering output of birrd to BANKS
            r_o_birrd_data_bus_valid            <=  i_data_bus_from_birrd_valid;
            r_partial_sum_done                  <=  r_o_birrd_data_bus_valid;

            for(GENVAR_COL_SRAM_ITER=0; GENVAR_COL_SRAM_ITER<DPE_COL_NUM; GENVAR_COL_SRAM_ITER=GENVAR_COL_SRAM_ITER+1)
            begin

                //---------------------------------------------------------------------------------------------//
                //  parsing the instructions
                //  i_outbuf_wr_instr is coming from top
                if(!(w_bypass_to_scale[GENVAR_COL_SRAM_ITER]))
                begin
                    r_outbuf_sram_a_wr_addr_temp[(GENVAR_COL_SRAM_ITER*OUTBUF_SRAM_BANK_ADDR_WIDTH) +:  OUTBUF_SRAM_BANK_ADDR_WIDTH] <=  i_outbuf_wr_instr           [(GENVAR_COL_SRAM_ITER*OUTBUF_SRAM_BANK_ADDR_WIDTH) +:  OUTBUF_SRAM_BANK_ADDR_WIDTH];
                    r_outbuf_sram_a_wr_addr     [(GENVAR_COL_SRAM_ITER*OUTBUF_SRAM_BANK_ADDR_WIDTH) +:  OUTBUF_SRAM_BANK_ADDR_WIDTH] <=  r_outbuf_sram_a_wr_addr_temp[(GENVAR_COL_SRAM_ITER*OUTBUF_SRAM_BANK_ADDR_WIDTH) +:  OUTBUF_SRAM_BANK_ADDR_WIDTH];
                    r_outbuf_sram_b_rd_addr     [(GENVAR_COL_SRAM_ITER*OUTBUF_SRAM_BANK_ADDR_WIDTH) +:  OUTBUF_SRAM_BANK_ADDR_WIDTH] <=  i_outbuf_wr_instr           [(GENVAR_COL_SRAM_ITER*OUTBUF_SRAM_BANK_ADDR_WIDTH) +:  OUTBUF_SRAM_BANK_ADDR_WIDTH];
                end
                else
                begin
                    r_outbuf_sram_a_wr_addr     [(GENVAR_COL_SRAM_ITER*OUTBUF_SRAM_BANK_ADDR_WIDTH) +:  OUTBUF_SRAM_BANK_ADDR_WIDTH] <=  i_outbuf_wr_instr           [(GENVAR_COL_SRAM_ITER*OUTBUF_SRAM_BANK_ADDR_WIDTH) +:  OUTBUF_SRAM_BANK_ADDR_WIDTH];
                end
                r_bypass_to_scale               [GENVAR_COL_SRAM_ITER] <=  w_bypass_to_scale[GENVAR_COL_SRAM_ITER];
                r_mul_with_scale                [GENVAR_COL_SRAM_ITER] <=  w_mul_with_scale;
                r_mul_with_scale_and_store_oact [GENVAR_COL_SRAM_ITER] <=  r_mul_with_scale;

                //  registering output of birrd
                //  so that (1) data is registered to be aligned with r_o_birrd_data_prev and (2) registering so that timing is better w.r.t. adder
                if(i_data_bus_from_birrd_valid[GENVAR_COL_SRAM_ITER])
                begin
                    r_o_birrd_data[GENVAR_COL_SRAM_ITER]            <=  w_o_birrd_data[GENVAR_COL_SRAM_ITER];
                end

                //  Bypass to scale is disabled => need to add with partial sum stored in that location
                //  Fetch r_o_birrd_data_prev and add with r_o_birrd_data for the banks whose r_bypass_to_scale is high
                //  if r_mul_with_scale of that bank is high, meaning complete sum is ready, so multiply it with scale and send to store the oAct in the iActs ping/pong buffer
                if(!(r_bypass_to_scale[GENVAR_COL_SRAM_ITER]))
                begin
                    if(r_o_birrd_data_bus_valid[GENVAR_COL_SRAM_ITER])
                    begin
                        r_o_birrd_data_prev[GENVAR_COL_SRAM_ITER]   <=  i_outbuf_sram_b_rd_data[(GENVAR_COL_SRAM_ITER*PE_OUTPUT_WIDTH)    +:  PE_OUTPUT_WIDTH];   /*  r_oacts_banks_sram[GENVAR_COL_SRAM_ITER][r_outbuf_sram_a_wr_addr]; */
                    end
                    if(r_partial_sum_done[GENVAR_COL_SRAM_ITER])
                    begin
                        // storing back the partial sum
                        /*  r_oacts_banks_sram[GENVAR_COL_SRAM_ITER][r_outbuf_sram_a_wr_addr]  */
                        r_outbuf_sram_a_wr_data[(GENVAR_COL_SRAM_ITER*PE_OUTPUT_WIDTH)    +:  PE_OUTPUT_WIDTH] <=  r_o_birrd_data_prev[GENVAR_COL_SRAM_ITER] + r_o_birrd_data[GENVAR_COL_SRAM_ITER];

                        //  complete sum is ready, now multiply with scale and then store in iacts ping/pong buffer
                        if(r_mul_with_scale[GENVAR_COL_SRAM_ITER] == 1)
                        begin
                            r_o_birrd_data_for_mul[GENVAR_COL_SRAM_ITER]    <=  r_o_birrd_data_prev[GENVAR_COL_SRAM_ITER] + r_o_birrd_data[GENVAR_COL_SRAM_ITER];
                        end
                    end
                end
                //  Bypass to Scale is high => no need for partial sum, meaning complete sum is ready, so multiply it with scale and send to store the oAct in the iActs ping/pong buffer
                else
                begin
                    if(r_o_birrd_data_bus_valid[GENVAR_COL_SRAM_ITER] == 1)
                    begin
                        /*  r_oacts_banks_sram[GENVAR_COL_SRAM_ITER][r_outbuf_sram_a_wr_addr]  */
                        r_outbuf_sram_a_wr_data[(GENVAR_COL_SRAM_ITER*PE_OUTPUT_WIDTH)    +:  PE_OUTPUT_WIDTH] <=  r_o_birrd_data[GENVAR_COL_SRAM_ITER];
                    end

                    //  complete sum is ready, now multiply with scale and then store in iacts ping/pong buffer
                    if(r_mul_with_scale[GENVAR_COL_SRAM_ITER] == 1)
                    begin
                        r_o_birrd_data_for_mul[GENVAR_COL_SRAM_ITER]    <=  r_o_birrd_data[GENVAR_COL_SRAM_ITER];
                    end
                end

                if(r_mul_with_scale_and_store_oact[GENVAR_COL_SRAM_ITER] == 1)
                begin
                    r_oacts_post_quant[GENVAR_COL_SRAM_ITER]           <=  w_o_birrd_data_post_quant[GENVAR_COL_SRAM_ITER][OACTS_DATA_WIDTH-1:0];  //Extracting OACTS_DATA_WIDTH bits, per column

                    r_oacts_wr_addr_to_pingpong[GENVAR_COL_SRAM_ITER*OUTBUF_SRAM_BANK_ADDR_WIDTH   +:  OUTBUF_SRAM_BANK_ADDR_WIDTH]   <=  r_outbuf_sram_a_wr_addr [GENVAR_COL_SRAM_ITER*OUTBUF_SRAM_BANK_ADDR_WIDTH   +:  OUTBUF_SRAM_BANK_ADDR_WIDTH];
                end

            end // end GENVAR_COL_SRAM_ITER
        end
    end



//____________________________________________________________________________________________________________________________
//Multiplying with scale
    generate
        for(GENVAR_COL_SRAM_ITER2=0; GENVAR_COL_SRAM_ITER2<DPE_COL_NUM; GENVAR_COL_SRAM_ITER2=GENVAR_COL_SRAM_ITER2+1)
        begin:QUANTIZATION_OPERATION_PER_DPE_COLUMN
            assign  w_o_birrd_data_post_quant[GENVAR_COL_SRAM_ITER2] =  r_o_birrd_data_for_mul[GENVAR_COL_SRAM_ITER2]*r_scale_val;

            assign  w_oacts_post_quant[(GENVAR_COL_SRAM_ITER2*OACTS_DATA_WIDTH) +: OACTS_DATA_WIDTH]   = r_oacts_post_quant[GENVAR_COL_SRAM_ITER2];
        end
    endgenerate
//____________________________________________________________________________________________________________________________



// connect to ports

    // Write enable is conditional to r_bypass_to_scale, becase the partial sum can come 1 cycle later (r_partial_sum_done)
    assign  o_outbuf_sram_a_wr_en   =  (r_bypass_to_scale == 1)   ?   r_o_birrd_data_bus_valid
                                                                :   r_partial_sum_done;
    assign  o_outbuf_sram_b_rd_en   =  ~0;  // Tied to high to always read. read data controlled by address

    //  indicate to feather Top that birrd outputs are ready to be written to outbuf banks
    //  now feather top (from outsite) must generate i_outbuf_wr_instr
    assign  o_outbuf_data_wr_rdy    =  i_data_bus_from_birrd_valid;

    assign  o_iacts_from_ctrl_to_dpe            =   w_iacts_from_ctrl_to_dpe            ;
    assign  o_iacts_valid_from_ctrl_to_dpe      =   w_iacts_valid_from_ctrl_to_dpe      ;
    assign  o_weights_from_ctrl_to_dpe          =   w_weights_from_ctrl_to_dpe          ;
    assign  o_weights_valid_from_ctrl_to_dpe    =   w_weights_valid_from_ctrl_to_dpe    ;
    assign  o_iacts_zp_from_ctrl_to_dpe         =   r_iacts_zp                          ;
    assign  o_weights_zp_from_ctrl_to_dpe       =   r_weights_zp                        ;
    assign  o_weights_ping_pong_sel             =   r_weights_ping_pong_sel             ;
    assign  o_pe_sel                            =   r_pe_sel                            ;
    assign  o_weights_to_use                    =   r_weights_to_use                    ;
    assign  o_birrd_instr                       =   r_birrd_instr                       ;
    assign  o_iacts_sram_a_wr_data_ping         =   w_iacts_sram_a_wr_data_ping         ;
    assign  o_iacts_sram_a_wr_addr_ping         =   w_iacts_sram_a_wr_addr_ping         ;
    assign  o_iacts_sram_a_wr_en_ping           =   w_iacts_sram_a_wr_en_ping           ;
    assign  o_iacts_sram_b_rd_addr_ping         =   w_iacts_sram_b_rd_addr_ping         ;
    assign  o_iacts_sram_b_rd_en_ping           =   w_iacts_sram_b_rd_en_ping           ;
    assign  o_iacts_sram_a_wr_data_pong         =   w_iacts_sram_a_wr_data_pong         ;
    assign  o_iacts_sram_a_wr_addr_pong         =   w_iacts_sram_a_wr_addr_pong         ;
    assign  o_iacts_sram_a_wr_en_pong           =   w_iacts_sram_a_wr_en_pong           ;
    assign  o_iacts_sram_b_rd_addr_pong         =   w_iacts_sram_b_rd_addr_pong         ;
    assign  o_iacts_sram_b_rd_en_pong           =   w_iacts_sram_b_rd_en_pong           ;
    assign  o_weights_sram_a_wr_data_ping       =   w_weights_sram_a_wr_data_ping       ;
    assign  o_weights_sram_a_wr_addr_ping       =   w_weights_sram_a_wr_addr_ping       ;
    assign  o_weights_sram_a_wr_en_ping         =   w_weights_sram_a_wr_en_ping         ;
    assign  o_weights_sram_b_rd_addr_ping       =   w_weights_sram_b_rd_addr_ping       ;
    assign  o_weights_sram_b_rd_en_ping         =   w_weights_sram_b_rd_en_ping         ;
    assign  o_weights_sram_a_wr_data_pong       =   w_weights_sram_a_wr_data_pong       ;
    assign  o_weights_sram_a_wr_addr_pong       =   w_weights_sram_a_wr_addr_pong       ;
    assign  o_weights_sram_a_wr_en_pong         =   w_weights_sram_a_wr_en_pong         ;
    assign  o_weights_sram_b_rd_addr_pong       =   w_weights_sram_b_rd_addr_pong       ;
    assign  o_weights_sram_b_rd_en_pong         =   w_weights_sram_b_rd_en_pong         ;
    assign  o_outbuf_sram_a_wr_data             =   r_outbuf_sram_a_wr_data             ;
    assign  o_outbuf_sram_a_wr_addr             =   r_outbuf_sram_a_wr_addr             ;
    assign  o_outbuf_sram_b_rd_addr             =   r_outbuf_sram_b_rd_addr             ;

    assign  o_instr_sram_a_wr_data              =   i_instr_write_data                  ;
    assign  o_instr_sram_a_wr_addr              =   i_instr_write_addr                  ;
    assign  o_instr_sram_a_wr_en                =   i_instr_write_valid                 ;
    assign  o_instr_sram_b_rd_addr              =   r_iacts_pingpong_rd_addr            ;
    assign  o_instr_sram_b_rd_en                =   (|(w_iacts_sram_b_rd_en_ping))|(|(w_iacts_sram_b_rd_en_pong));
    assign  o_oacts_read_data                   =   w_oacts_read_data                   ;

endmodule
