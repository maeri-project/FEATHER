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
    Top Module:  Feather NEST's PE
    Data:        Only data width matters.
    Format:      keeping the input format unchanged
    Timing:      Sequential Logic
    Reset:       Asynchronized Reset [Low Reset]

    Function:    MAC Unit of the Feather
*/

`timescale 1ns / 1ps

module feather_pe #(
    parameter THIS_PE_ID            = 0 ,   //  ID/Sel for this PE
    parameter IACTS_DATA_WIDTH      = 8 ,   //  Iacts data width
    parameter WEIGHTS_DATA_WIDTH    = 8 ,   //  Weights data width
    parameter WEIGHTS_DEPTH         = 4 ,   //  depth of the PE's internal weights local buffer
    parameter LOG2_WEIGHTS_DEPTH    = 2 ,   //  Log2 of WEIGHTS_DEPTH for address register
    parameter PE_SEL_WIDTH          = 2 ,   //  Log2 of WEIGHTS_DEPTH for address register
    parameter PE_OUTPUT_WIDTH       = 32    //  Width of PE output

)(
    // timing signals
    clk,
    rst_n,

    // data signals
    i_iacts,                                        //  Iacts data
    i_iacts_valid,                                  //  Iacts data valid to register the data and control flow
    i_weights,                                      //  Weights data
    i_weights_valid,                                //  Weights data valid to register the data and control flow
    i_iacts_zp,                                     //  Iacts Zero Point
    i_weights_zp,                                   //  Weights Zero Point

    // control signals for localbuffer+pe operation
    i_weights_ping_pong_sel,                        //  Weights internal Ping-pong buffer selection
    i_pe_sel,                                       //  PE selection/Enable
    i_weights_to_use,                               //  Total number of weights that are to be used
    o_weights_ping_pong_sel,                        //  registered output of i_weights_ping_pong_sel, to next PE
    o_pe_sel,                                       //  registered output of i_pe_sel, to next PE
    o_weights_to_use,                               //  registered output of i_weights_to_use, to next PE

    o_iacts,                                        //  registered output of i_iacts, to next PE
    o_iacts_valid,                                  //  registered output of i_iacts_valid, to next PE
    o_weights,                                      //  registered output of i_weights, to next PE
    o_weights_valid,                                //  registered output of i_weights_valid, to next PE

    o_out_data,                                     //  Computed PE output
    o_out_data_valid                                //  data valid for the PE output
);


    /*
        ports
    */
    input                                   clk;
    input                                   rst_n;
    input    [IACTS_DATA_WIDTH   -1: 0]     i_iacts;
    input                                   i_iacts_valid;
    input    [WEIGHTS_DATA_WIDTH -1: 0]     i_weights;
    input                                   i_weights_valid;
    input    [IACTS_DATA_WIDTH   -1: 0]     i_iacts_zp;
    input    [WEIGHTS_DATA_WIDTH -1: 0]     i_weights_zp;

    input                                   i_weights_ping_pong_sel;
    input   [PE_SEL_WIDTH        -1: 0]     i_pe_sel;
    input   [LOG2_WEIGHTS_DEPTH  -1: 0]     i_weights_to_use;
    output                                  o_weights_ping_pong_sel;
    output  [PE_SEL_WIDTH        -1: 0]     o_pe_sel;
    output  [LOG2_WEIGHTS_DEPTH  -1: 0]     o_weights_to_use;

    output  [IACTS_DATA_WIDTH    -1: 0]     o_iacts;
    output                                  o_iacts_valid;
    output  [WEIGHTS_DATA_WIDTH  -1: 0]     o_weights;
    output                                  o_weights_valid;
    output  [PE_OUTPUT_WIDTH     -1: 0]     o_out_data;
    output                                  o_out_data_valid;

    /*
        inner logics
    */
    reg     [IACTS_DATA_WIDTH                               -1: 0]      r_i_iacts_zp;
    reg     [WEIGHTS_DATA_WIDTH                             -1: 0]      r_i_weights_zp;
    reg     [WEIGHTS_DATA_WIDTH                             -1: 0]      r_local_weights_buffer_ping[WEIGHTS_DEPTH-1:0];
    reg     [WEIGHTS_DATA_WIDTH                             -1: 0]      r_local_weights_buffer_pong[WEIGHTS_DEPTH-1:0];
    reg     [LOG2_WEIGHTS_DEPTH                             -1: 0]      r_weights_wr_cntr;
    reg     [PE_OUTPUT_WIDTH                                -1: 0]      r_sum;
    reg                                                                 r_weights_ping_pong_sel;
    reg     [PE_SEL_WIDTH                                   -1: 0]      r_pe_sel;
    reg     [LOG2_WEIGHTS_DEPTH                             -1: 0]      r_weights_sel_for_iacts_use;
    reg     [LOG2_WEIGHTS_DEPTH                             -1: 0]      r_weights_to_use;
    reg     [IACTS_DATA_WIDTH                               -1: 0]      r_iacts;
    reg                                                                 r_iacts_valid;
    reg     [WEIGHTS_DATA_WIDTH                             -1: 0]      r_weights;
    reg                                                                 r_weights_valid;
    reg     [PE_OUTPUT_WIDTH                                -1: 0]      r_out_data;

    wire    [IACTS_DATA_WIDTH                               -1: 0]      w_iacts;
    wire    [WEIGHTS_DATA_WIDTH                             -1: 0]      w_selected_weight;
    reg     [WEIGHTS_DATA_WIDTH                             -1: 0]      r_selected_weight;
    wire    [IACTS_DATA_WIDTH  +1                           -1: 0]      w_iacts_sub_zp;
    reg     [IACTS_DATA_WIDTH  +1                           -1: 0]      r_iacts_sub_zp;
    wire    [WEIGHTS_DATA_WIDTH+1                           -1: 0]      w_weights_sub_zp;
    reg     [WEIGHTS_DATA_WIDTH+1                           -1: 0]      r_weights_sub_zp;
    wire    [WEIGHTS_DATA_WIDTH  + IACTS_DATA_WIDTH + 2     -1: 0]      w_mul_iacts_weights;
    reg     [WEIGHTS_DATA_WIDTH  + IACTS_DATA_WIDTH + 2     -1: 0]      r_mul_iacts_weights;
    reg                                                                 r_weight_sel_and_use_is_equal_del_for_reg_weight_sel;
    reg                                                                 r_weight_sel_and_use_is_equal_del_for_reg_zub_zp    ;
    reg                                                                 r_output_ready                                      ;
    reg                                                                 r_next_sum_in_prog                                  ;
    reg                                                                 r_out_data_valid                                    ;

    integer i;
    /*
        register the iacts, weights, zp data
        store and forward the iacts, weights and generate valids
    */
    always @(posedge clk or negedge rst_n)
    begin
        if(!rst_n)
        begin
            r_iacts                         <=  0;
            r_weights                       <=  0;
            r_weights_wr_cntr               <=  0;
            r_weights_ping_pong_sel         <=  0;
            r_pe_sel                        <=  0;
            r_weights_to_use                <=  0;
            r_weights_sel_for_iacts_use     <=  0;
            r_iacts_valid                   <=  0;
            r_weights_valid                 <=  0;

            for (i=0; i<WEIGHTS_DEPTH; i=i+1)
            begin
                r_local_weights_buffer_ping[i]   <= 0;
                r_local_weights_buffer_pong[i]   <= 0;
            end
        end
        else
        begin

            r_iacts_valid               <=  i_iacts_valid;
            r_iacts                     <=  i_iacts;
            r_weights_to_use            <=  i_weights_to_use;

            r_weights_valid             <=  i_weights_valid;
            r_weights                   <=  i_weights;
            r_weights_ping_pong_sel     <=  i_weights_ping_pong_sel;
            r_pe_sel                    <=  i_pe_sel;

            // to track the weight being used for multiplication
            if((i_iacts_valid == 1) | (i_weights_valid == 1))
            begin
                if(r_weights_sel_for_iacts_use < i_weights_to_use)
                begin
                    r_weights_sel_for_iacts_use <=  r_weights_sel_for_iacts_use + 1;
                end
                else
                begin
                    r_weights_sel_for_iacts_use <=  0;
                end
            end

            if(i_weights_valid == 1)
            begin
                if(i_pe_sel == THIS_PE_ID)
                begin
                    if(r_weights_wr_cntr < WEIGHTS_DEPTH)
                    begin
                        if(i_weights_ping_pong_sel == 0)
                        begin
                            r_local_weights_buffer_ping[r_weights_wr_cntr] <=  i_weights;
                        end
                        else
                        begin
                            r_local_weights_buffer_pong[r_weights_wr_cntr] <=  i_weights;
                        end
                        r_weights_wr_cntr   <=  r_weights_wr_cntr + 1;
                    end
                    else
                    begin
                        r_weights_wr_cntr   <=  0;
                    end
                end

            end
        end
    end



    /*
        MAC for the weights and iacts logic
    */
    assign  w_iacts = i_iacts;

    assign w_selected_weight                    =   (i_weights_ping_pong_sel == 1)  ?   r_local_weights_buffer_ping[r_weights_sel_for_iacts_use]
                                                                                    :   r_local_weights_buffer_pong[r_weights_sel_for_iacts_use];

    assign w_iacts_sub_zp                       =   {1'b0,w_iacts} - {1'b0,r_i_iacts_zp};
    assign w_weights_sub_zp                     =   {1'b0,w_selected_weight} - {1'b0,r_i_weights_zp};
    assign w_mul_iacts_weights                  =   r_iacts_sub_zp * r_weights_sub_zp;

    assign w_weight_sel_and_use_is_equal        =   (r_weights_sel_for_iacts_use == i_weights_to_use)   ?   1'b1
                                                                                                        :   1'b0;
    always @(posedge clk or negedge rst_n)
    begin
        if(!rst_n)
        begin
            r_out_data                                              <=  0;
            r_sum                                                   <=  0;
            r_i_iacts_zp                                            <=  0;
            r_i_weights_zp                                          <=  0;
            r_mul_iacts_weights                                     <=  0;
            r_iacts_sub_zp                                          <=  0;
            r_weights_sub_zp                                        <=  0;
            r_selected_weight                                       <=  0;
            r_weight_sel_and_use_is_equal_del_for_reg_weight_sel    <=  0;
            r_weight_sel_and_use_is_equal_del_for_reg_zub_zp        <=  0;
            r_output_ready                                          <=  0;
            r_next_sum_in_prog                                      <=  0;
            r_out_data_valid                                        <=  0;
        end
        else
        begin
            r_i_iacts_zp                <=  i_iacts_zp;
            r_i_weights_zp              <=  i_weights_zp;
            r_selected_weight           <=  w_selected_weight;

            r_iacts_sub_zp              <=  w_iacts_sub_zp;
            r_weights_sub_zp            <=  w_weights_sub_zp;

            r_mul_iacts_weights         <=  w_mul_iacts_weights;

            if(r_output_ready)
            begin
                r_next_sum_in_prog  <=  1;
            end
            else if(r_weight_sel_and_use_is_equal_del_for_reg_zub_zp)
            begin
                r_next_sum_in_prog  <=  0;
            end

            r_weight_sel_and_use_is_equal_del_for_reg_weight_sel    <=   w_weight_sel_and_use_is_equal;
            r_weight_sel_and_use_is_equal_del_for_reg_zub_zp        <=   r_weight_sel_and_use_is_equal_del_for_reg_weight_sel;
            r_output_ready                                          <=   r_weight_sel_and_use_is_equal_del_for_reg_zub_zp;

            if(i_iacts_valid==0)
            begin
                r_sum                   <=  0;  //  flushing the previous sum
            end
            else
            begin

                if( r_output_ready)
                begin
                    r_out_data          <=  r_sum;                              //  sending the sum to output
                    r_sum               <=  r_mul_iacts_weights;                //  flushing the previous sum by overwriting with new product
                    r_out_data_valid    <=  1;
                end
                else if(r_next_sum_in_prog)
                begin
                    r_sum               <=  r_sum + r_mul_iacts_weights;        // MAC
                    r_out_data_valid    <=  0;
                end
            end

        end
    end


    assign  o_out_data_valid            =   r_out_data_valid        ;
    assign  o_weights_ping_pong_sel     =   r_weights_ping_pong_sel ;
    assign  o_pe_sel                    =   r_pe_sel                ;
    assign  o_weights_to_use            =   r_weights_to_use        ;
    assign  o_iacts                     =   r_iacts                 ;
    assign  o_iacts_valid               =   r_iacts_valid           ;
    assign  o_weights                   =   r_weights               ;
    assign  o_weights_valid             =   r_weights_valid         ;
    assign  o_out_data                  =   r_out_data              ;

endmodule