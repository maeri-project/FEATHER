/*
    Top Module:  Generic Single Port SRAM Bank
    Data:        Only data width matters.
    Format:      keeping the input format unchanged
    Timing:      Sequential Logic
    Reset:       Asynchronized Reset [Low Reset]

    Function:    Single, Common RD/WR enable to control read and write

    Author:      Jianming Tong (jianming.tong@gatech.edu), Anirudh Itagi (aitagi7@gatech.edu)
*/

`timescale 1ns / 1ps

module sram_bank_sp #(
    parameter SRAM_BANK_DATA_WIDTH      = 8  ,                      //
    parameter SRAM_BANK_ADDR_WIDTH      = 10 ,                      //
    parameter SRAM_BANK_DEPTH           = 2**SRAM_BANK_ADDR_WIDTH   //

)(
    clk,
    rst_n,

    // Single Port
    i_rd_wr_en,
    i_addr,
    i_wr_data,
    o_rd_data
);


    /*
        ports
    */
    input                                   clk;
    input                                   rst_n;

    input                                   i_rd_wr_en;
    input   [SRAM_BANK_ADDR_WIDTH   -1: 0]  i_addr;
    input   [SRAM_BANK_DATA_WIDTH   -1: 0]  i_wr_data;
    output  [SRAM_BANK_DATA_WIDTH   -1: 0]  o_rd_data;

    /*
        inner logics
    */
    reg     [SRAM_BANK_DATA_WIDTH   -1: 0]  r_sram_bank [0  :   SRAM_BANK_DEPTH-1];
    reg     [SRAM_BANK_DATA_WIDTH   -1: 0]  r_o_rd_data;

    integer i;


    /*
        Dual Port SRAM
        ->  i_rd_wr_en == 1 => Write , else Read
    */
    always @(posedge clk or negedge rst_n)
    begin
        if(!rst_n)
        begin
            for (i=0; i<SRAM_BANK_DEPTH; i=i+1)
            begin
                r_sram_bank[i]   <=  0;
            end
        end
        else
        begin
            if(i_rd_wr_en==1)
            begin
                r_sram_bank[$unsigned(i_addr)]  <=  i_wr_data;
            end
        end
    end


    always @(posedge clk or negedge rst_n)
    begin
        if(!rst_n)
        begin
            r_o_rd_data        <=  0;
        end
        else
        begin
            if(i_rd_wr_en==0)
            begin
                r_o_rd_data    <=  r_sram_bank[$unsigned(i_addr)];
            end
        end
    end

    assign  o_rd_data    =  r_o_rd_data;

endmodule