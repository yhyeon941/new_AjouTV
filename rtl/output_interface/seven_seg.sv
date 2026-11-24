
`timescale 1ns / 1ps

// seven_seg_controller
// Basys3 4-digit 7-Segment : "AJOU" always ON
//
// Clock   : 100 MHz
// seg[0]  : a
// seg[1]  : b
// ...
// seg[6]  : g
//
// Active-Low : 0 = ON, 1 = OFF
// Left to right : A J O U
//
// No reset needed: operates independently of main system FSM.

module seven_seg_controller (
    input  wire       clk,
    output reg  [6:0] seg,
    output reg  [3:0] an,
    output wire       dp
);

    // 100 MHz clock
    // Each digit refreshed at ~1.526 kHz
    reg [15:0] refresh_count = 16'd0;

    always @(posedge clk) begin
        refresh_count <= refresh_count + 16'd1;
    end

    wire [1:0] digit_sel = refresh_count[15:14];

    // Decimal point OFF
    assign dp = 1'b1;

    always @(*) begin
        an  = 4'b1111;
        seg = 7'b1111111;

        case (digit_sel)
            2'd0: begin
                an  = 4'b1110;  // Rightmost digit
                seg = 7'b1000001; // U
            end

            2'd1: begin
                an  = 4'b1101;
                seg = 7'b1000000; // O
            end

            2'd2: begin
                an  = 4'b1011;
                seg = 7'b1100001; // J
            end

            2'd3: begin
                an  = 4'b0111;  // Leftmost digit
                seg = 7'b0001000; // A
            end
        endcase
    end

endmodule
