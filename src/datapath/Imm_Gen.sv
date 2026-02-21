module Imm_Gen
(
   input  logic [31:0] instruction,
   output logic [31:0] imm
);

always_comb begin

    if (instruction[6:0] == 7'b0000011 || instruction[6:0] == 7'b0010011) begin
        imm = { { 20{ instruction[31] } } , instruction[31:20] }; // I-Type (load)
    end

    else if (instruction[6:0] == 7'b0100011) begin
        imm = { { 20{ instruction[31] } } , instruction[31:25] , instruction[11:7] }; // S-Type (store)
    end

    else if (instruction[6:0] == 7'b1100011) begin
        imm = { { 19{ instruction[31] } } , instruction[31] , instruction[7] , instruction[30:25] , instruction[11:8], 1'b0}; // B-Type (branch)
    end

    else if (instruction[6:0] == 7'b1101111) begin
        imm = { { 19{ instruction[31] } } , instruction[31] , instruction[19:12] , instruction[20] , instruction[30:21], 1'b0}; // J-Type (jump)
    end

    else if (instruction[6:0] == 7'b0110111) begin
        imm = { instruction[31:12], { 12{ 1'b0 } } }; // U-Type (Upper imm)
    end

    else if (instruction[19:15] == 7'b1010111) begin
        imm = { { 27{ instruction[19] } } , instruction[19:15] }; // Vector-OPIVI
    end

    else begin
        imm = 32'b0;
    end

end
endmodule