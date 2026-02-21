module ALU 
(
    input  logic [31:0] operand1,
    input  logic [31:0] operand2,
    input  logic [ 3:0] operation,
    input  logic        mask,
    output logic [31:0] alu_out
);

logic [31:0] alu_res;

always_comb begin
    case (operation)
        4'b0000: alu_res = operand1 + operand2;                   // vadd.v
        4'b0001: alu_res = operand1 - operand2;                   // vsub.v
        4'b0010: alu_res = operand1 & operand2;                   // vand.v
        4'b0011: alu_res = operand1 | operand2;                   // vor.v
        4'b0100: alu_res = operand1 ^ operand2;                   // vxor.v
        4'b0101: alu_res = operand1 << operand2[4:0];             // vsll.v
        4'b0110: alu_res = operand1 >> operand2[4:0];             // vsrl.v
        4'b0111: alu_res = $signed(operand1) >>> operand2[4:0];   // vsra.v
        default: alu_res = '0;
    endcase

    alu_out = alu_res
    //if mask==0, keep old value
    //alu_out = mask ? alu_res : out_rd_value;
end
endmodule
