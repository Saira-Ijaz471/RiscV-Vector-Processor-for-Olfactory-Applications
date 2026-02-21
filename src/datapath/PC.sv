module PC
(
   input  logic [31:0] in_address,
   input  logic        clk,
   input  logic        rst,
   output logic [31:0] out_address
);

always_ff @(posedge clk or negedge rst) begin

   if (!rst) begin
      out_address <= 32'b0; 
   end 

   else begin
      out_address <= in_address; 
   end

end
endmodule