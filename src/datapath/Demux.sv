module Deux
(
   input  logic [31:0] a,
   input  logic        sel,
   output logic [31:0] b,
   output logic [31:0] c
);

always_comb begin

   if (sel) begin
      c = a; 
   end 

   else begin
      b = a; 
   end

end
endmodule