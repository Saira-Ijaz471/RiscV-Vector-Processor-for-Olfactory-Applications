# RISC-V Wine Classifier Makefile
# Optimized for 32-bit Fixed-Point Vector Processor

CROSS_COMPILE = riscv64-unknown-elf-
CC = $(CROSS_COMPILE)gcc
CXX = $(CROSS_COMPILE)g++
OBJDUMP = $(CROSS_COMPILE)objdump
OBJCOPY = $(CROSS_COMPILE)objcopy
SIZE = $(CROSS_COMPILE)size

# Architecture: RV32IMV (32-bit Integer + Multiply + Vector)
ARCH = -march=rv32imv -mabi=ilp32

# Compiler flags optimized for fixed-point vector operations
CXXFLAGS = $(ARCH) -O3 -ftree-vectorize -ffast-math \
           -std=c++11 -Wall -ffreestanding -nostdlib \
           -nostartfiles -fno-exceptions -fno-rtti \
           -Wno-unused-variable -fno-builtin

# Linker flags - use rv32im libgcc since rv32imv doesn't have pre-built library
LIBGCC_DIR = /opt/riscv/lib/gcc/riscv64-unknown-elf/15.1.0/rv32im/ilp32

# Target files
TARGET = wine_classifier
CPP_SOURCES = main_adaptive.cpp
ASM_SOURCES = startup.S
OBJECTS = main_adaptive.o startup.o

# Default target
all: $(TARGET).elf $(TARGET).bin $(TARGET).asm $(TARGET).hex
	@echo ""
	@echo "════════════════════════════════════════════════"
	@echo "✅ Build Complete!"
	@echo "════════════════════════════════════════════════"
	@$(SIZE) $(TARGET).elf
	@echo ""
	@echo "📊 Generated Files:"
	@ls -lh $(TARGET).elf $(TARGET).bin $(TARGET).hex $(TARGET).asm 2>/dev/null | awk '{print "  " $$9 " (" $$5 ")"}'
	@echo ""
	@echo "🔍 Quick Vector Check:"
	@echo -n "  Vector instructions found: "
	@grep -cE '\s(v[a-z]+\.|vsetvli|vle[0-9]+\.v|vse[0-9]+\.v)' $(TARGET).asm 2>/dev/null || echo "0"
	@echo ""

# Compile C++ source
%.o: %.cpp
	@echo "🔨 Compiling $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Assemble startup code
%.o: %.S
	@echo "🔧 Assembling $<..."
	$(CC) $(ARCH) -c $< -o $@

# Link ELF executable
$(TARGET).elf: $(OBJECTS)
	@echo "🔗 Linking $@..."
	$(CXX) $(ARCH) -nostdlib -nostartfiles -T linker.ld -Wl,-Map,wine_classifier.map $(OBJECTS) -L$(LIBGCC_DIR) -lgcc -o $@

# Create raw binary (for direct memory loading)
$(TARGET).bin: $(TARGET).elf
	@echo "📦 Creating binary $@..."
	$(OBJCOPY) -O binary $< $@

# Create Intel HEX format (for some bootloaders/programmers)
$(TARGET).hex: $(TARGET).elf
	@echo "📦 Creating HEX file $@..."
	$(OBJCOPY) -O ihex $< $@

# Create assembly listing
$(TARGET).asm: $(TARGET).elf
	@echo "📝 Creating assembly listing $@..."
	$(OBJDUMP) -D $< > $@

# Detailed vector instruction analysis
check-vector: $(TARGET).asm
	@echo "════════════════════════════════════════════════"
	@echo "🔍 Vector Instruction Analysis"
	@echo "════════════════════════════════════════════════"
	@echo ""
	@echo "📊 All Vector Instructions (sorted by count):"
	@echo "────────────────────────────────────────────────"
	@grep -oE '\s(v[a-z]+\.[a-z0-9]+|vsetvli|vle[0-9]+\.v|vse[0-9]+\.v)' $(TARGET).asm | sort | uniq -c | sort -rn
	@echo ""
	@echo "════════════════════════════════════════════════"
	@echo "📋 Unique vector operations:"
	@echo "════════════════════════════════════════════════"
	@grep -oE '\bv[a-z]+\.[a-z0-9]+\b|\bvsetvli\b|\bvle[0-9]+\.v\b|\bvse[0-9]+\.v\b' $(TARGET).asm | sort -u
	@echo ""

# Show vectorized functions
show-functions: $(TARGET).asm
	@echo "════════════════════════════════════════════════"
	@echo "📍 Vectorized Function Locations"
	@echo "════════════════════════════════════════════════"
	@echo ""
	@grep -n "<dot_product_rvv>:" $(TARGET).asm && echo "  ✅ dot_product_rvv found" || echo "  ❌ dot_product_rvv not found"
	@grep -n "<squared_distance_rvv>:" $(TARGET).asm && echo "  ✅ squared_distance_rvv found" || echo "  ❌ squared_distance_rvv not found"
	@grep -n "<standardize_rvv>:" $(TARGET).asm && echo "  ✅ standardize_rvv found" || echo "  ❌ standardize_rvv not found"
	@grep -n "<predict_fixed>:" $(TARGET).asm && echo "  ✅ predict_fixed found" || echo "  ❌ predict_fixed not found"
	@grep -n "<initialize_fixed_model>:" $(TARGET).asm && echo "  ✅ initialize_fixed_model found" || echo "  ❌ initialize_fixed_model not found"
	@echo ""
	@echo "💡 To view a function's assembly:"
	@echo "   grep -A 100 '<function_name>:' $(TARGET).asm | less"
	@echo ""

# Quick deployment info
deploy-info: $(TARGET).bin
	@echo "════════════════════════════════════════════════"
	@echo "🚀 Deployment Information"
	@echo "════════════════════════════════════════════════"
	@echo ""
	@echo "Binary file: $(TARGET).bin"
	@echo "Size: $$(wc -c < $(TARGET).bin) bytes"
	@echo ""
	@echo "Memory Layout:"
	@echo "  ROM Start:  0x80000000"
	@echo "  ROM Size:   1 MB"
	@echo "  RAM Start:  0x80100000"
	@echo "  RAM Size:   512 KB"
	@echo ""
	@echo "Loading Instructions:"
	@echo "  1. FPGA/ASIC: Load $(TARGET).bin at ROM address"
	@echo "  2. JTAG:      openocd -c 'program $(TARGET).bin 0x80000000'"
	@echo "  3. Bootloader: Load via your custom bootloader"
	@echo ""
	@echo "Required Hardware Interface:"
	@echo "  Input:  6 sensors @ 0x10000000-0x10000014 (32-bit each)"
	@echo "  Output: Classification result @ 0x20000000"
	@echo ""

# Memory map analysis
memory-map: $(TARGET).elf
	@echo "════════════════════════════════════════════════"
	@echo "🗺️  Memory Map"
	@echo "════════════════════════════════════════════════"
	@$(SIZE) -A $(TARGET).elf
	@echo ""
	@echo "Detailed sections:"
	@grep -A 20 "Memory Configuration" $(TARGET).map 2>/dev/null || echo "  Map file available at wine_classifier.map"

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -f *.o *.elf *.bin *.hex *.asm *.map wine_classifier_test
	@echo "✅ Clean complete!"

# Help target
help:
	@echo "════════════════════════════════════════════════"
	@echo "Wine Classifier Build System - 32-bit Fixed-Point Vector"
	@echo "════════════════════════════════════════════════"
	@echo ""
	@echo "Targets:"
	@echo "  make                - Build all (default)"
	@echo "  make clean          - Remove build artifacts"
	@echo "  make check-vector   - Analyze vector instructions"
	@echo "  make show-functions - Show function locations"
	@echo "  make deploy-info    - Show deployment instructions"
	@echo "  make memory-map     - Show memory usage"
	@echo "  make help           - Show this help"
	@echo ""
	@echo "Architecture: rv32imv (32-bit + Integer + Multiply + Vector)"
	@echo "Fixed-Point: Q16.16 format (16 int bits, 16 frac bits)"
	@echo ""
	@echo "Generated Files:"
	@echo "  wine_classifier.elf - ELF executable with headers"
	@echo "  wine_classifier.bin - Raw binary for hardware"
	@echo "  wine_classifier.hex - Intel HEX format"
	@echo "  wine_classifier.asm - Assembly listing"
	@echo "  wine_classifier.map - Memory map"
	@echo ""

.PHONY: all clean check-vector show-functions deploy-info memory-map help