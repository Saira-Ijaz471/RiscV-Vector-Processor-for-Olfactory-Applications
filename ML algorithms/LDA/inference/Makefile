# ============================================
# Makefile for LDA Fixed-Point Inference
# RISC-V Bare-Metal Production Build
# ============================================

# Toolchain prefix
PREFIX = riscv32-unknown-elf-
CC     = $(PREFIX)gcc
OBJCOPY = $(PREFIX)objcopy
OBJDUMP = $(PREFIX)objdump
SIZE   = $(PREFIX)size

# Target architecture and ABI
ARCH = rv32imv_zicsr_zifencei
ABI  = ilp32

# Compiler flags
CFLAGS  = -march=$(ARCH) -mabi=$(ABI)
CFLAGS += -Wall -Wextra -std=c99
CFLAGS += -O2 -ffunction-sections -fdata-sections
CFLAGS += -nostdlib -nostartfiles
CFLAGS += -ffreestanding

# Linker flags
LDFLAGS  = -march=$(ARCH) -mabi=$(ABI)
LDFLAGS += -nostdlib -nostartfiles
LDFLAGS += -T linker.ld
LDFLAGS += -Wl,--gc-sections
LDFLAGS += -Wl,-Map=$(TARGET).map

# Target executable
TARGET = lda_inference

# Source files
C_SRCS   = main.c
ASM_SRCS = startup.S
SRCS     = $(C_SRCS) $(ASM_SRCS)

# Object files
OBJS = $(SRCS:.c=.o)
OBJS := $(OBJS:.S=.o)

# Header dependencies
DEPS = lda_model_fixed.h

# Output formats
ELF = $(TARGET).elf
BIN = $(TARGET).bin
HEX = $(TARGET).hex
LST = $(TARGET).lst

# ============================================
# Main targets
# ============================================

.PHONY: all clean size asm

all: $(ELF) $(BIN) $(HEX) $(LST)
	@echo "✓ Build complete"
	@$(SIZE) $(ELF)

# Link all objects into ELF
$(ELF): $(OBJS) linker.ld
	@echo "Linking $(ELF)..."
	$(CC) $(LDFLAGS) -o $@ $(OBJS)

# Generate raw binary
$(BIN): $(ELF)
	@echo "Creating binary $(BIN)..."
	$(OBJCOPY) -O binary $< $@

# Generate Intel HEX
$(HEX): $(ELF)
	@echo "Creating hex file $(HEX)..."
	$(OBJCOPY) -O ihex $< $@

# Generate disassembly listing
$(LST): $(ELF)
	@echo "Creating listing $(LST)..."
	$(OBJDUMP) -D -S $< > $@

# Compile C source files
%.o: %.c $(DEPS)
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) -c $< -o $@

# Assemble .S files
%.o: %.S
	@echo "Assembling $<..."
	$(CC) $(CFLAGS) -c $< -o $@

# Generate assembly from C source
asm: $(C_SRCS)
	@echo "Generating assembly with vector extension..."
	@for src in $(C_SRCS); do \
		base=$${src%.c}; \
		$(CC) $(CFLAGS) -S $$src -o $$base.s; \
	done

# Show memory usage
size: $(ELF)
	@echo "Memory usage:"
	@$(SIZE) $(ELF)

# Clean build artifacts
clean:
	@echo "Cleaning..."
	rm -f $(OBJS) $(ELF) $(BIN) $(HEX) $(LST) $(TARGET).map *.s
	@echo "✓ Clean complete"
