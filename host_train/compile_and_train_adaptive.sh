#!/bin/bash

# Wine Classifier - Adaptive Training Script
# This script compiles and runs the adaptive training code to generate model headers

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================"
echo "  Wine Classifier - Adaptive Training Script"
echo -e "================================================${NC}\n"

# Configuration
ALGLIB_PATH="$HOME/alglib-cpp/src"
LIBSVM_PATH="$HOME/alglib-cpp/src/libsvm"
DATASET_DIR="../dataset"  # Relative to host_train directory
OUTPUT_DIR="generated"
OUTPUT_PREFIX="model"

# Training parameters (can be overridden with environment variables)
PCA_COMPONENTS=${PCA_COMPONENTS:-"-1"}      # -1 = auto-select
VARIANCE_THRESHOLD=${VARIANCE_THRESHOLD:-"0.95"}
SVM_C=${SVM_C:-"1.0"}
SVM_GAMMA=${SVM_GAMMA:-"-1"}                # -1 = auto (1/K)

# =====================================================
# Step 0: Environment Check
# =====================================================
echo -e "${CYAN}[0/5] Checking environment...${NC}"

# Check if we're in the right directory
if [ ! -f "train_export_adaptive.cpp" ]; then
    # Try old filename for backward compatibility
    if [ -f "train_export.cpp" ]; then
        echo -e "${YELLOW}Warning: Found train_export.cpp (old name)${NC}"
        echo -e "${YELLOW}Please rename to train_export_adaptive.cpp${NC}"
        TRAIN_FILE="train_export.cpp"
    else
        echo -e "${RED}Error: train_export_adaptive.cpp not found!${NC}"
        echo "Please run this script from the host_train directory"
        exit 1
    fi
else
    TRAIN_FILE="train_export_adaptive.cpp"
fi
echo -e "${GREEN}✓ Training source file found: $TRAIN_FILE${NC}"

# Check if dataset exists
if [ ! -d "$DATASET_DIR" ]; then
    echo -e "${RED}Error: Dataset directory not found!${NC}"
    echo "Expected: $DATASET_DIR/"
    echo "Current directory: $(pwd)"
    exit 1
fi
echo -e "${GREEN}✓ Dataset directory found${NC}"

# Count classes and files
CLASS_COUNT=$(find "$DATASET_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
if [ $CLASS_COUNT -eq 0 ]; then
    echo -e "${RED}Error: No class folders found in dataset!${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Found $CLASS_COUNT classes in dataset${NC}"

# Check if ALGLIB exists
if [ ! -d "$ALGLIB_PATH" ]; then
    echo -e "${RED}Error: ALGLIB not found at $ALGLIB_PATH${NC}"
    echo "Please update ALGLIB_PATH in this script or install ALGLIB:"
    echo "  wget https://www.alglib.net/translator/re/alglib-cpp.zip"
    echo "  unzip alglib-cpp.zip -d ~/alglib-cpp"
    exit 1
fi
echo -e "${GREEN}✓ ALGLIB found${NC}"

# Check if LIBSVM exists
if [ ! -d "$LIBSVM_PATH" ]; then
    echo -e "${RED}Error: LIBSVM not found at $LIBSVM_PATH${NC}"
    echo "Please update LIBSVM_PATH in this script"
    echo "LIBSVM is usually included with ALGLIB at: ~/alglib-cpp/src/libsvm"
    exit 1
fi
echo -e "${GREEN}✓ LIBSVM found${NC}\n"

# =====================================================
# Step 1: Create symbolic links
# =====================================================
echo -e "${CYAN}[1/5] Creating symbolic links...${NC}"
ln -sf "$ALGLIB_PATH" alglib
ln -sf "$LIBSVM_PATH/svm.h" ./svm.h
echo -e "${GREEN}✓ Symbolic links created${NC}\n"

# =====================================================
# Step 2: Create output directory
# =====================================================
echo -e "${CYAN}[2/5] Creating output directory...${NC}"
mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}✓ Output directory ready: $OUTPUT_DIR/${NC}\n"

# =====================================================
# Step 3: Compile adaptive training code
# =====================================================
echo -e "${CYAN}[3/5] Compiling adaptive training code...${NC}"

# Clean old binary
rm -f train_export train_export_adaptive

g++ -o train_export_adaptive "$TRAIN_FILE" \
    "$ALGLIB_PATH/dataanalysis.cpp" \
    "$ALGLIB_PATH/alglibinternal.cpp" \
    "$ALGLIB_PATH/alglibmisc.cpp" \
    "$ALGLIB_PATH/ap.cpp" \
    "$ALGLIB_PATH/linalg.cpp" \
    "$ALGLIB_PATH/specialfunctions.cpp" \
    "$ALGLIB_PATH/statistics.cpp" \
    "$ALGLIB_PATH/optimization.cpp" \
    "$ALGLIB_PATH/solvers.cpp" \
    "$LIBSVM_PATH/svm.cpp" \
    -std=c++17 -O2

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Compilation successful${NC}\n"
else
    echo -e "${RED}✗ Compilation failed${NC}"
    echo "Check for syntax errors in $TRAIN_FILE"
    exit 1
fi

# =====================================================
# Step 4: Display training configuration
# =====================================================
echo -e "${CYAN}[4/5] Training configuration...${NC}"
echo "Dataset path: $DATASET_DIR"
echo "Output prefix: $OUTPUT_DIR/$OUTPUT_PREFIX"
echo "Classes detected: $CLASS_COUNT"
echo ""
echo "Training parameters:"
if [ "$PCA_COMPONENTS" = "-1" ]; then
    echo "  PCA components: auto-select (${VARIANCE_THRESHOLD} variance)"
else
    echo "  PCA components: $PCA_COMPONENTS"
fi
echo "  SVM C: $SVM_C"
if [ "$SVM_GAMMA" = "-1" ]; then
    echo "  SVM gamma: auto (1/K)"
else
    echo "  SVM gamma: $SVM_GAMMA"
fi
echo ""

# =====================================================
# Step 5: Run adaptive training
# =====================================================
echo -e "${CYAN}[5/5] Running adaptive training...${NC}"
echo ""

# Build command with parameters
TRAIN_CMD="./train_export_adaptive $DATASET_DIR $OUTPUT_DIR/$OUTPUT_PREFIX"

if [ "$PCA_COMPONENTS" != "-1" ]; then
    TRAIN_CMD="$TRAIN_CMD --pca-components $PCA_COMPONENTS"
fi

if [ "$VARIANCE_THRESHOLD" != "0.95" ]; then
    TRAIN_CMD="$TRAIN_CMD --variance-threshold $VARIANCE_THRESHOLD"
fi

if [ "$SVM_C" != "1.0" ]; then
    TRAIN_CMD="$TRAIN_CMD --svm-C $SVM_C"
fi

if [ "$SVM_GAMMA" != "-1" ]; then
    TRAIN_CMD="$TRAIN_CMD --svm-gamma $SVM_GAMMA"
fi

echo -e "${YELLOW}Executing: $TRAIN_CMD${NC}\n"

$TRAIN_CMD

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}================================================"
    echo "  ✓ Training completed successfully!"
    echo -e "================================================${NC}\n"
    
    # Show generated files with details
    echo "Generated model headers:"
    ls -lh "$OUTPUT_DIR"/*.h
    
    # Count generated files
    HEADER_COUNT=$(ls "$OUTPUT_DIR"/*.h 2>/dev/null | wc -l)
    echo ""
    echo -e "${GREEN}Generated $HEADER_COUNT header files${NC}"
    
    # Check for config file
    if [ -f "$OUTPUT_DIR/model_config.h" ]; then
        echo ""
        echo "Model Configuration:"
        grep "define NUM_" "$OUTPUT_DIR/model_config.h" | sed 's/#define /  /'
    fi
    
    # Estimate memory footprint
    echo ""
    echo "Estimated Memory Usage:"
    if command -v du &> /dev/null; then
        TOTAL_SIZE=$(du -sh "$OUTPUT_DIR" | cut -f1)
        echo "  Total header size: $TOTAL_SIZE"
    fi
    
    echo ""
    echo -e "${BLUE}================================================"
    echo "  Next Steps"
    echo -e "================================================${NC}"
    echo ""
    echo "1. Review generated headers:"
    echo "   ls -la $OUTPUT_DIR/"
    echo ""
    echo "2. Copy headers to RISC-V inference directory:"
    echo "   ${YELLOW}cp $OUTPUT_DIR/*.h ../riscv_inference/${NC}"
    echo ""
    echo "3. Compile for RISC-V:"
    echo "   ${YELLOW}cd ../riscv_inference && make${NC}"
    echo ""
    echo "OR use the automated build script:"
    echo "   ${YELLOW}cd .. && ./build_adaptive.sh${NC}"
    
else
    echo ""
    echo -e "${RED}================================================"
    echo "  ✗ Training failed"
    echo -e "================================================${NC}"
    echo ""
    echo "Possible issues:"
    echo "  - Dataset files have inconsistent feature dimensions"
    echo "  - No data files found in class folders"
    echo "  - Invalid data format (should be space-separated numbers)"
    echo ""
    echo "Debug tips:"
    echo "  1. Check dataset structure:"
    echo "     tree $DATASET_DIR"
    echo ""
    echo "  2. Verify data format:"
    echo "     head $DATASET_DIR/*/sample*.txt"
    echo ""
    echo "  3. Count features per sample:"
    echo "     awk '{print NF}' $DATASET_DIR/*/*.txt | sort -u"
    echo ""
    exit 1
fi