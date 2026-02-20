#!/bin/bash
set -e  # Exit immediately if any command fails

echo "=============================================="
echo "🚀 PCA + LDA Training Pipeline (Shark + ALGLIB)"
echo "=============================================="

# ---- Paths ----
PROJECT_ROOT="$HOME/LDA"
SRC_DIR="$PROJECT_ROOT/train"
BUILD_DIR="$PROJECT_ROOT/build"
DATASET_PATH="$PROJECT_ROOT/dataset"

SRC="$SRC_DIR/train.cpp"
EXEC="$BUILD_DIR/train_exec"

# ---- Shark paths ----
SHARK_INC="/home/tayyba/Shark/install/include"
SHARK_LIB="/home/tayyba/Shark/install/lib"

# ---- ALGLIB paths ----
ALGLIB_INC="/home/tayyba/alglib-cpp/src"

# Find all ALGLIB cpp files
ALGLIB_CPP=$(find /home/tayyba/alglib-cpp/src -name "*.cpp")

# ---- Sanity checks ----
echo "🔎 Running sanity checks..."

for item in "$SRC"; do
    if [ ! -f "$item" ]; then
        echo "❌ Missing required file: $item"
        exit 1
    fi
done

if [ ! -d "$DATASET_PATH" ]; then
    echo "❌ Dataset directory not found: $DATASET_PATH"
    exit 1
fi

echo "✅ All required files found"
echo

# ---- Build ----
mkdir -p "$BUILD_DIR"

echo "🛠 Compiling train.cpp with Shark + ALGLIB..."

g++ -std=c++17 -O2 -Wall -Wextra -fopenmp \
    -Wno-deprecated-copy \
    -Wno-unused-parameter \
    -Wno-unused-variable \
    -Wno-unused-function \
    -Wno-strict-aliasing \
    -I"$SRC_DIR" \
    -I"$SHARK_INC" \
    -I"$ALGLIB_INC" \
    "$SRC" \
    $ALGLIB_CPP \
    "$SHARK_LIB/libshark.a" \
    -llapack -lblas \
    -lboost_system -lboost_filesystem -lboost_serialization \
    -o "$EXEC"

echo "✅ Compilation successful"
echo

# ---- Run ----
echo "▶ Running training pipeline with dataset:"
echo "   $DATASET_PATH"
echo "----------------------------------------------"

"$EXEC" "$DATASET_PATH"

echo "----------------------------------------------"
echo "✅ Pipeline finished successfully"
