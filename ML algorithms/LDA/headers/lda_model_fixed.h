#ifndef LDA_MODEL_FIXED_H
#define LDA_MODEL_FIXED_H

#include <stdint.h>

// =============================================
// LDA Model Parameters - Auto-generated
// Training pipeline: Raw → Standardize → PCA → LDA
// All values in Q16.16 fixed-point format
// =============================================

#define NUM_RAW_FEATURES 6      // Original sensor features
#define NUM_PCA_COMPONENTS 6    // After PCA dimensionality reduction
#define NUM_CLASSES 4           // Number of output classes

// =============================================
// Class Names
// =============================================
static const char* class_names[NUM_CLASSES] = {
    "AQ_Wines",
    "Ethanol",
    "HQ_Wines",
    "LQ_Wines"
};

// =============================================
// STEP 1: Standardization Parameters
// =============================================

// Feature means (Q16.16 fixed-point)
static const int32_t feature_means_fixed_data[NUM_RAW_FEATURES] = {
    899165, 2019914, 3970558, 2969023, 1622881, 3081375
};

// Feature inverse standard deviations (Q16.16 fixed-point)
// Pre-computed as 1/stddev for efficiency
static const int32_t feature_stddevs_inv_fixed_data[NUM_RAW_FEATURES] = {
    1917, 5177, 1453, 560, 4774, 1671
};

// =============================================
// STEP 2: PCA Projection Matrix
// =============================================

// PCA matrix: [NUM_RAW_FEATURES][NUM_PCA_COMPONENTS] (Q16.16 fixed-point)
// Usage: x_pca[j] = sum(x_std[i] * pca_matrix[i][j])
static const int32_t pca_matrix_fixed_data[NUM_RAW_FEATURES][NUM_PCA_COMPONENTS] = {
    {23778, 39113, -9484, -44166, -12566, -1090},
    {27012, -21719, -43286, -4071, 28813, 19319},
    {28701, -14968, 24410, -9428, 26192, -43317},
    {23509, 40072, 535, 43907, 14387, 1174},
    {28544, -17533, -14885, 17311, -48728, -16649},
    {28438, -12497, 38906, -3360, -6280, 42019}
};

// =============================================
// STEP 3: LDA Classification
// =============================================

// LDA weights: [NUM_CLASSES][NUM_PCA_COMPONENTS] (Q16.16 fixed-point)
// Each row represents weights for one class
static const int32_t lda_weights_fixed_data[NUM_CLASSES][NUM_PCA_COMPONENTS] = {
    {4774, -2725, -115515, 172437, 329567, 398413},
    {8252, -13545, -27003, -93676, 37971, -176924},
    {11543, -24592, -65730, 177718, -73750, 366532},
    {-24569, 40863, 208248, -256479, -293788, -588021}
};

// LDA bias: [NUM_CLASSES] (Q16.16 fixed-point)
// One bias value per class
static const int32_t lda_bias_fixed_data[NUM_CLASSES] = {
    -114120, -96785, -106139, -157507
};

#endif // LDA_MODEL_FIXED_H
