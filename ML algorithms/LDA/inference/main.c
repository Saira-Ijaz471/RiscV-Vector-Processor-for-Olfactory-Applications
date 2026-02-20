#include <stdint.h>
#include <stddef.h>           
#include "lda_model_fixed.h" 
//#include <stdio.h>
// ============================================
// FIXED-POINT CONFIGURATION
// ============================================
#define FRAC_BITS 16
#define FIXED_ONE (1 << FRAC_BITS)  // 65536 = 1.0 in Q16.16


// ============================================
// FIXED-POINT UTILITIES
// ============================================
static inline int32_t fixed_mul(int32_t a, int32_t b) {
    return (int32_t)(((int64_t)a * b) >> FRAC_BITS);
}

// ============================================
// FIXED-POINT MODEL STORAGE
// ============================================

static int32_t feature_means_fixed[NUM_RAW_FEATURES];
static int32_t feature_stddevs_inv_fixed[NUM_RAW_FEATURES];
static int32_t pca_matrix_fixed[NUM_RAW_FEATURES][NUM_PCA_COMPONENTS];
static int32_t lda_weights_fixed[NUM_CLASSES][NUM_PCA_COMPONENTS];
static int32_t lda_bias_fixed[NUM_CLASSES];


// ============================================
// MODEL INITIALIZATION (called once at startup)
// ============================================
void initialize_model_fixed(void) {
    // Copy feature means
    for (int i = 0; i < NUM_RAW_FEATURES; i++) {
        feature_means_fixed[i] = feature_means_fixed_data[i];
    }
    
    // Copy inverse stddevs
    for (int i = 0; i < NUM_RAW_FEATURES; i++) {
        feature_stddevs_inv_fixed[i] = feature_stddevs_inv_fixed_data[i];
    }
    
    // Copy PCA matrix (2D array)
    for (int i = 0; i < NUM_RAW_FEATURES; i++) {
        for (int j = 0; j < NUM_PCA_COMPONENTS; j++) {
            pca_matrix_fixed[i][j] = pca_matrix_fixed_data[i][j];
        }
    }
    
    // Copy LDA weights (2D array)
    for (int i = 0; i < NUM_CLASSES; i++) {
        for (int j = 0; j < NUM_PCA_COMPONENTS; j++) {
            lda_weights_fixed[i][j] = lda_weights_fixed_data[i][j];
        }
    }
    
    // Copy LDA bias
    for (int i = 0; i < NUM_CLASSES; i++) {
        lda_bias_fixed[i] = lda_bias_fixed_data[i];
    }
}

// ============================================
// STEP 1: STANDARDIZATION
// ============================================
/**
 * Standardize raw input features
 * Formula: x_std[i] = (x_raw[i] - mean[i]) / stddev[i]
 * In fixed-point: x_std[i] = (x_raw[i] - mean[i]) * inv_stddev[i]
 */
void standardize_features(
    const int32_t input[NUM_RAW_FEATURES],
    int32_t output[NUM_RAW_FEATURES]
)
{
    for (int i = 0; i < NUM_RAW_FEATURES; i++) {
        int32_t centered = input[i] - feature_means_fixed[i];
        output[i] = fixed_mul(centered, feature_stddevs_inv_fixed[i]);
    }
}

// ============================================
// STEP 2: PCA PROJECTION
// ============================================
/**
 * Project standardized features onto PCA components
 * Formula: x_pca[j] = sum_i(x_std[i] * pca_matrix[i][j]) for each component j
 * This reduces dimensionality from NUM_RAW_FEATURES to NUM_PCA_COMPONENTS
 */
void pca_project(
    const int32_t input[NUM_RAW_FEATURES],
    int32_t output[NUM_PCA_COMPONENTS]
)
{
    for (int j = 0; j < NUM_PCA_COMPONENTS; j++) {

        int64_t acc = 0;

        for (int i = 0; i < NUM_RAW_FEATURES; i++) {
            acc += (int64_t)input[i] * pca_matrix_fixed[i][j];
        }

        output[j] = (int32_t)(acc >> FRAC_BITS);
    }
}

// ============================================
// STEP 3: LDA SCORING
// ============================================
/**
 * Compute LDA scores for all classes
 * Formula: score[c] = dot(lda_weights[c], x_pca) + lda_bias[c]
 */
void lda_score(
    const int32_t* pca_input,
    int32_t* scores_output
) {
    // For each class
    for (int c = 0; c < NUM_CLASSES; c++) {
        int64_t accumulator = 0;
        
        // Dot product with LDA weights for this class
        for (int j = 0; j < NUM_PCA_COMPONENTS; j++) {
            // Multiply and accumulate
            int64_t product = (int64_t)pca_input[j] * (int64_t)lda_weights_fixed[c][j];
            accumulator += product;
        }
        
        // Scale back from Q32.32 to Q16.16
        int32_t dot_result = (int32_t)(accumulator >> FRAC_BITS);
        
        // Add bias
        scores_output[c] = dot_result + lda_bias_fixed[c];
    }
}

// ============================================
// STEP 4: ARGMAX (Find class with highest score)
// ============================================
/**
 * Find the class with the highest score
 */
int argmax(const int32_t* scores) {
    int best_class = 0;
    int32_t best_score = scores[0];
    
    for (int c = 1; c < NUM_CLASSES; c++) {
        if (scores[c] > best_score) {
            best_score = scores[c];
            best_class = c;
        }
    }
    
    return best_class;
}

// ============================================
// COMPLETE INFERENCE PIPELINE
// ============================================

/**
 * Complete LDA inference pipeline
 * 
 * Pipeline: Raw Input → Standardize → PCA → LDA Score → ArgMax
 * 
 * @param raw_input: Raw sensor features (Q16.16)
 * @return: Predicted class index (0 to NUM_CLASSES-1)
 */
int lda_predict(const int32_t* raw_input) {
    // Temporary buffers for intermediate results
    int32_t standardized[NUM_RAW_FEATURES];
    int32_t pca_projected[NUM_PCA_COMPONENTS];
    int32_t class_scores[NUM_CLASSES];
    
    // Step 1: Standardize raw input
    standardize_features(raw_input, standardized);
    
    // Step 2: PCA projection
    pca_project(standardized, pca_projected);
    
    // Step 3: LDA scoring
    lda_score(pca_projected, class_scores);
    
    // Step 4: ArgMax - find best class
    int predicted_class = argmax(class_scores);
    
    return predicted_class;
}


// ============================================
// MAIN FUNCTION
// ============================================
int main(void) {
    // Initialize model
    initialize_model_fixed();
    
    // Define memory location where sensor data is stored
    volatile int32_t* sensor_data = (volatile int32_t*)0x80000000;  // Example address
    
    // Load sensor data from memory location
    int32_t raw_input[NUM_RAW_FEATURES];
    for (int i = 0; i < NUM_RAW_FEATURES; i++) {
        raw_input[i] = sensor_data[i];
    }
    
    // Run inference
    int predicted_class = lda_predict(raw_input);
    
    // Output prediction
    volatile int32_t* output_register = (volatile int32_t*)0x80001000;  // Example address
    *output_register = predicted_class;

    return 0;
}

/*
int main(void) {
    // Initialize model
    initialize_model_fixed();
    
    //int32_t raw_input[NUM_RAW_FEATURES] = {
    //488734,   // 0.7379 AQ
    //2851333,  // 13.7804
    //4560833,  // 14.3927
    //1502189,  // 3.6468
    //1938450,  // 9.4687
    //2893375   // 8.7678
    // ... fill remaining features if NUM_RAW_FEATURES > 6
    //};

    // Run inference
    //int predicted_class = lda_predict(raw_input);
    
    int32_t raw_input[NUM_RAW_FEATURES] = {
    63714,   // 0.7379 LQ
    728170,  // 13.7804
    762780,  // 14.3927
    268324,  // 3.6468
    454944,  // 9.4687
    534315   // 8.7678
    // ... fill remaining features if NUM_RAW_FEATURES > 6
    };

    // Run inference
    int predicted_class = lda_predict(raw_input);
    printf("Predicted class: %d\n", predicted_class);


    return 0;
}
*/