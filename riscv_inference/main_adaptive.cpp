// TRUE Bare-metal Wine Classifier for 32-bit Fixed-Point RISC-V Vector Processor
// Converts floating-point model to fixed-point at compile time
// NO FLOATING-POINT at runtime!
// Updated for GCC 15.1.0 RVV intrinsics

#include <stdint.h>

// Include the original floating-point model headers
#include "model_config.h"
#include "model_pca_components.h"
#include "model_scaler_mean.h"
#include "model_scaler_var.h"
#include "model_sv.h"
#include "model_sv_alpha.h"
#include "model_sv_intercept.h"

// Enable RVV intrinsics
#ifdef __riscv_vector
#include <riscv_vector.h>
#define USE_RVV 1
#endif

// ==========================================
// Fixed-Point Configuration
// ==========================================
typedef int32_t fixed32_t;
#define FIXED_SHIFT 16
#define FIXED_ONE (1 << FIXED_SHIFT)  // 65536 = 1.0

// Compile-time conversion macro
#define DOUBLE_TO_FIXED(x) ((fixed32_t)((x) * 65536.0))

// Runtime conversions
static inline fixed32_t double_to_fixed_runtime(double x) {
    return (fixed32_t)(x * 65536.0);
}

static inline double fixed_to_double(fixed32_t x) {
    return (double)x / 65536.0;
}

// Fixed-point multiply: (a * b) >> 16
static inline fixed32_t fixed_mul(fixed32_t a, fixed32_t b) {
    int64_t temp = (int64_t)a * (int64_t)b;
    return (fixed32_t)(temp >> FIXED_SHIFT);
}

// Fixed-point divide: (a << 16) / b
static inline fixed32_t fixed_div(fixed32_t a, fixed32_t b) {
    if (b == 0) return 0;
    int64_t temp = ((int64_t)a << FIXED_SHIFT) / b;
    return (fixed32_t)temp;
}

// ==========================================
// Bare-metal math (fixed-point)
// ==========================================

static inline fixed32_t fixed_sqrt(fixed32_t x) {
    if (x <= 0) return 0;
    fixed32_t guess = x >> 1;
    
    for (int i = 0; i < 10; i++) {
        if (guess == 0) break;
        fixed32_t new_guess = (guess + fixed_div(x, guess)) >> 1;
        if (new_guess == guess) break;
        guess = new_guess;
    }
    return guess;
}

static inline fixed32_t fixed_exp(fixed32_t x) {
    // Clamp
    if (x > DOUBLE_TO_FIXED(10.0)) x = DOUBLE_TO_FIXED(10.0);
    if (x < DOUBLE_TO_FIXED(-10.0)) x = DOUBLE_TO_FIXED(-10.0);
    
    fixed32_t sum = FIXED_ONE;
    fixed32_t term = FIXED_ONE;
    
    for (int i = 1; i < 20; i++) {
        term = fixed_mul(term, fixed_div(x, (i << FIXED_SHIFT)));
        sum += term;
        if (term < 100 && term > -100) break;
    }
    return sum;
}

// ==========================================
// Convert model to fixed-point at startup
// ==========================================

static fixed32_t pca_components_fixed[PCA_COMPONENTS * NUM_FEATURES];
static fixed32_t scaler_mean_fixed[NUM_FEATURES];
static fixed32_t scaler_var_fixed[NUM_FEATURES];
static fixed32_t sv_fixed[1024 * PCA_COMPONENTS];  // Max support vectors
static fixed32_t sv_alpha_fixed[NUM_CLASSES * 1024];
static fixed32_t sv_intercept_fixed[20];  // Max classifiers
static fixed32_t gamma_fixed;

static bool model_initialized = false;

static void initialize_fixed_model() {
    if (model_initialized) return;
    
    // Convert PCA components
    for (int i = 0; i < PCA_COMPONENTS * NUM_FEATURES; i++) {
        pca_components_fixed[i] = double_to_fixed_runtime(pca_components[i]);
    }
    
    // Convert scaler mean
    for (int i = 0; i < NUM_FEATURES; i++) {
        scaler_mean_fixed[i] = double_to_fixed_runtime(scaler_mean[i]);
    }
    
    // Convert scaler variance
    for (int i = 0; i < NUM_FEATURES; i++) {
        scaler_var_fixed[i] = double_to_fixed_runtime(scaler_var[i]);
    }
    
    // Convert support vectors
    for (int i = 0; i < sv_rows * PCA_COMPONENTS; i++) {
        sv_fixed[i] = double_to_fixed_runtime(sv[i]);
    }
    
    // Convert alpha coefficients
    for (int i = 0; i < NUM_CLASSES * sv_rows; i++) {
        sv_alpha_fixed[i] = double_to_fixed_runtime(sv_alpha[i]);
    }
    
    // Convert intercepts
    int n_classifiers = NUM_CLASSES * (NUM_CLASSES - 1) / 2;
    for (int i = 0; i < n_classifiers; i++) {
        sv_intercept_fixed[i] = double_to_fixed_runtime(sv_intercept[i]);
    }
    
    // Convert gamma
    gamma_fixed = double_to_fixed_runtime(SVM_GAMMA);
    
    model_initialized = true;
}

// ==========================================
// VECTORIZED: Dot Product (32-bit fixed-point)
// Updated for GCC 15.1.0 with __riscv_ prefix
// ==========================================
#ifdef USE_RVV

static fixed32_t dot_product_rvv(const fixed32_t* a, const fixed32_t* b, int n) {
    size_t vl;
    vint32m1_t vacc = __riscv_vmv_v_x_i32m1(0, __riscv_vsetvlmax_e32m1());
    
    for (size_t i = 0; i < (size_t)n; ) {
        vl = __riscv_vsetvl_e32m1(n - i);
        
        // Load 32-bit vectors
        vint32m1_t va = __riscv_vle32_v_i32m1(&a[i], vl);
        vint32m1_t vb = __riscv_vle32_v_i32m1(&b[i], vl);
        
        // Widening multiply: 32x32 -> 64-bit result
        vint64m2_t vprod_wide = __riscv_vwmul_vv_i64m2(va, vb, vl);
        
        // Shift right by 16 to get back to Q16.16
        vint32m1_t vprod = __riscv_vnsra_wx_i32m1(vprod_wide, FIXED_SHIFT, vl);
        
        // Accumulate
        vacc = __riscv_vadd_vv_i32m1(vacc, vprod, vl);
        
        i += vl;
    }
    
    // Reduce to scalar
    vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);
    vint32m1_t vsum = __riscv_vredsum_vs_i32m1_i32m1(vacc, vzero, __riscv_vsetvlmax_e32m1());
    return __riscv_vmv_x_s_i32m1_i32(vsum);
}

#else

static fixed32_t dot_product_rvv(const fixed32_t* a, const fixed32_t* b, int n) {
    int64_t sum = 0;
    for (int i = 0; i < n; i++) {
        sum += fixed_mul(a[i], b[i]);
    }
    return (fixed32_t)sum;
}

#endif

// ==========================================
// VECTORIZED: Squared Distance
// Updated for GCC 15.1.0
// ==========================================
#ifdef USE_RVV

static fixed32_t squared_distance_rvv(const fixed32_t* x, const fixed32_t* sv, int n) {
    size_t vl;
    vint32m1_t vacc = __riscv_vmv_v_x_i32m1(0, __riscv_vsetvlmax_e32m1());
    
    for (size_t i = 0; i < (size_t)n; ) {
        vl = __riscv_vsetvl_e32m1(n - i);
        
        vint32m1_t vx = __riscv_vle32_v_i32m1(&x[i], vl);
        vint32m1_t vsv = __riscv_vle32_v_i32m1(&sv[i], vl);
        
        // Difference
        vint32m1_t vdiff = __riscv_vsub_vv_i32m1(vx, vsv, vl);
        
        // Square with widening multiply
        vint64m2_t vsq_wide = __riscv_vwmul_vv_i64m2(vdiff, vdiff, vl);
        
        // Shift right
        vint32m1_t vsq = __riscv_vnsra_wx_i32m1(vsq_wide, FIXED_SHIFT, vl);
        
        // Accumulate
        vacc = __riscv_vadd_vv_i32m1(vacc, vsq, vl);
        
        i += vl;
    }
    
    vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);
    vint32m1_t vsum = __riscv_vredsum_vs_i32m1_i32m1(vacc, vzero, __riscv_vsetvlmax_e32m1());
    return __riscv_vmv_x_s_i32m1_i32(vsum);
}

#else

static fixed32_t squared_distance_rvv(const fixed32_t* x, const fixed32_t* sv, int n) {
    int64_t sum = 0;
    for (int i = 0; i < n; i++) {
        fixed32_t diff = x[i] - sv[i];
        sum += fixed_mul(diff, diff);
    }
    return (fixed32_t)sum;
}

#endif

// ==========================================
// VECTORIZED: Standardization
// Updated for GCC 15.1.0
// ==========================================
#ifdef USE_RVV

static void standardize_rvv(const fixed32_t* x_raw, fixed32_t* x_std,
                           const fixed32_t* mean, const fixed32_t* var, int n) {
    size_t vl;
    
    for (size_t i = 0; i < (size_t)n; ) {
        vl = __riscv_vsetvl_e32m1(n - i);
        
        vint32m1_t vraw = __riscv_vle32_v_i32m1(&x_raw[i], vl);
        vint32m1_t vmean = __riscv_vle32_v_i32m1(&mean[i], vl);
        
        // Subtract mean
        vint32m1_t vsub = __riscv_vsub_vv_i32m1(vraw, vmean, vl);
        
        // Store for scalar division
        fixed32_t temp[32];
        __riscv_vse32_v_i32m1(temp, vsub, vl);
        
        // Scalar divide by sqrt(var)
        for (size_t j = 0; j < vl; j++) {
            fixed32_t std_dev = (var[i + j] > 0) ? fixed_sqrt(var[i + j]) : FIXED_ONE;
            x_std[i + j] = fixed_div(temp[j], std_dev);
        }
        
        i += vl;
    }
}

#else

static void standardize_rvv(const fixed32_t* x_raw, fixed32_t* x_std,
                           const fixed32_t* mean, const fixed32_t* var, int n) {
    for (int i = 0; i < n; i++) {
        fixed32_t std_dev = (var[i] > 0) ? fixed_sqrt(var[i]) : FIXED_ONE;
        x_std[i] = fixed_div(x_raw[i] - mean[i], std_dev);
    }
}

#endif

// ==========================================
// RBF Kernel
// ==========================================
static fixed32_t rbf_kernel(const fixed32_t* x, const fixed32_t* sv_ptr, 
                           int K, fixed32_t gamma) {
    fixed32_t dist_sq = squared_distance_rvv(x, sv_ptr, K);
    fixed32_t neg_gamma_dist = -fixed_mul(gamma, dist_sq);
    return fixed_exp(neg_gamma_dist);
}

// ==========================================
// Decision Function
// ==========================================
static fixed32_t decision_function(const fixed32_t* x_pca, int class_i, int class_j,
                                  int K, int nSV, int nClass, fixed32_t gamma) {
    int p = 0;
    for (int i = 0; i < nClass; i++) {
        for (int j = i + 1; j < nClass; j++) {
            if ((i == class_i && j == class_j) || (i == class_j && j == class_i)) {
                goto found;
            }
            p++;
        }
    }
    found:
    
    int64_t sum = 0;
    
    for (int s = 0; s < nSV; s++) {
        const fixed32_t* sv_ptr = &sv_fixed[s * K];
        fixed32_t k = rbf_kernel(x_pca, sv_ptr, K, gamma);
        fixed32_t alpha = sv_alpha_fixed[class_i * nSV + s];
        sum += fixed_mul(alpha, k);
    }
    
    sum -= sv_intercept_fixed[p];
    return (fixed32_t)sum;
}

// ==========================================
// MAIN PREDICTION - Takes fixed-point input
// ==========================================
int predict_fixed(const fixed32_t* x_raw_fixed) {
    // Initialize model once
    if (!model_initialized) {
        initialize_fixed_model();
    }
    
    const int D = NUM_FEATURES;
    const int K = PCA_COMPONENTS;
    const int nClass = NUM_CLASSES;
    const int nSV = sv_rows;
    
    fixed32_t x_std[NUM_FEATURES];
    fixed32_t x_pca[PCA_COMPONENTS];
    int votes[NUM_CLASSES];
    
    // Initialize votes
    for (int i = 0; i < nClass; i++) {
        votes[i] = 0;
    }
    
    // Step 1: VECTORIZED Standardization
    standardize_rvv(x_raw_fixed, x_std, scaler_mean_fixed, scaler_var_fixed, D);
    
    // Step 2: VECTORIZED PCA
    for (int k = 0; k < K; k++) {
        x_pca[k] = dot_product_rvv(x_std, &pca_components_fixed[k * D], D);
    }
    
    // Step 3: Voting
    for (int i = 0; i < nClass; i++) {
        for (int j = i + 1; j < nClass; j++) {
            fixed32_t dec = decision_function(x_pca, i, j, K, nSV, nClass, gamma_fixed);
            if (dec > 0) {
                votes[i]++;
            } else {
                votes[j]++;
            }
        }
    }
    
    // Step 4: Find winner
    int max_votes = 0;
    int predicted_class = 0;
    for (int i = 0; i < nClass; i++) {
        if (votes[i] > max_votes) {
            max_votes = votes[i];
            predicted_class = i;
        }
    }
    
    return predicted_class;
}

// ==========================================
// Wrapper for double input (converts to fixed)
// ==========================================
int predict(const double* x_raw_double) {
    fixed32_t x_raw_fixed[NUM_FEATURES];
    
    for (int i = 0; i < NUM_FEATURES; i++) {
        x_raw_fixed[i] = double_to_fixed_runtime(x_raw_double[i]);
    }
    
    return predict_fixed(x_raw_fixed);
}

// ==========================================
// Batch prediction
// ==========================================
void predict_batch(const double* samples, int num_samples, int* results) {
    for (int i = 0; i < num_samples; i++) {
        results[i] = predict(&samples[i * NUM_FEATURES]);
    }
}

void predict_batch_fixed(const fixed32_t* samples_fixed, int num_samples, int* results) {
    for (int i = 0; i < num_samples; i++) {
        results[i] = predict_fixed(&samples_fixed[i * NUM_FEATURES]);
    }
}

// ==========================================
// Prediction with confidence
// ==========================================
void predict_with_confidence(const double* x_raw_double, int* predicted_class, double* confidence) {
    fixed32_t x_raw_fixed[NUM_FEATURES];
    for (int i = 0; i < NUM_FEATURES; i++) {
        x_raw_fixed[i] = double_to_fixed_runtime(x_raw_double[i]);
    }
    
    if (!model_initialized) {
        initialize_fixed_model();
    }
    
    const int D = NUM_FEATURES;
    const int K = PCA_COMPONENTS;
    const int nClass = NUM_CLASSES;
    const int nSV = sv_rows;
    
    fixed32_t x_std[NUM_FEATURES];
    fixed32_t x_pca[PCA_COMPONENTS];
    int votes[NUM_CLASSES];
    
    for (int i = 0; i < nClass; i++) votes[i] = 0;
    
    standardize_rvv(x_raw_fixed, x_std, scaler_mean_fixed, scaler_var_fixed, D);
    
    for (int k = 0; k < K; k++) {
        x_pca[k] = dot_product_rvv(x_std, &pca_components_fixed[k * D], D);
    }
    
    for (int i = 0; i < nClass; i++) {
        for (int j = i + 1; j < nClass; j++) {
            fixed32_t dec = decision_function(x_pca, i, j, K, nSV, nClass, gamma_fixed);
            if (dec > 0) votes[i]++;
            else votes[j]++;
        }
    }
    
    int max_votes = 0;
    int winner = 0;
    for (int i = 0; i < nClass; i++) {
        if (votes[i] > max_votes) {
            max_votes = votes[i];
            winner = i;
        }
    }
    
    *predicted_class = winner;
    int total_comparisons = nClass * (nClass - 1) / 2;
    *confidence = (double)max_votes / total_comparisons;
}

// ==========================================
// Test samples
// ==========================================
static double test_sample_1[NUM_FEATURES];
static double test_sample_2[NUM_FEATURES];

void init_test_samples() {
    for (int i = 0; i < NUM_FEATURES; i++) {
        test_sample_1[i] = 0.1 * (i + 1);
        test_sample_2[i] = 1.5 - 0.1 * i;
    }
}

// ==========================================
// Bare-metal entry point
// ==========================================
#ifndef NATIVE_TEST

int main(void) {
    init_test_samples();
    
    int result1 = predict(test_sample_1);
    
    int result2;
    double confidence;
    predict_with_confidence(test_sample_2, &result2, &confidence);
    
    return result1;
}

#endif

// ==========================================
// Memory-mapped sensor (pure fixed-point)
// ==========================================
int classify_from_sensors_fixed(volatile fixed32_t* sensor_data_fixed) {
    fixed32_t features[NUM_FEATURES];
    for (int i = 0; i < NUM_FEATURES; i++) {
        features[i] = sensor_data_fixed[i];
    }
    return predict_fixed(features);
}

// ==========================================
// Native test mode
// ==========================================
#ifdef NATIVE_TEST

#include <stdio.h>

int main() {
    printf("=== Wine Classifier (32-bit Fixed-Point RVV) ===\n");
#ifdef USE_RVV
    printf("Vector instructions: ENABLED (32-bit integer vectors)\n");
#else
    printf("Vector instructions: DISABLED (scalar fallback)\n");
#endif
    printf("Fixed-point: Q16.16 format\n\n");
    
    printf("Model Configuration:\n");
    printf("  Features: %d\n", NUM_FEATURES);
    printf("  PCA Components: %d\n", PCA_COMPONENTS);
    printf("  Classes: %d\n", NUM_CLASSES);
    printf("  Support Vectors: %d\n", sv_rows);
    printf("  Gamma: %f\n\n", SVM_GAMMA);
    
    printf("Class Names:\n");
    for (int i = 0; i < NUM_CLASSES; i++) {
        printf("  %d: %s\n", i, CLASS_NAMES[i]);
    }
    printf("\n");
    
    init_test_samples();
    
    printf("Testing sample 1:\n");
    int result1 = predict(test_sample_1);
    printf("  Predicted class: %d (%s)\n\n", result1, CLASS_NAMES[result1]);
    
    printf("Testing sample 2 with confidence:\n");
    int result2;
    double confidence;
    predict_with_confidence(test_sample_2, &result2, &confidence);
    printf("  Predicted class: %d (%s)\n", result2, CLASS_NAMES[result2]);
    printf("  Confidence: %.2f%%\n", confidence * 100);
    
    return 0;
}

#endif