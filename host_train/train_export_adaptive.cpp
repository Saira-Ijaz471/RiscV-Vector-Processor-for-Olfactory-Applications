#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <filesystem>
#include <cmath>
#include <iomanip>
#include <map>
#include <algorithm>
#include <random>

#include "alglib/ap.h"
#include "alglib/dataanalysis.h"
#include "alglib/linalg.h"
using namespace alglib;

extern "C" {
#include "svm.h"
}

namespace fs = std::filesystem;

// ============================================================
// CONFIGURATION
// ============================================================
struct TrainConfig {
    double test_split = 0.2;        // 20% for testing, 80% for training
    int random_seed = 42;           // For reproducible splits
    bool shuffle = true;            // Shuffle before splitting
    bool stratified = true;         // Keep class ratios in splits
    
    // Model parameters
    int pca_components = -1;        // -1 = auto
    double variance_threshold = 0.95;
    double svm_C = 1.0;
    double svm_gamma = -1.0;        // -1 = auto
};

// ============================================================
// EVALUATION METRICS
// ============================================================
struct EvaluationMetrics {
    int total_samples;
    int correct_predictions;
    double accuracy;
    std::vector<std::vector<int>> confusion_matrix;
    std::vector<double> per_class_accuracy;
    
    void compute(const std::vector<int>& true_labels, 
                const std::vector<int>& predictions,
                int num_classes) {
        total_samples = true_labels.size();
        correct_predictions = 0;
        
        // Initialize confusion matrix
        confusion_matrix.resize(num_classes, std::vector<int>(num_classes, 0));
        per_class_accuracy.resize(num_classes, 0.0);
        std::vector<int> class_counts(num_classes, 0);
        
        // Fill confusion matrix
        for (size_t i = 0; i < true_labels.size(); i++) {
            int true_label = true_labels[i];
            int pred_label = predictions[i];
            
            confusion_matrix[true_label][pred_label]++;
            class_counts[true_label]++;
            
            if (true_label == pred_label) {
                correct_predictions++;
            }
        }
        
        // Compute overall accuracy
        accuracy = (double)correct_predictions / total_samples;
        
        // Compute per-class accuracy
        for (int i = 0; i < num_classes; i++) {
            if (class_counts[i] > 0) {
                per_class_accuracy[i] = (double)confusion_matrix[i][i] / class_counts[i];
            }
        }
    }
    
    void print(const std::vector<std::string>& class_names) {
        std::cout << "\n╔════════════════════════════════════════════╗\n";
        std::cout << "║        MODEL EVALUATION RESULTS            ║\n";
        std::cout << "╚════════════════════════════════════════════╝\n\n";
        
        std::cout << "Total Samples: " << total_samples << "\n";
        std::cout << "Correct: " << correct_predictions << "\n";
        std::cout << "Overall Accuracy: " << std::fixed << std::setprecision(2) 
                  << (accuracy * 100) << "%\n\n";
        
        // Confusion Matrix
        std::cout << "Confusion Matrix:\n";
        std::cout << "           ";
        for (size_t i = 0; i < class_names.size(); i++) {
            std::cout << std::setw(8) << class_names[i].substr(0, 7);
        }
        std::cout << "\n";
        
        for (size_t i = 0; i < confusion_matrix.size(); i++) {
            std::cout << std::setw(10) << class_names[i].substr(0, 9) << " ";
            for (size_t j = 0; j < confusion_matrix[i].size(); j++) {
                std::cout << std::setw(8) << confusion_matrix[i][j];
            }
            std::cout << "\n";
        }
        
        // Per-class accuracy
        std::cout << "\nPer-Class Accuracy:\n";
        for (size_t i = 0; i < class_names.size(); i++) {
            std::cout << "  " << std::setw(15) << class_names[i] << ": "
                      << std::setw(6) << std::fixed << std::setprecision(2)
                      << (per_class_accuracy[i] * 100) << "%\n";
        }
        std::cout << "\n";
    }
};

// ============================================================
// DATA SPLITTING FUNCTIONS
// ============================================================
void stratified_train_test_split(
    const std::vector<std::vector<double>>& X,
    const std::vector<int>& y,
    std::vector<std::vector<double>>& X_train,
    std::vector<int>& y_train,
    std::vector<std::vector<double>>& X_test,
    std::vector<int>& y_test,
    double test_size,
    int random_seed)
{
    // Group indices by class
    int num_classes = *std::max_element(y.begin(), y.end()) + 1;
    std::vector<std::vector<size_t>> class_indices(num_classes);
    
    for (size_t i = 0; i < y.size(); i++) {
        class_indices[y[i]].push_back(i);
    }
    
    // Shuffle and split each class
    std::mt19937 rng(random_seed);
    
    for (int c = 0; c < num_classes; c++) {
        std::shuffle(class_indices[c].begin(), class_indices[c].end(), rng);
        
        size_t test_count = (size_t)(class_indices[c].size() * test_size);
        
        // Split this class
        for (size_t i = 0; i < class_indices[c].size(); i++) {
            size_t idx = class_indices[c][i];
            
            if (i < test_count) {
                X_test.push_back(X[idx]);
                y_test.push_back(y[idx]);
            } else {
                X_train.push_back(X[idx]);
                y_train.push_back(y[idx]);
            }
        }
    }
    
    std::cout << "Data Split Summary:\n";
    std::cout << "  Training samples: " << X_train.size() << " ("
              << (100.0 * X_train.size() / X.size()) << "%)\n";
    std::cout << "  Testing samples:  " << X_test.size() << " ("
              << (100.0 * X_test.size() / X.size()) << "%)\n";
    
    // Show per-class split
    std::vector<int> train_class_counts(num_classes, 0);
    std::vector<int> test_class_counts(num_classes, 0);
    
    for (int label : y_train) train_class_counts[label]++;
    for (int label : y_test) test_class_counts[label]++;
    
    std::cout << "\n  Per-Class Split:\n";
    for (int c = 0; c < num_classes; c++) {
        std::cout << "    Class " << c << ": "
                  << train_class_counts[c] << " train, "
                  << test_class_counts[c] << " test\n";
    }
    std::cout << "\n";
}

// ============================================================
// NaN HANDLING FUNCTION
// ============================================================
void clean_dataset(std::vector<std::vector<double>>& X) {
    if (X.empty()) return;
    
    int D = X[0].size();
    int N = X.size();
    
    std::cout << "Checking for NaN values...\n";
    
    // Compute column means (ignoring NaN)
    std::vector<double> col_means(D, 0.0);
    std::vector<int> col_counts(D, 0);
    
    for (int j = 0; j < D; j++) {
        for (int i = 0; i < N; i++) {
            if (!std::isnan(X[i][j]) && !std::isinf(X[i][j])) {
                col_means[j] += X[i][j];
                col_counts[j]++;
            }
        }
        if (col_counts[j] > 0) {
            col_means[j] /= col_counts[j];
        }
    }
    
    // Replace NaN with column mean
    int nan_count = 0;
    std::vector<int> nan_per_col(D, 0);
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            if (std::isnan(X[i][j]) || std::isinf(X[i][j])) {
                X[i][j] = col_means[j];
                nan_count++;
                nan_per_col[j]++;
            }
        }
    }
    
    if (nan_count > 0) {
        std::cout << "⚠ WARNING: Found and replaced " << nan_count 
                  << " NaN/Inf values with column means\n";
        std::cout << "\nNaN count per feature:\n";
        for (int j = 0; j < D; j++) {
            if (nan_per_col[j] > 0) {
                std::cout << "  Feature " << (j+1) << ": " << nan_per_col[j] 
                          << " NaN values (" << std::fixed << std::setprecision(1)
                          << (100.0 * nan_per_col[j] / N) << "%)\n";
            }
        }
        std::cout << "\n";
    } else {
        std::cout << "✓ No NaN values found\n\n";
    }
}

// ============================================================
// HELPER FUNCTIONS
// ============================================================
void write_1d(const std::string& f, const std::string& name, 
              const std::vector<double>& a) {
    std::ofstream o(f);
    o << "#pragma once\n";
    o << "static const double " << name << "[" << a.size() << "] = {";
    for(size_t i = 0; i < a.size(); i++) {
        if(i % 8 == 0) o << "\n    ";
        o << std::setprecision(12) << a[i];
        if(i + 1 < a.size()) o << ",";
    }
    o << "\n};\n";
    o << "static const int " << name << "_size = " << a.size() << ";\n";
}

void write_2d(const std::string& f, const std::string& name, 
              const std::vector<double>& a, int r, int c) {
    std::ofstream o(f);
    o << "#pragma once\n";
    o << "static const double " << name << "[" << r * c << "] = {";
    for(int i = 0; i < r * c; i++) {
        if(i % 8 == 0) o << "\n    ";
        o << std::setprecision(12) << a[i];
        if(i + 1 < r * c) o << ",";
    }
    o << "\n};\n";
    o << "static const int " << name << "_rows = " << r << ";\n";
    o << "static const int " << name << "_cols = " << c << ";\n";
}

std::map<std::string, int> discover_classes(const std::string& base) {
    std::map<std::string, int> class_map;
    std::vector<std::string> folders;
    
    for(auto& entry : fs::directory_iterator(base)) {
        if(entry.is_directory()) {
            folders.push_back(entry.path().filename().string());
        }
    }
    
    std::sort(folders.begin(), folders.end());
    
    for(size_t i = 0; i < folders.size(); i++) {
        class_map[folders[i]] = i;
    }
    
    return class_map;
}

void load_dataset(const std::string& base,
                  std::vector<std::vector<double>>& X,
                  std::vector<int>& y,
                  std::map<std::string, int>& class_map)
{
    class_map = discover_classes(base);
    
    int feature_dim = -1;
    int skipped_lines = 0;
    
    for(auto& [class_name, label] : class_map) {
        std::string path = base + "/" + class_name;
        
        for(auto& entry : fs::directory_iterator(path)) {
            if(!entry.is_regular_file()) continue;
            
            std::ifstream fin(entry.path());
            std::string line;
            int line_num = 0;
            
            while(std::getline(fin, line)) {
                line_num++;
                if(line.empty()) continue;
                
                std::stringstream ss(line);
                std::string token;
                std::vector<double> row;
                
                // Parse line, handling "nan", "NaN", "inf" strings
                while(ss >> token) {
                    double v;
                    try {
                        if (token == "nan" || token == "NaN" || token == "NAN") {
                            v = std::numeric_limits<double>::quiet_NaN();
                        } else if (token == "inf" || token == "Inf" || token == "INF") {
                            v = std::numeric_limits<double>::infinity();
                        } else {
                            v = std::stod(token);
                        }
                        row.push_back(v);
                    } catch (const std::exception& e) {
                        std::cerr << "Warning: Invalid value '" << token 
                                  << "' in " << entry.path().filename() 
                                  << " line " << line_num << ", treating as NaN\n";
                        row.push_back(std::numeric_limits<double>::quiet_NaN());
                    }
                }
                
                if(row.empty()) continue;
                
                if(feature_dim == -1) {
                    feature_dim = row.size();
                    std::cout << "Detected feature dimension: " << feature_dim << "\n";
                } else if((int)row.size() != feature_dim) {
                    std::cerr << "WARNING: Inconsistent dimensions in " 
                              << entry.path().filename() << " line " << line_num 
                              << " (expected " << feature_dim << ", got " << row.size() 
                              << "), skipping\n";
                    skipped_lines++;
                    continue;
                }
                
                X.push_back(row);
                y.push_back(label);
            }
        }
    }
    
    if (skipped_lines > 0) {
        std::cout << "⚠ Skipped " << skipped_lines << " lines due to dimension mismatch\n";
    }
}

// ============================================================
// PREDICTION FUNCTION FOR TESTING
// ============================================================
int predict_sample(const std::vector<double>& x_raw,
                  const std::vector<double>& mean,
                  const std::vector<double>& var,
                  const std::vector<double>& pca_components,
                  int K, int D,
                  svm_model* model)
{
    // Standardize
    std::vector<double> x_std(D);
    for (int j = 0; j < D; j++) {
        double std_dev = (var[j] > 0) ? std::sqrt(var[j]) : 1.0;
        x_std[j] = (x_raw[j] - mean[j]) / std_dev;
    }
    
    // Apply PCA
    std::vector<double> x_pca(K);
    for (int k = 0; k < K; k++) {
        double sum = 0.0;
        for (int j = 0; j < D; j++) {
            sum += x_std[j] * pca_components[k * D + j];
        }
        x_pca[k] = sum;
    }
    
    // Create SVM node
    std::vector<svm_node> nodes(K + 1);
    for (int k = 0; k < K; k++) {
        nodes[k].index = k + 1;
        nodes[k].value = x_pca[k];
    }
    nodes[K].index = -1;
    nodes[K].value = 0;
    
    // Predict
    return (int)svm_predict(model, nodes.data());
}

// ============================================================
// MAIN WITH TRAIN/TEST SPLIT
// ============================================================
int main(int argc, char** argv) {
    if(argc < 3) {
        std::cout << "Usage: " << argv[0] << " <dataset_dir> <output_prefix> [options]\n";
        std::cout << "\nOptions:\n";
        std::cout << "  --test-split T       : Test set ratio (default: 0.2)\n";
        std::cout << "  --pca-components N   : PCA components (default: auto)\n";
        std::cout << "  --svm-C C           : SVM C parameter (default: 1.0)\n";
        std::cout << "  --svm-gamma G       : SVM gamma (default: auto)\n";
        std::cout << "  --random-seed S     : Random seed (default: 42)\n";
        return 1;
    }

    std::string base = argv[1];
    std::string out = argv[2];
    
    TrainConfig config;
    
    // Parse arguments
    for(int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if(arg == "--test-split" && i+1 < argc) {
            config.test_split = std::stod(argv[++i]);
        } else if(arg == "--pca-components" && i+1 < argc) {
            config.pca_components = std::stoi(argv[++i]);
        } else if(arg == "--svm-C" && i+1 < argc) {
            config.svm_C = std::stod(argv[++i]);
        } else if(arg == "--svm-gamma" && i+1 < argc) {
            config.svm_gamma = std::stod(argv[++i]);
        } else if(arg == "--random-seed" && i+1 < argc) {
            config.random_seed = std::stoi(argv[++i]);
        }
    }

    // Load full dataset
    std::vector<std::vector<double>> X, X_train, X_test;
    std::vector<int> y, y_train, y_test;
    std::map<std::string, int> class_map;
    
    std::cout << "Loading dataset...\n";
    load_dataset(base, X, y, class_map);
    
    if (X.empty()) {
        std::cerr << "ERROR: No data loaded!\n";
        return 1;
    }
    
    // Clean NaN values
    clean_dataset(X);
    
    int N = X.size();
    int D = X[0].size();
    int num_classes = class_map.size();
    
    std::cout << "Dataset: " << N << " samples, " << D << " features, "
              << num_classes << " classes\n\n";

    // ============================================================
    // TRAIN/TEST SPLIT
    // ============================================================
    std::cout << "Splitting data (test_split=" << config.test_split << ")...\n";
    stratified_train_test_split(X, y, X_train, y_train, X_test, y_test,
                                config.test_split, config.random_seed);

    int N_train = X_train.size();
    int N_test = X_test.size();

    // ============================================================
    // TRAIN ON TRAINING SET ONLY
    // ============================================================
    
    // Convert training data to ALGLIB matrix
    real_2d_array A_train;
    A_train.setlength(N_train, D);
    for(int i = 0; i < N_train; i++)
        for(int j = 0; j < D; j++)
            A_train[i][j] = X_train[i][j];

    // Compute mean/var from TRAINING data only
    std::vector<double> mean(D, 0), var(D, 0);
    for(int j = 0; j < D; j++) {
        double s = 0;
        for(int i = 0; i < N_train; i++) s += A_train[i][j];
        mean[j] = s / N_train;
    }
    for(int j = 0; j < D; j++) {
        double s = 0;
        for(int i = 0; i < N_train; i++) {
            double d = A_train[i][j] - mean[j];
            s += d * d;
        }
        var[j] = s / (N_train - 1);
    }

    // Standardize training data
    real_2d_array Astd_train;
    Astd_train.setlength(N_train, D);
    for(int i = 0; i < N_train; i++)
        for(int j = 0; j < D; j++)
            Astd_train[i][j] = (A_train[i][j] - mean[j]) / 
                               (var[j] > 0 ? std::sqrt(var[j]) : 1.0);

    // PCA on training data
    real_1d_array s2;
    real_2d_array v;
    ae_int_t info;
    pcabuildbasis(Astd_train, N_train, D, info, s2, v);

    int K = (config.pca_components > 0) ? 
            std::min(config.pca_components, D) : 
            std::min(D, 8);
    
    std::vector<double> pca_flat;
    for(int i = 0; i < K; i++)
        for(int j = 0; j < D; j++)
            pca_flat.push_back(v[i][j]);

    // Project training data to PCA
    std::vector<double> Xp_train(N_train * K);
    for(int i = 0; i < N_train; i++) {
        for(int k = 0; k < K; k++) {
            double sum = 0;
            for(int j = 0; j < D; j++)
                sum += Astd_train[i][j] * v[k][j];
            Xp_train[i * K + k] = sum;
        }
    }

    // Train SVM on training data
    svm_problem prob;
    prob.l = N_train;
    prob.y = (double*) malloc(sizeof(double) * N_train);
    prob.x = (svm_node**) malloc(sizeof(svm_node*) * N_train);
    std::vector<std::vector<svm_node>> nodes(N_train);

    for(int i = 0; i < N_train; i++) {
        prob.y[i] = y_train[i];
        nodes[i].reserve(K + 1);
        for(int k = 0; k < K; k++) {
            svm_node nd;
            nd.index = k + 1;
            nd.value = Xp_train[i * K + k];
            nodes[i].push_back(nd);
        }
        svm_node end;
        end.index = -1;
        end.value = 0;
        nodes[i].push_back(end);
        prob.x[i] = nodes[i].data();
    }

    svm_parameter param;
    param.svm_type = C_SVC;
    param.kernel_type = RBF;
    param.C = config.svm_C;
    param.gamma = (config.svm_gamma > 0) ? config.svm_gamma : (1.0 / K);
    param.eps = 1e-3;
    param.cache_size = 200;
    param.shrinking = 1;
    param.probability = 0;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;
    param.degree = 3;
    param.coef0 = 0;
    param.nu = 0.5;
    param.p = 0.1;

    std::cout << "Training SVM on " << N_train << " samples...\n";
    std::cout << "SVM Parameters: C=" << param.C << ", gamma=" << param.gamma << "\n";
    svm_model* model = svm_train(&prob, &param);
    std::cout << "✓ Training complete\n\n";

    int nSV = model->l;
    std::cout << "Support Vectors: " << nSV << "\n";
    if (nSV > 1000) {
        std::cout << "⚠ WARNING: Very high number of support vectors!\n";
        std::cout << "  This indicates overfitting. Consider:\n";
        std::cout << "  - Increasing SVM C parameter (--svm-C 10.0)\n";
        std::cout << "  - Increasing gamma (--svm-gamma 1.0)\n";
        std::cout << "  - Reducing PCA components\n\n";
    }

    // ============================================================
    // EVALUATE ON TEST SET
    // ============================================================
    std::cout << "Evaluating on " << N_test << " test samples...\n";
    
    std::vector<int> predictions;
    for (int i = 0; i < N_test; i++) {
        int pred = predict_sample(X_test[i], mean, var, pca_flat, K, D, model);
        predictions.push_back(pred);
    }
    
    // Compute and display metrics
    EvaluationMetrics metrics;
    std::vector<std::string> class_names;
    for (auto& [name, idx] : class_map) {
        class_names.push_back(name);
    }
    metrics.compute(y_test, predictions, num_classes);
    metrics.print(class_names);

    // ============================================================
    // EXPORT MODEL (trained on training data)
    // ============================================================
    
    int nClass = model->nr_class;

    std::vector<double> sv_flat;
    for(int s = 0; s < nSV; s++) {
        std::vector<double> row(K, 0);
        svm_node* p = model->SV[s];
        while(p->index != -1) {
            row[p->index - 1] = p->value;
            p++;
        }
        for(double val : row) sv_flat.push_back(val);
    }

    std::vector<double> alpha_flat;
    for(int i = 0; i < nClass - 1; i++)
        for(int j = 0; j < nSV; j++)
            alpha_flat.push_back(model->sv_coef[i][j]);

    std::vector<double> rho_flat;
    for(int i = 0; i < nClass * (nClass - 1) / 2; i++)
        rho_flat.push_back(model->rho[i]);

    write_2d(out + "_pca_components.h", "pca_components", pca_flat, K, D);
    write_1d(out + "_scaler_mean.h", "scaler_mean", mean);
    write_1d(out + "_scaler_var.h", "scaler_var", var);
    write_2d(out + "_sv.h", "sv", sv_flat, nSV, K);
    write_2d(out + "_sv_alpha.h", "sv_alpha", alpha_flat, nClass - 1, nSV);
    write_1d(out + "_sv_intercept.h", "sv_intercept", rho_flat);

    // Write config
    std::ofstream cfg(out + "_config.h");
    cfg << "#pragma once\n\n";
    cfg << "#define NUM_CLASSES " << num_classes << "\n";
    cfg << "#define NUM_FEATURES " << D << "\n";
    cfg << "#define PCA_COMPONENTS " << K << "\n";
    cfg << "#define SVM_GAMMA " << param.gamma << "\n\n";
    cfg << "// Test Accuracy: " << std::fixed << std::setprecision(2) 
        << (metrics.accuracy * 100) << "%\n";
    cfg << "// Support Vectors: " << nSV << "\n\n";
    cfg << "static const char* CLASS_NAMES[" << num_classes << "] = {\n";
    for(size_t i = 0; i < class_names.size(); i++) {
        cfg << "    \"" << class_names[i] << "\"";
        if(i + 1 < class_names.size()) cfg << ",";
        cfg << "\n";
    }
    cfg << "};\n";

    // Estimate model size
    size_t model_size = (pca_flat.size() + mean.size() + var.size() + 
                        sv_flat.size() + alpha_flat.size() + rho_flat.size()) * sizeof(double);
    
    std::cout << "\n✓ Model exported with " << std::fixed << std::setprecision(2)
              << (metrics.accuracy * 100) << "% test accuracy\n";
    std::cout << "  Model size: " << (model_size / 1024) << " KB\n";
    std::cout << "  Support vectors: " << nSV << "\n";

    svm_free_and_destroy_model(&model);
    free(prob.y);
    free(prob.x);

    return 0;
}