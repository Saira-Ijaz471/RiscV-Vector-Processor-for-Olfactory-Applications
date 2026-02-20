#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <random>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <stdexcept>
//SHARK LIBRARY
#include <shark/Algorithms/Trainers/LDA.h>
#include <shark/Data/Dataset.h>
//ALGLIB FOR PCA
#include "ap.h"
#include "dataanalysis.h"
using namespace alglib;

namespace fs = std::filesystem;
using namespace shark;

// =====================
// UTILITY CHECKS
// =====================
bool isValidNumber(double x) {                //checks is data has any NaN, +∞ or -∞
    return std::isfinite(x);
}

// =====================
// PARESE EACH LINE 
// =====================
bool parseSampleLine(
    const std::string& line, 
    std::vector<double>& sample,
    size_t expected_features
)  {
    sample.clear();                              
    std::stringstream ss(line);                  //read #s separated by space
    std::string token;                           

    while (ss >> token) {                        //Extracts each space separated token
        try {
            double val = std::stod(token);       //string to floating point 
            if (!isValidNumber(val))
                return false;
            sample.push_back(val);
        } catch (...) {
            return false;                        // non-numeric string
        }
    }

    if (sample.empty())                         //reject empty lines
        return false;

    if (expected_features != 0 && sample.size() != expected_features)     //enforece feature consistency
        return false;

    return true;
}

// =====================
// LOAD ONE CLASS FILE
// =====================
size_t loadClassData(
    const fs::path& file_path,
    int label,
    size_t& feature_dim,
    std::vector<std::vector<double>>& X,  
    std::vector<int>& y
) {
    std::ifstream file(file_path);
    if (!file.is_open()) {                                              //ensures file is readable
        std::cerr << "❌ Failed to open: " << file_path << "\n";
        return 0;
    }

    size_t valid_samples = 0;
    size_t discarded_samples = 0;
    std::string line;

    while (std::getline(file, line)) {              //read line by line
        std::vector<double> sample;
        if (!parseSampleLine(line, sample, feature_dim)) {
            discarded_samples++;
            continue;
        }

        if (feature_dim == 0)                        //get feature dimension
            feature_dim = sample.size();

        X.push_back(sample);
        y.push_back(label);
        valid_samples++;
    }

    std::cout << "   ✔ Loaded " << valid_samples
              << " samples (" << discarded_samples << " discarded)\n";

    return valid_samples;
}

// =====================
// DATASET STATISTICS
// =====================
void computeDatasetStats(
    const std::vector<std::vector<double>>& X
) {
    if (X.empty()) return;

    size_t N = X.size();                            //samples
    size_t D = X[0].size();                         //features

    std::vector<double> mean(D, 0.0);
    std::vector<double> minv(D, std::numeric_limits<double>::max());
    std::vector<double> maxv(D, std::numeric_limits<double>::lowest());

    for (const auto& sample : X) { 
        for (size_t d = 0; d < D; d++) {
            mean[d] += sample[d];
            minv[d] = std::min(minv[d], sample[d]);
            maxv[d] = std::max(maxv[d], sample[d]);
        }
    }

    for (double& m : mean)
        m /= static_cast<double>(N);

    std::cout << "\n📊 Dataset statistics:\n";
    std::cout << "Samples: " << N << "\n";
    std::cout << "Features: " << D << "\n";

    for (size_t d = 0; d < D; d++) {
        std::cout << "  Feature " << d
                  << " | mean=" << mean[d]
                  << " min=" << minv[d]
                  << " max=" << maxv[d] << "\n";
    }
}

// =====================
// LOAD ENTIRE DATASET
// =====================
void loadDataset(
    const std::string& dataset_root,
    std::vector<std::vector<double>>& X,
    std::vector<int>& y,
    std::vector<std::string>& class_names
) {
    X.clear();        //features
    y.clear();        //labels
    class_names.clear();

    size_t feature_dim = 0;

    std::vector<fs::path> class_dirs;
    for (const auto& entry : fs::directory_iterator(dataset_root)) {        //collect directories
        if (entry.is_directory())
            class_dirs.push_back(entry.path());
    }

    std::sort(class_dirs.begin(), class_dirs.end());                        //sort for labels

    int label = 0;
    for (const auto& class_dir : class_dirs) {                              //for each class
        std::string class_name = class_dir.filename().string();             
        size_t total_loaded = 0;

        for (const auto& entry : fs::directory_iterator(class_dir)) {       //read all files 
            if (!entry.is_regular_file())
                continue;

            fs::path data_file = entry.path();
            std::cout << "   Processing file: " << data_file << "\n";

            size_t loaded = loadClassData(data_file, label, feature_dim, X, y);
            total_loaded += loaded;
        }

        if (total_loaded > 0) {                                           //read class if valid
            class_names.push_back(class_name);
            std::cout << "   ✔ Total valid samples for class " << class_name
                    << ": " << total_loaded << "\n";
            label++;
        } else {
            std::cout << "   ⚠ No valid samples found, class skipped\n";
        }
    }


    if (X.empty()) {
        std::cerr << "\n❌ Dataset is empty after loading!\n";
        std::exit(EXIT_FAILURE);
    }

    std::cout << "\n✅ Dataset loaded successfully\n";
}

// =====================
// TRAIN/TEST SPLITTING
// =====================
void stratified_train_test_split(
    const std::vector<std::vector<double>>& X,
    const std::vector<int>& y,
    std::vector<std::vector<double>>& X_train,
    std::vector<int>& y_train,
    std::vector<std::vector<double>>& X_test,
    std::vector<int>& y_test,
    double test_ratio,
    int random_seed
) {
    if (X.size() != y.size()) {
        throw std::runtime_error("X and y size mismatch");
    }

    // Group sample indices by class
    std::unordered_map<int, std::vector<size_t>> class_indices;
    for (size_t i = 0; i < y.size(); i++) {
        class_indices[y[i]].push_back(i);
    }

    std::mt19937 rng(random_seed);

    size_t total_train = 0;
    size_t total_test  = 0;

    std::cout << "\n📊 Stratified train/test split\n";
    std::cout << "Requested test split: " << test_ratio * 100.0 << "%\n\n";

    // Split each class independently
    for (const auto& kv : class_indices) {
        int class_label = kv.first;
        const auto& indices = kv.second;

        if (indices.size() < 2) {
            throw std::runtime_error(
                "Class " + std::to_string(class_label) +
                " has fewer than 2 samples — cannot split"
            );
        }

        // Shuffle each class
        std::vector<size_t> shuffled = indices;
        std::shuffle(shuffled.begin(), shuffled.end(), rng);

        // Compute split size
        size_t n_total = shuffled.size();
        size_t n_test  = static_cast<size_t>(n_total * test_ratio);

        // Enforce at least one sample in each split
        if (n_test == 0) n_test = 1;
        if (n_test >= n_total) n_test = n_total - 1;

        size_t n_train = n_total - n_test;

        // Copy samples
        for (size_t i = 0; i < n_train; i++) {
            X_train.push_back(X[shuffled[i]]);
            y_train.push_back(class_label);
        }

        for (size_t i = n_train; i < n_total; i++) {
            X_test.push_back(X[shuffled[i]]);
            y_test.push_back(class_label);
        }

        total_train += n_train;
        total_test  += n_test;

        // Per-class logging
        std::cout << "Class " << class_label
                  << " | total=" << n_total
                  << " train=" << n_train
                  << " test=" << n_test << "\n";
    }

    // Global logging
    std::cout << "\n📈 Final split summary\n";
    std::cout << "Train samples: " << total_train << "\n";
    std::cout << "Test  samples: " << total_test  << "\n";
    std::cout << "Train %: "
              << std::fixed << std::setprecision(2)
              << (100.0 * total_train / (total_train + total_test))
              << "%\n";
    std::cout << "Test  %: "
              << (100.0 * total_test / (total_train + total_test))
              << "%\n";
}


// =====================
// STANDARDIZATION
// =====================
void standardizeData(
    std::vector<std::vector<double>>& X_train,
    std::vector<std::vector<double>>& X_test,
    std::vector<double>& mean,
    std::vector<double>& stddev
) {
    if (X_train.empty()) return;

    size_t N = X_train.size();
    size_t D = X_train[0].size();

    mean.assign(D, 0.0);
    stddev.assign(D, 0.0);

    // 1. Compute mean for each feature
    for (size_t d = 0; d < D; d++) {
        for (size_t i = 0; i < N; i++)
            mean[d] += X_train[i][d];
        mean[d] /= static_cast<double>(N);
    }

    // 2. Compute std deviation for each feature
    for (size_t d = 0; d < D; d++) {
        for (size_t i = 0; i < N; i++) {
            double diff = X_train[i][d] - mean[d];
            stddev[d] += diff * diff;
        }
        stddev[d] = std::sqrt(stddev[d] / static_cast<double>(N));
        if (stddev[d] == 0.0) stddev[d] = 1.0; // avoid division by zero
    }

    // 3. Standardize training data
    for (size_t i = 0; i < N; i++)
        for (size_t d = 0; d < D; d++)
            X_train[i][d] = (X_train[i][d] - mean[d]) / stddev[d];

    // 4. Standardize test data using training mean/std
    for (auto& sample : X_test)
        for (size_t d = 0; d < D; d++)
            sample[d] = (sample[d] - mean[d]) / stddev[d];

    std::cout << "\n✅ Data standardized successfully\n";
}

// =====================
// LOGGING FUNCTION (IMPROVED)
// =====================
void logData(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& mean,
    const std::vector<double>& stddev,
    const std::string& name,
    size_t max_samples = 5,
    bool isPCA = false  // new flag
) {
    std::cout << "\n📄 Logging " << name << " (showing up to " << max_samples << " samples)\n";

    if (!isPCA) {
        // Print mean and stddev only for regular standardized features
        std::cout << "Feature means: ";
        for (double m : mean) std::cout << std::fixed << std::setprecision(4) << m << " ";
        std::cout << "\n";

        std::cout << "Feature stddevs: ";
        for (double s : stddev) std::cout << std::fixed << std::setprecision(4) << s << " ";
        std::cout << "\n";
    } else {
        std::cout << "(PCA data, mean/stddev not shown)\n";
    }

    // Print sample values
    size_t N = std::min(X.size(), max_samples);
    for (size_t i = 0; i < N; i++) {
        std::cout << "Sample " << i << ": ";
        for (double val : X[i])
            std::cout << std::fixed << std::setprecision(4) << val << " ";
        std::cout << "\n";
    }
}


// =====================
// PCA COMPUTATION
// =====================
void computePCA(
    const std::vector<std::vector<double>>& X_train,
    const std::vector<std::vector<double>>& X_test,
    std::vector<std::vector<double>>& X_train_pca,
    std::vector<std::vector<double>>& X_test_pca,
    int num_components,
    real_2d_array& V_out  // ADD THIS
){
    size_t N_train = X_train.size();
    size_t D = X_train[0].size();
    size_t N_test = X_test.size();

    // Convert training data to ALGLIB matrix
    real_2d_array A;
    A.setlength(N_train, D);
    for (size_t i = 0; i < N_train; i++)
        for (size_t j = 0; j < D; j++)
            A[i][j] = X_train[i][j];

    // Prepare output arrays
    real_1d_array s2;  // variances(eigen values)
    real_2d_array V;   // PCA basis vectors(principle components/eigen vectors)

    // Build PCA basis
    pcabuildbasis(A, N_train, D, s2, V);

    std::cout << "\n📄 PCA Components (top " << num_components << "):\n";
    for (int j = 0; j < num_components; j++) {       // for each component
        std::cout << "PC" << j << ": ";
        for (size_t d = 0; d < D; d++) {
            std::cout << std::fixed << std::setprecision(4) << V[d][j] << " ";
        }
        std::cout << "\n";
    }

    // Limit components
    num_components = std::min<size_t>(num_components, D);

    // ---- Project TRAIN data ----
    X_train_pca.resize(N_train, std::vector<double>(num_components, 0.0));
    for (size_t i = 0; i < N_train; i++) {
        for (int j = 0; j < num_components; j++) {
            double val = 0.0;
            for (size_t d = 0; d < D; d++)
                val += X_train[i][d] * V[d][j];  // column j of V
            X_train_pca[i][j] = val;
        }
    }

    // ---- Project TEST data ----
    X_test_pca.resize(N_test, std::vector<double>(num_components, 0.0));
    for (size_t i = 0; i < N_test; i++) {
        for (int j = 0; j < num_components; j++) {
            double val = 0.0;
            for (size_t d = 0; d < D; d++)
                val += X_test[i][d] * V[d][j];
            X_test_pca[i][j] = val;
        }
    }

    V_out = V; 
    
    std::cout << "\n✅ PCA computed and data projected onto " << num_components << " components\n";
}

// ===============================
// CONVERT DATASET TO SHART FORMAT
// ===============================
ClassificationDataset makeSharkDataset(
    const std::vector<std::vector<double>>& X,
    const std::vector<int>& y
) {
    std::vector<RealVector> inputs;
    std::vector<unsigned int> labels;

    inputs.reserve(X.size());
    labels.reserve(y.size());

    for (size_t i = 0; i < X.size(); i++) {
        RealVector v(X[i].size());
        for (size_t j = 0; j < X[i].size(); j++)
            v(j) = X[i][j];

        inputs.push_back(v);
        labels.push_back(static_cast<unsigned int>(y[i]));
    }

    return createLabeledDataFromRange(inputs, labels);
}

// =====================
// LDA TRAINING 
// =====================
void trainLDA(
    const std::vector<std::vector<double>>& X_train_pca,
    const std::vector<int>& y_train,
    LinearClassifier<RealVector>& lda_model
) {
    auto dataset = makeSharkDataset(X_train_pca, y_train);

    LDA lda_trainer;
    lda_trainer.train(lda_model, dataset);

    std::cout << "✅ LDA training completed\n";
}

void logLDAModel(const LinearClassifier<RealVector>& lda_model) {
    const RealMatrix& W = lda_model.decisionFunction().matrix();
    const RealVector& b = lda_model.decisionFunction().offset();

    std::cout << "\n🔎 LDA Model Info:\n";
    std::cout << "Projection matrix: "
              << W.size1() << " x " << W.size2() << "\n";
    std::cout << "Bias vector size: " << b.size() << "\n";
}

// =====================
// LDA Prediction Helpers
// =====================
int predictLDA(
    const shark::LinearClassifier<shark::RealVector>& model,
    const std::vector<double>& x
) {
    shark::RealVector v(x.size());
    for (size_t i = 0; i < x.size(); i++)
        v(i) = x[i];

    shark::RealVector scores = model.decisionFunction()(v);

    int best = 0;
    double best_score = scores(0);
    for (int i = 1; i < scores.size(); i++) {
        if (scores(i) > best_score) {
            best_score = scores(i);
            best = i;
        }
    }
    return best;
}

void evaluateLDA(
    const shark::LinearClassifier<shark::RealVector>& model,
    const std::vector<std::vector<double>>& X_test_pca,
    const std::vector<int>& y_test,
    int num_classes
) {
    int correct = 0;
    std::vector<int> class_correct(num_classes, 0);
    std::vector<int> class_total(num_classes, 0);

    for (size_t i = 0; i < X_test_pca.size(); i++) {
        int pred = predictLDA(model, X_test_pca[i]);
        int true_label = y_test[i];

        if (pred == true_label) {
            correct++;
            class_correct[true_label]++;
        }
        class_total[true_label]++;
    }

    double acc = 100.0 * correct / X_test_pca.size();
    std::cout << "\n📊 LDA Test Accuracy: " << acc << "%\n";

    std::cout << "\n📌 Per-class accuracy:\n";
    for (int c = 0; c < num_classes; c++) {
        double a = class_total[c] > 0
                   ? 100.0 * class_correct[c] / class_total[c]
                   : 0.0;
        std::cout << "  Class " << c
                  << ": " << a
                  << "% (" << class_correct[c]
                  << "/" << class_total[c] << ")\n";
    }
}

// =====================
// EXPORT MODEL TO HEADER FILE
// =====================
void exportModelToHeader(
    const LinearClassifier<RealVector>& lda_model,
    const std::vector<double>& mean,
    const std::vector<double>& stddev,
    const real_2d_array& pca_matrix_V,  // PCA projection matrix
    int num_pca_components,
    const std::vector<std::string>& class_names,
    const std::string& output_path = "lda_model_fixed.h"
) {
    // Extract model parameters
    const RealMatrix& W = lda_model.decisionFunction().matrix();
    const RealVector& b = lda_model.decisionFunction().offset();
    
    size_t num_classes = W.size1();
    size_t num_features = mean.size();  // Original features before PCA
    size_t num_pca_dims = W.size2();    // Features after PCA
    
    // =============================
    // FIXED-POINT CONFIG
    // =============================
    const int FRAC_BITS = 16;
    const int SCALE = 1 << FRAC_BITS;

    auto float_to_fixed = [&](double x) -> int32_t {
        return static_cast<int32_t>(std::round(x * SCALE));
    };

    // Open output file
    std::ofstream file(output_path);
    if (!file.is_open()) {
        std::cerr << "❌ Failed to create header file: " << output_path << "\n";
        return;
    }
    
    // Write header guard
    file << "#ifndef LDA_MODEL_FIXED_H\n";
    file << "#define LDA_MODEL_FIXED_H\n\n";
    
    file << "#include <stdint.h>\n\n";  
    
    // Write metadata
    file << "// =============================================\n";
    file << "// LDA Model Parameters - Auto-generated\n";
    file << "// Training pipeline: Raw → Standardize → PCA → LDA\n";
    file << "// All values in Q16.16 fixed-point format\n";
    file << "// =============================================\n\n";
    
    file << "#define NUM_RAW_FEATURES " << num_features << "      // Original sensor features\n";
    file << "#define NUM_PCA_COMPONENTS " << num_pca_dims << "    // After PCA dimensionality reduction\n";
    file << "#define NUM_CLASSES " << num_classes << "           // Number of output classes\n\n";
    
    // Write class names
    file << "// =============================================\n";
    file << "// Class Names\n";
    file << "// =============================================\n";
    file << "static const char* class_names[NUM_CLASSES] = {\n";
    for (size_t i = 0; i < class_names.size(); i++) {
        file << "    \"" << class_names[i] << "\"";
        if (i < class_names.size() - 1) file << ",";
        file << "\n";
    }
    file << "};\n\n";
    
    // Write feature means
    file << "// =============================================\n";
    file << "// STEP 1: Standardization Parameters\n";
    file << "// =============================================\n\n";
    file << "// Feature means (Q16.16 fixed-point)\n";
    file << "static const int32_t feature_means_fixed_data[NUM_RAW_FEATURES] = {\n    ";
    for (size_t i = 0; i < mean.size(); i++) {
        file << float_to_fixed(mean[i]);
        if (i < mean.size() - 1) file << ", ";
    }
    file << "\n};\n\n";

    
    // Write feature inverse standard deviations
    file << "// Feature inverse standard deviations (Q16.16 fixed-point)\n";
    file << "// Pre-computed as 1/stddev for efficiency\n";
    file << "static const int32_t feature_stddevs_inv_fixed_data[NUM_RAW_FEATURES] = {\n    ";
    for (size_t i = 0; i < stddev.size(); i++) {
        double inv = 1.0 / stddev[i];
        file << float_to_fixed(inv);
        if (i < stddev.size() - 1) file << ", ";
    }
    file << "\n};\n\n";

    
    // Write PCA projection matrix
    file << "// =============================================\n";
    file << "// STEP 2: PCA Projection Matrix\n";
    file << "// =============================================\n\n";
    file << "// PCA matrix: [NUM_RAW_FEATURES][NUM_PCA_COMPONENTS] (Q16.16 fixed-point)\n";
    file << "// Usage: x_pca[j] = sum(x_std[i] * pca_matrix[i][j])\n";
    file << "static const int32_t pca_matrix_fixed_data[NUM_RAW_FEATURES][NUM_PCA_COMPONENTS] = {\n";
    for (size_t i = 0; i < num_features; i++) {
        file << "    {";
        for (size_t j = 0; j < num_pca_dims; j++) {
            file << float_to_fixed(pca_matrix_V[i][j]);
            if (j < num_pca_dims - 1) file << ", ";
        }
        file << "}";
        if (i < num_features - 1) file << ",";
        file << "\n";
    }
    file << "};\n\n";

    
    // Write LDA weight matrix
    file << "// =============================================\n";
    file << "// STEP 3: LDA Classification\n";
    file << "// =============================================\n\n";
    file << "// LDA weights: [NUM_CLASSES][NUM_PCA_COMPONENTS] (Q16.16 fixed-point)\n";
    file << "// Each row represents weights for one class\n";
    file << "static const int32_t lda_weights_fixed_data[NUM_CLASSES][NUM_PCA_COMPONENTS] = {\n";
    for (size_t i = 0; i < num_classes; i++) {
        file << "    {";
        for (size_t j = 0; j < num_pca_dims; j++) {
            file << float_to_fixed(W(i, j));
            if (j < num_pca_dims - 1) file << ", ";
        }
        file << "}";
        if (i < num_classes - 1) file << ",";
        file << "\n";
    }
    file << "};\n\n";

    
    // Write bias vector
    file << "// LDA bias: [NUM_CLASSES] (Q16.16 fixed-point)\n";
    file << "// One bias value per class\n";
    file << "static const int32_t lda_bias_fixed_data[NUM_CLASSES] = {\n    ";
    for (size_t i = 0; i < num_classes; i++) {
        file << float_to_fixed(b(i));
        if (i < num_classes - 1) file << ", ";
    }
    file << "\n};\n\n";
   
    // Close header guard
    file << "#endif // LDA_MODEL_FIXED_H\n";
    
    file.close();
    
    std::cout << "\n✅ Model exported successfully to: " << output_path << "\n";
    std::cout << "   Parameters exported:\n";
    std::cout << "   - Feature means: " << mean.size() << " values\n";
    std::cout << "   - Feature stddevs (inverted): " << stddev.size() << " values\n";
    std::cout << "   - PCA matrix: " << num_features << "x" << num_pca_dims << " matrix\n";
    std::cout << "   - LDA weights: " << num_classes << "x" << num_pca_dims << " matrix\n";
    std::cout << "   - LDA bias: " << num_classes << " values\n";
    std::cout << "   - Class names: " << class_names.size() << " classes\n";
    std::cout << "   - Format: Q16.16 fixed-point (16 fractional bits)\n";
}

// =====================
// MAIN
// =====================
int main(int argc, char* argv[]) {
    if (argc < 2) {                                                         //dataset must be provided 
        std::cerr << "Usage: " << argv[0] << " <dataset_path>\n";
        return EXIT_FAILURE;
    }
 
    std::string dataset_path = argv[1];                                     //read dataset path

    // Raw dataset containers (ML convention)
    std::vector<std::vector<double>> X;
    std::vector<int> y;
    std::vector<std::string> class_names;

    // Load dataset
    try {
        loadDataset(dataset_path, X, y, class_names);
    } catch (const std::exception& e) {
        std::cerr << "❌ Dataset loading failed: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    if (X.empty()) {
        std::cerr << "❌ Dataset is empty after loading\n";
        return EXIT_FAILURE;
    }

    // Dataset sanity & statistics
    computeDatasetStats(X);

    std::cout << "\n📌 Class label mapping:\n";
    for (size_t i = 0; i < class_names.size(); i++) {
        std::cout << "  " << i << " -> " << class_names[i] << "\n";
    }

    // Stratified train/test split
    std::vector<std::vector<double>> X_train, X_test;
    std::vector<int> y_train, y_test;

    constexpr double TEST_SPLIT = 0.20;
    constexpr int RANDOM_SEED = 42;

    try {
        stratified_train_test_split(
            X, y,
            X_train, y_train,
            X_test, y_test,
            TEST_SPLIT,
            RANDOM_SEED
        );
    } catch (const std::exception& e) {
        std::cerr << "❌ Train/test split failed: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    // Final confirmation
    std::cout << "\n✅ Dataset preparation completed successfully\n";
    std::cout << "   Train samples: " << X_train.size() << "\n";
    std::cout << "   Test  samples: " << X_test.size() << "\n";
    std::cout << "\n🎯 Ready for standardization → PCA → LDA\n";

    // Standardize data
    std::vector<double> mean, stddev;
    standardizeData(X_train, X_test, mean, stddev);

    logData(X_train, mean, stddev, "X_train"); 
    logData(X_test,  mean, stddev, "X_test");

    std::vector<std::vector<double>> X_train_pca, X_test_pca;
    int num_pca_components = 6; 

    real_2d_array pca_matrix_V; 
    computePCA(X_train, X_test, X_train_pca, X_test_pca, num_pca_components, pca_matrix_V);

    // log first few PCA samples
    logData(X_train_pca,
            std::vector<double>(num_pca_components,0.0),
            std::vector<double>(num_pca_components,1.0),
            "X_train PCA",
            5,
            true);   // set isPCA=true

    logData(X_test_pca,
            std::vector<double>(num_pca_components,0.0),
            std::vector<double>(num_pca_components,1.0),
            "X_test PCA",
            5,
            true);

    // Train LDA on PCA-reduced training data
    LinearClassifier<RealVector> lda_model;

    trainLDA(X_train_pca, y_train, lda_model);
    logLDAModel(lda_model);

    // Evaluate on PCA-reduced test data
    evaluateLDA(
        lda_model,
        X_test_pca,
        y_test,
        class_names.size()
    );

    // Export model to header file
    exportModelToHeader(
        lda_model,
        mean,
        stddev,
        pca_matrix_V,        // PCA matrix
        num_pca_components,  // Number of components  
        class_names,
        "lda_model_fixed.h"
    );

    return EXIT_SUCCESS;
}