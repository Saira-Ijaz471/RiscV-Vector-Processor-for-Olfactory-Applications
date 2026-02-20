#pragma once

#define NUM_CLASSES 4
#define NUM_FEATURES 6
#define PCA_COMPONENTS 6
#define SVM_GAMMA 0.166667

// Test Accuracy: 63.50%
// Support Vectors: 697

static const char* CLASS_NAMES[4] = {
    "AQ_Wines",
    "Ethanol",
    "HQ_Wines",
    "LQ_Wines"
};
