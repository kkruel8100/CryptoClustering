### Welcome to Kim's Unsupervised Learning Challenge

where we use KMeans, PCA, and StandardScaler for unsupervised learning.

#### Overview

Showcasing Pandas, Python, KMeans, PCA, and StandardScaler, the challenge gathers data from crypto market data. It determines optimal k value and creates a scatter plot using original scaled data. Then it optimizes clusters with PCA. It determines the optimal k value and creates a scatter plot using pca data. Using the PCA component weights, it determines the features with the strongest positive and negative influece.

#### Program

    └───root
        │   Crypto_Clustering.ipynb
        │   README.md
        ├───helpers
        │       k_means_helpers.py
        │       pca_helpers.py
        └───Resources
                crypto_market_data.csv

Step 1:

Navigate to the "Crypto_Clustering.ipynb" file in GitHub repo. The output for each panel can be viewed in the "Preview" panel.

Alternatively,

Step 1:

Clone the repository.

Step 2:

In terminal, activate your conda environment and type conda list scikit-learn

Step 3:

If the library is installed in your conda environment, the terminal will output the package name and version number. If installed, skip to Step 5. If not installed, go to Step 4.

Step 4:

pip install -U scikit-learn

Step 5:

Open "Crypto_Clustering.ipynb"

Step 6:

Run remaining panels.

#### Resources

- crypto_market_data.csv supplied by ASU edX Boot Camps LLC
