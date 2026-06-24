<img src="web/project_io_banner.png" alt="project-io logo" width="500">

# project-io

A self-paced series of Jupyter notebooks covering Python programming, statistics, classical machine learning, ensemble methods, neural networks, unsupervised learning, and model evaluation. Built to help anyone learn machine learning from the ground up, starting with no assumed programming knowledge.

---

## Quick Start

The easiest way to run project-io is to pull the pre-built image from Docker Hub. You do not need to clone this repository or install Python.

**Step 1 — Pull the image:**

```bash
docker pull scienceguyrob/project-io
```

**Step 2 — Run the container:**

```bash
docker run --rm -p 8888:8888 scienceguyrob/project-io
```

**Step 3 — Open your browser** and go to:

```
http://localhost:8888
```

JupyterLab will open with all notebooks and files ready to use. No password or token is required.

The image is available on Docker Hub at [hub.docker.com/r/scienceguyrob/project-io](https://hub.docker.com/r/scienceguyrob/project-io).

---

## Saving Your Work

By default, changes made inside the container are lost when it stops. To persist your work, mount a local folder into the container:

```bash
# macOS and Linux
docker run --rm -p 8888:8888 \
  -v /path/to/your/work:/home/notebook/work \
  scienceguyrob/project-io

# Windows (PowerShell)
docker run --rm -p 8888:8888 -v C:/Users/YourName/work:/home/notebook/work scienceguyrob/project-io
```

A `work/` directory will appear in the JupyterLab file browser. Anything saved there is written to your local machine immediately and survives the container stopping.

### Port conflicts

If port 8888 is already in use, map the container to a different local port:

```bash
docker run --rm -p 8889:8888 scienceguyrob/project-io
```

Then open `http://localhost:8889` instead.

### Full Docker instructions

See [`docker/DOCKER_BUILD.md`](docker/DOCKER_BUILD.md) for complete step-by-step instructions covering macOS, Windows, and Linux, including Docker installation, volume mounts, and troubleshooting.

---

## Notebooks

The series currently has 14 numbered notebooks plus 3 bonus notebooks, organised so each one builds on the previous.

| Notebook | Title | Topics |
|---|---|---|
| 0 | A Python Primer for Machine Learning | Variables, types, control flow, functions, scope, data structures |
| 1 | Introduction to Python Programming | JupyterLab, arithmetic, strings, lists, dicts, sets, tuples, straight lines, residuals |
| 2 | Control Flow, Functions, and Decision Boundaries | Comparison operators, loops, list comprehensions, functions, loss landscapes |
| 3 | NumPy, Statistics, and Feature Analysis | Arrays, broadcasting, distributions, correlation, Cohen's d, feature selection |
| 4 | Pandas, Optimisation, and the Curse of Dimensionality | Series, DataFrames, GroupBy, gradient descent, dimensionality |
| 5 | Regression and Classification | Linear regression, least squares, logistic regression, sigmoid, MLE |
| 6 | Naïve Bayes, Decision Trees, Random Forests, and SVMs | Bayes' theorem, CART, bagging, maximum-margin hyperplane, kernel trick |
| 6 Bonus | Introduction to Neural Networks | McCulloch–Pitts neuron, perceptron, MLP, backpropagation, activation functions |
| 7 | Visualisation for Machine Learning and Feature Selection | Histograms, KDE, box plots, heatmaps, pairplots, filter/wrapper/embedded methods |
| 7 Bonus | Imbalanced Learning | Undersampling, oversampling, SMOTE, Borderline-SMOTE, cost-sensitive learning |
| 8 | Unsupervised Learning and Clustering | Distance metrics, k-NN, k-means, hierarchical clustering, DBSCAN |
| 8 Bonus | Mixture Models and the EM Algorithm | GMMs, Expectation–Maximisation, covariance types, AIC/BIC, anomaly detection |
| 9 | OPTICS, Dimensionality Reduction, and PCA | OPTICS, covariance matrices, eigenvectors/eigenvalues, scree plots, biplots |
| 10 | Classifier Evaluation | Memorisation problem, i.i.d. assumption, No Free Lunch, k-fold CV, confusion matrix |
| 11 | Evaluation Metrics and the Bias–Variance Trade-off | Precision, recall, F1, MCC, Cohen's kappa, regularisation, L1/L2 |
| 12 | ROC Curves, Precision–Recall Curves, and Clustering Evaluation | AUC, PR curves, WCSS, silhouette score, Davies–Bouldin index |
| 13 | Ensemble Learning | Bagging, bootstrap aggregating, out-of-bag evaluation, boosting, AdaBoost |

---

## Algorithms Covered

Linear Regression, Ordinary Least Squares, Logistic Regression, Maximum Likelihood Estimation, Gradient Descent, Naïve Bayes (categorical and Gaussian), Decision Tree (CART), Random Forest, Support Vector Machine (linear and RBF kernel), Ridge Regression, Lasso, k-Nearest Neighbours, k-Means, Hierarchical Clustering, DBSCAN, OPTICS, Gaussian Mixture Model, Expectation–Maximisation, Principal Component Analysis, Perceptron, Neural Network (MLP), SMOTE, Borderline-SMOTE, Isolation Forest, One-Class SVM, Bagging, AdaBoost.

---

## Challenges

Most notebooks include a paired challenge file that applies the notebook's techniques to a realistic domain-specific dataset. The challenges are designed to be genuinely tricky — not just repetition of worked examples.

| Challenge | Topic | Domain |
|---|---|---|
| 5 | Multinomial logistic regression | Particle physics (CERN detector signatures) |
| 5a | Binary logistic regression | Operational oceanography (buoy sea state) |
| 6a | Decision tree classifier | Remote sensing (Sentinel-2 land cover) |
| 6b | Gaussian Naïve Bayes with missing data | Industrial gas turbine monitoring |
| 7a | RBF SVM and feature scaling | Astrophysics (variable star classification) |
| 7b | Random Forest with GridSearchCV | Cybersecurity (network intrusion detection) |
| 8a | K-Means, GMM, semi-supervised KNN | Extragalactic astrophysics (galaxy typing) |
| 9 | PCA and MLP | Radio astronomy (pulsar candidate classification) |
| 10 | K-NN and SMOTE | Satellite telemetry anomaly detection |
| 11 | SVM overfitting and regularisation | Space weather (solar flare classification) |
| 12 | SVM vs Random Forest vs MLP | Quantitative finance (stock movement prediction) |

---

## Datasets

All challenge datasets are synthetic but are grounded in the physics, geometry, and statistics of their real-world domains. Notebook datasets include a mix of synthetic data and standard benchmarks loaded directly from scikit-learn.

| File | Used in | Description |
|---|---|---|
| `Notebook_4/data/Universities.csv` | Notebook 4 | Real US university programme completion statistics (207 records) |
| `Notebook_6/data/clouds.csv` | Notebook 6 | Synthetic cloud observations, 4 features, 3 classes, 1,800 rows |
| `Notebook_6/data/clouds_complex.csv` | Notebook 6 | As above but 10 features, non-linear relationships, label noise, 6,000 rows |
| `Notebook_6/data/credit.csv` | Notebook 6 | Synthetic credit application records, 6 features, binary risk label, 5,000 rows |
| `Notebook_6/data/robsat.csv` | Challenge 6a | Synthetic Sentinel-2 inspired pixel data, 8 features, 3 land cover classes |
| `Notebook_6/data/temp_monitoring.csv` | Challenge 6b | Synthetic gas turbine telemetry, 7 sensor readings, 4 operational states, missing values |
| `Notebook_7/data/spectra.csv` | Challenge 7a | Synthetic variable star photometry, 9 features, 7 star types |
| `Notebook_7/data/security.csv` | Challenge 7b | Synthetic network connection records, 15 features, 4 traffic classes |
| `Notebook_8/data/galaxies.csv` | Challenge 8a | Synthetic galaxy survey data, 8 features, 3 types, 10,000 labelled rows |
| `Notebook_8/data/unknown_galaxies.csv` | Challenge 8a | As above but unlabelled, 2,000 rows |
| `Notebook_9/data/radio.csv` | Challenge 9 | Synthetic pulsar candidate data, 20 features, binary label, 10% positive rate |
| `Notebook_10/data/satellite.csv` | Challenge 10 | Synthetic satellite telemetry, 10 features, 4 anomaly classes |
| `Notebook_11/data/sol.csv` | Challenge 11 | Synthetic solar observatory data, 12 features, 3 activity classes |
| `Notebook_12/data/market.csv` | Challenge 12 | Synthetic stock market technical indicators, 12 features, 3 movement classes |
| `Notebook_5/data/particles.csv` | Challenge 5 | Synthetic CERN-inspired detector data, 8 features, 3 particle types |
| `Notebook_5/data/sea_state.csv` | Challenge 5a | Synthetic ocean buoy measurements, 5 features, binary sea state label |
| sklearn built-in | Notebooks 7, 13 | Breast Cancer Wisconsin Diagnostic dataset (569 rows, 30 features) |
| sklearn built-in | Notebook 9 | Iris dataset (150 rows, 4 features) |

---

## Repository Structure

```
project-io/
├── index.html                        <- series home page
├── web/
│   ├── search.html                   <- searchable index of all topics
│   ├── challenges.html               <- challenge reference guide
│   ├── data.html                     <- dataset reference
│   ├── how_to.html                   <- container usage guide
│   └── dook.png                      <- project logo
├── docker/
│   ├── Dockerfile                    <- builds the project-io image
│   ├── .dockerignore                 <- excludes unnecessary files from the build
│   └── DOCKER_BUILD.md               <- step-by-step build and run instructions
├── Notebook_0/
│   └── Notebook_0.ipynb
├── Notebook_1/
│   ├── Notebook_1.ipynb
│   ├── visualisations/
│   │   └── Figure_N.py
│   └── data/
├── Notebook_2/ ... Notebook_13/      <- same structure throughout
├── Notebook_6_Bonus/
├── Notebook_7_Bonus/
└── Notebook_8_Bonus/
```

Each numbered notebook folder follows the same pattern:

```
Notebook_N/
├── Notebook_N.ipynb          <- main notebook
├── Challenge_N.ipynb         <- paired challenge (where one exists)
├── visualisations/
│   └── Figure_N.py           <- interactive figure scripts
└── data/
    └── *.csv                 <- datasets for that notebook
```

---

## Web Pages

The `web/` directory contains a small static site that documents the series and is designed to be opened directly in a browser via JupyterLab's file server.

- **`index.html`** — series overview and notebook listing
- **`search.html`** — full-text search across 2,500+ indexed entries covering every heading, concept, algorithm, function, and library across all notebooks; built on [Fuse.js](https://fusejs.io/)
- **`challenges.html`** — reference guide for all challenge notebooks including dataset variables, class distributions, and goals
- **`data.html`** — reference for every dataset used in the series
- **`how_to.html`** — guide to running the series via Docker

Once the container is running, open these pages directly in your browser using their `/files/` URL — for example:

```
http://localhost:8888/files/index.html
http://localhost:8888/files/web/search.html
```

The `/files/` prefix is a JupyterLab URL keyword that serves a file as a raw webpage. It is not a folder — it is JupyterLab's way of distinguishing "serve this file directly" from "open this file in the editor".

---

## Building from Source

If you prefer to build the image yourself rather than pulling it from Docker Hub, clone the repository and run the following from the project root:

```bash
docker build -f docker/Dockerfile -t project-io .
docker run --rm -p 8888:8888 project-io
```

See [`docker/DOCKER_BUILD.md`](docker/DOCKER_BUILD.md) for full instructions.

---

## Running Without Docker

The notebooks are standard Jupyter `.ipynb` files and can be opened with JupyterLab, VS Code, or any other compatible tool. Install the dependencies manually using pip:

```bash
pip install \
  numpy==2.2.4 \
  pandas==2.2.3 \
  matplotlib==3.10.0 \
  seaborn==0.13.2 \
  scipy==1.15.2 \
  scikit-learn==1.6.1 \
  ipywidgets==8.1.7 \
  ipython==8.30.0 \
  imbalanced-learn==0.14.1 \
  jupyterlab==4.3.6 \
  ipympl==0.9.7
```

The series is developed and tested with **JupyterLab**. Using a different tool is possible but some interactive widgets may not render correctly.

**Full dependency list:**

| Library | Version |
|---|---|
| Python | 3.13.2 |
| numpy | 2.2.4 |
| pandas | 2.2.3 |
| matplotlib | 3.10.0 |
| seaborn | 0.13.2 |
| scipy | 1.15.2 |
| scikit-learn | 1.6.1 |
| ipywidgets | 8.1.7 |
| IPython | 8.30.0 |
| imbalanced-learn | 0.14.1 |

---

## Author

Dr Rob Lyon — [github.com/scienceguyrob](https://github.com/scienceguyrob)

---

## License

Copyright © 2026 Robert Lyon. All Rights Reserved.

Permission is granted solely to read, study, and analyse this material for personal educational purposes. No other rights are granted. Without prior written consent you may not copy, redistribute, modify, adapt, incorporate into other projects, or use this material commercially. All intellectual property rights remain exclusively vested in the author.