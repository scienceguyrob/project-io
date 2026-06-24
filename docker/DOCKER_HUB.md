# Docker Hub Repository Content

This file contains the text used for the project-io Docker Hub repository at
https://hub.docker.com/r/scienceguyrob/project-io

---

## Short Description

```
Self-paced machine learning notebooks with JupyterLab. No setup required.
```

---

## Category

**Machine Learning & AI **

---

## Repository Overview

---

# project-io

A self-paced series of Jupyter notebooks covering machine learning from the ground up, packaged as a ready-to-run Docker container. Pull the image, run one command, and JupyterLab opens in your browser with everything pre-installed and pre-configured. No Python installation required.

The series starts with Python fundamentals and builds through statistics, classical machine learning algorithms, ensemble methods, neural networks, unsupervised learning, and model evaluation. Datasets, visualisations, and reference web pages are all included in the image.

**Source:** [github.com/scienceguyrob/project-io](https://github.com/scienceguyrob/project-io)  
**Author:** Dr Rob Lyon

---

## Quick Start

```bash
docker pull scienceguyrob/project-io
docker run --rm -p 8888:8888 scienceguyrob/project-io
```

Then open your browser and go to:

```
http://localhost:8888
```

No password or token is required.

---

## What is Included

- 14 numbered notebooks and 3 bonus notebooks
- 11 paired challenge notebooks applying techniques to real-world domains
- All datasets used in the notebooks and challenges
- A searchable index covering every topic, algorithm, function, and concept across the series
- Reference pages for notebooks, challenges, and datasets

Once the container is running, open the project home page at:

```
http://localhost:8888/files/index.html
```

---

## Notebooks

| Notebook | Title |
|---|---|
| 0 | A Python Primer for Machine Learning |
| 1 | Introduction to Python Programming |
| 2 | Control Flow, Functions, and Decision Boundaries |
| 3 | NumPy, Statistics, and Feature Analysis |
| 4 | Pandas, Optimisation, and the Curse of Dimensionality |
| 5 | Regression and Classification |
| 6 | Naive Bayes, Decision Trees, Random Forests, and SVMs |
| 6 Bonus | Introduction to Neural Networks |
| 7 | Visualisation for Machine Learning and Feature Selection |
| 7 Bonus | Imbalanced Learning |
| 8 | Unsupervised Learning and Clustering |
| 8 Bonus | Mixture Models and the EM Algorithm |
| 9 | OPTICS, Dimensionality Reduction, and PCA |
| 10 | Classifier Evaluation |
| 11 | Evaluation Metrics and the Bias-Variance Trade-off |
| 12 | ROC Curves, Precision-Recall Curves, and Clustering Evaluation |
| 13 | Ensemble Learning |

---

## Saving Your Work

By default, changes made inside the container are lost when it stops. To persist your work, mount a local folder into the container:

**macOS and Linux:**
```bash
docker run --rm -p 8888:8888 \
  -v /path/to/your/folder:/home/notebook/work \
  scienceguyrob/project-io
```

**Windows (PowerShell):**
```powershell
docker run --rm -p 8888:8888 `
  -v C:/Users/YourName/folder:/home/notebook/work `
  scienceguyrob/project-io
```

A `work/` folder will appear in the JupyterLab file browser. Anything saved there is written to your local machine immediately.

---

## Python Environment

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

## Port

JupyterLab runs on port **8888** inside the container. If that port is already in use on your machine, map it to a different local port:

```bash
docker run --rm -p 8889:8888 scienceguyrob/project-io
```

Then open `http://localhost:8889` instead.

---

## Full Instructions

For complete setup instructions including Docker installation on Windows, macOS, and Linux, volume mounting, troubleshooting, and more, see the
[project-io Docker guide](https://github.com/scienceguyrob/project-io/blob/main/docker/DOCKER_BUILD.md)
in the GitHub repository.

---

## License

Copyright (c) 2026 Robert Lyon. All Rights Reserved.  
Permission is granted solely to read, study, and analyse this material for personal educational purposes.
