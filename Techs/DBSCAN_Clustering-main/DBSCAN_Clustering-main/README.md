# DBSCAN Clustering

## Introduction

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a popular unsupervised machine learning algorithm used for clustering spatial data points. Unlike K-Means clustering, DBSCAN does not require the number of clusters to be specified in advance and is capable of identifying clusters of arbitrary shapes and sizes. This repository provides an overview of DBSCAN clustering along with examples and implementations in Python.


## How DBSCAN Clustering Works

DBSCAN clustering works by grouping together closely packed data points into clusters based on two key parameters: epsilon (ε) and minimum points (MinPts).
- **Epsilon (ε)**: The radius around each data point within which neighboring points are considered part of the same cluster.
- **Minimum Points (MinPts)**: The minimum number of data points required to form a dense region (core point) in order for it to be considered as part of a cluster.

### Detailed Steps:

1. **Core Point Identification**:
   - For each data point, identify its ε-neighborhood (including the point itself).
   - If the number of points in the neighborhood is greater than or equal to MinPts, the point is considered a core point.

2. **Cluster Expansion**:
   - Assign each core point and its ε-neighborhood to the same cluster.
   - If a non-core point is within the ε-neighborhood of a core point, it is assigned to the same cluster as the core point.
   - Repeat this process until all data points have been assigned to clusters or marked as noise.

3. **Noise Identification**:
   - Any data points that are not assigned to any cluster are considered noise points.

## Key Parameters in DBSCAN Clustering

- **Epsilon (ε)**: The radius around each data point within which neighboring points are considered part of the same cluster.
- **Minimum Points (MinPts)**: The minimum number of data points required to form a dense region (core point) in order for it to be considered as part of a cluster.

## Advantages of DBSCAN Clustering

- Does not require the number of clusters to be specified in advance.
- Capable of identifying clusters of arbitrary shapes and sizes.
- Robust to noise and outliers.
- Does not assume clusters are spherical or have similar densities.

## Limitations of DBSCAN Clustering

- Sensitivity to the choice of ε and MinPts parameters.
- May struggle with clusters of varying densities or non-uniform distribution of data points.
- Computationally more expensive compared to K-Means clustering, especially for large datasets.

## Applications of DBSCAN Clustering

- Image segmentation and object detection.
- Anomaly detection in cybersecurity.
- Identifying spatial clusters in geographic data.
- Customer segmentation in marketing.
- Identifying natural groupings in biological data.

## Datasets

This repository includes sample datasets in CSV format that can be used to practice DBSCAN clustering. The datasets contain spatial data points with relevant attributes for clustering tasks.

##  Repository Structure

```sh
└── DBSCAN_Clustering/
    ├── README.md
    ├── Wine_Dataset_DBSCAN.ipynb
    ├── requirements.txt
    ├── wine-clustering.csv
    └── wine-dataset-EDA.html
```

---

##  Getting Started

***Requirements***

Ensure you have the following dependencies installed on your system:

* **JupyterNotebook**

###  Installation

1. Clone the DBSCAN_Clustering repository:

```sh
git clone https://github.com/sumony2j/DBSCAN_Clustering.git
```

2. Change to the project directory:

```sh
cd DBSCAN_Clustering
```

3. Install the dependencies:

```sh
pip install -r requirements.txt
```

###  Running DBSCAN_Clustering

Use the following command to run DBSCAN Clustering:

```sh
jupyter nbconvert --execute notebook.ipynb
```

---


##  Contributing

Contributions are welcome! Here are several ways you can contribute:

- **[Submit Pull Requests](https://github.com/sumony2j/DBSCAN_Clustering.git/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.
- **[Join the Discussions](https://github.com/sumony2j/DBSCAN_Clustering.git/discussions)**: Share your insights, provide feedback, or ask questions.
- **[Report Issues](https://github.com/sumony2j/DBSCAN_Clustering.git/issues)**: Submit bugs found or log feature requests for Dbscan_clustering.

<details closed>
    <summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your GitHub account.
2. **Clone Locally**: Clone the forked repository to your local machine using a Git client.
   ```sh
   git clone https://github.com/sumony2j/DBSCAN_Clustering.git
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to GitHub**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.

Once your PR is reviewed and approved, it will be merged into the main branch.

</details>

---
