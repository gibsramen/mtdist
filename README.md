# mtdist

**mtdist** is a Python package for computing **m**ixed-**t**ype **dist**ance metrics on high-dimensional data. Standard distance metrics (Euclidean, Manhattan, etc.) are usually restricted to numeric data, making integration with other data types (categorical, ordinal, etc.) difficult. Distance metrics that are built to handle mixed-type data, such as Gower distance (Gower 1971), are often only available in R, increasing the burden on data scientists.

mtdist aims to bring these valuable distance metrics to Python, allowing researchers to more easily analyze mixed-type data for clustering, visualization, and more.

## References

[1] Gower, J. C. "A General Coefficient of Similarity and Some of Its Properties." _Biometrics_ 27, no. 4 (1971): 857-71. doi:10.2307/2528823.
