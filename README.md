# k-means Clustering

This is a simple pythonic implementation of the two centroid-based
partitioned clustering algorithms: **k-means** and **bisecting k-means**.

## Requirements

To run this program, you need to have python 3.x installed with
following packages:

- numpy (for computation)
- matplotlib (for visualization)
- click (for command line interface)

You can install these with the following command:

    pip3 install -r requirements.txt

## Usage

First of all, you need to have a data file. A sample data file `data.txt`
is included in this repo. It contains 100 two-dimensional Cartesian points with
10 Gaussian clusters. The data can be visualized at `data.pdf`.

Now you can run and test the two clustering algorithms. The sample result is
shown in `result.pdf`.

### Standard k-means

For running the program on the sample dataset, run:

    python3 test_kmeans.py --verbose

To use your own datasets and change various clustering paramters, see help
text.

    python3 test_kmeans.py --help

### Bisecting k-means

Bisecting k-means internally uses the standard k-means with k=2.

For running the program on the sample dataset, run:

    python3 test_bisecting.py --verbose

To use your own datasets and change various clustering paramters, see help
text.

    python3 test_bisecting.py --help


## Author

Manish Munikar <munikarmanish@gmail.com>
