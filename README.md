# k-means Clustering

This is a simple pythonic implementation of the two centroid-based
partitioned clustering algorithms: **k-means** and **bisecting k-means**.

## Requirements

To run this program, you need to have python 3.x installed with
following packages:

- numpy (for matrix calculations)
- matplotlib (for visualization)
- click (for command line interface)

You can install these with the following command:

    pip3 install -r requirements.txt

## Usage

First of all, you need to have a data file. A sample data file `demo/data.txt`
is included in this repo.

For running the program on the sample dataset, run:

    python3 test_kmeans.py --verbose

To test bisecting k-means, use your own datasets, and change various
clustering paramters, see help text.

    python3 test_kmeans.py --help

## Author

Manish Munikar <munikarmanish@gmail.com>
