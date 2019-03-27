#!/usr/bin/env python3

import click
import numpy as np

from kmeans import bisecting_kmeans, kmeans, visualize_clusters

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('datafile')
@click.option(
    '-k', '--k', default=2, show_default=True,
    help='Number of output clusters')
@click.option(
    '-e', '--epochs', default=10, show_default=True,
    help='Number of random initialization (to find global optimum)')
@click.option(
    '-i', '--max-iter', default=100, show_default=True,
    help='Maximum iterations')
@click.option(
    '-b', '--bisecting', default=False, is_flag=True,
    help='Use bisecting k-means algorithm')
@click.option(
    '-v', '--verbose', default=False, is_flag=True, help='Verbose output')
def main(datafile, k, verbose, max_iter, epochs, bisecting):
    """CLI for testing the k-means clustering algorithm."""
    points = np.loadtxt(datafile)
    algorithm = kmeans
    if bisecting:
        algorithm = bisecting_kmeans
    clusters = algorithm(
        points=points, k=k, verbose=verbose, max_iter=max_iter, epochs=epochs)
    visualize_clusters(clusters)


if __name__ == '__main__':
    main()
