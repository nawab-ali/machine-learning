/**
 * @file KMeans.h
 * @brief This file implements the KMeans clustering algorithm.
 * @author Nawab Ali
 */

#ifndef KMEANS_H
#define KMEANS_H

#include "Point.h"
#include "Cluster.h"
#include<cmath>
#include<random>
#include<vector>
#include<functional>

/**
 * @brief This function returns k random points from the dataset.
 * @param k Number of clusters
 * @param points Vector of points
 * @return Vector of k random points
 */
template <class T>
vector<Point<double>> get_random_points(unsigned int k, const vector<Point<T>>& points) {
	random_device device;
	mt19937 generator(device());
	uniform_int_distribution<int> distribution(0, points.size()-1);
	vector<Point<double>> random_points;

	if (k > points.size()) {
		throw invalid_argument("k > points.size()");
	}

	for (int i = 0; i < k; i++) {
		int index = distribution(generator);
		random_points.push_back(points[index]);
	}

	return(random_points);
}

/**
 * @brief This function initializes the centroid of each cluster.
 * @param clusters Vector of clusters
 * @param random_points Vector of random points from dataset.
 * @return Void
 */
template <class T>
void init_cluster_centroid(vector<Cluster<T>>& clusters,
													 vector<Point<double>>& random_points) {
	int i = 0;

	if (clusters.size() > random_points.size()) {
		throw invalid_argument("clusters.size() > random_points.size()");
	}

	// Initialize cluster centroids with random points from dataset
	for_each(clusters.begin(), clusters.end(),
					 [&random_points, &i](Cluster<T>& cluster) {
		cluster.set_centroid(random_points[i++]);
	});
}

/**
 * @brief This function calculates the euclidean distance between
 * 2 n-dimensional Points.
 * @param point1 Point 1
 * @param point2 Point 2
 * @return Euclidean distance between point1 and point2
 */
template <class T>
double euclidean_distance(const Point<T>& point1, const Point<T>& point2) {
	double distance = 0.0;

	if (point1.get_dimensions() != point2.get_dimensions()) {
		throw invalid_argument("point1.get_dimensions() != point2.get_dimensions()");
	}

	for (int i = 0; i < point1.get_dimensions(); i++) {
		T x1 = point1.get_coordinates()[i];
		T x2 = point2.get_coordinates()[i];
		distance += pow(x1-x2, 2);
	}

	return(sqrt(distance));
}

/**
 * @brief This function returns the cluster nearest to the point
 * @param point n-dimensional Point
 * @param clusters vector of k clusters
 * @return cluster_id of the nearest cluster
 */
template <class T>
int get_nearest_cluster(const Point<T>& point, vector<Cluster<T>>& clusters) {
	if (clusters.size() == 0) {
		throw invalid_argument("clusters.size() == 0");
	}

	Cluster<T> nearest_cluster = clusters[0];
	clusters[0].update_centroid();
	double min_distance = euclidean_distance(point, clusters[0].get_centroid());

	for (int i = 1; i < clusters.size(); i++) {
		clusters[i].update_centroid();
		double distance = euclidean_distance(point, clusters[i].get_centroid());
		if (distance < min_distance) {
			min_distance = distance;
			nearest_cluster = clusters[i];
		}
	}

	return(nearest_cluster.get_id());
}

/**
 * @brief This function categorizes points into clusters.
 * @param points vector of n-dimensional Points
 * @param clusters vector of k clusters
 * @return None
 */
template <class T>
void categorize_points_into_clusters(vector<Point<T>>& points, vector<Cluster<T>>& clusters) {
	for_each(points.begin(), points.end(), [&clusters](Point<T>& point) {
			int nearest_cluster_id = get_nearest_cluster(point, clusters);

			if (nearest_cluster_id < 0 || nearest_cluster_id >= clusters.size()) {
				throw out_of_range("nearest_cluster_id < 0 || nearest_cluster_id >= clusters.size()");
			}

			clusters[nearest_cluster_id].add_point(point);
	});

	for_each(clusters.begin(), clusters.end(), [](Cluster<T>& cluster) {
		cluster.update_centroid();
	});
}

/**
 * @brief This function removes points from clusters.
 * @param clusters vector of clusters
 * @return None
 */
template <class T>
void remove_points_from_clusters(vector<Cluster<T>>& clusters) {
	vector<Point<T>> empty_points;

	for_each(clusters.begin(), clusters.end(), [empty_points](Cluster<T>& cluster) {
		cluster.set_points(empty_points);
	});
}

/**
 * @brief This function implements the KMeans clustering algorithm.
 * @param k Number of clusters
 * @param points Vector of points
 * @param max_iterations Maximum number of iterations to run the clustering
 * algorithm.
 * @return Vector of clusters
 */
template <class T>
vector<Cluster<T>>
KMeans(unsigned int k, vector<Point<T>>& points, long max_iterations=1000) {
	vector<Cluster<T>> clusters;
	vector<Point<double>> random_points;

	if (k > points.size()) {
		throw invalid_argument("k > points.size()");
	}

	for (int i = 0; i < k; i++) {
		clusters.push_back(Cluster<T>(i));
	}

	random_points = get_random_points<T>(k, points);
	init_cluster_centroid<T>(clusters, random_points);

	for (int i = 0; i < max_iterations; i++) {
		remove_points_from_clusters<T>(clusters);
		categorize_points_into_clusters<T>(points, clusters);
	}

	return(clusters);
}

#endif //KMEANS_H
