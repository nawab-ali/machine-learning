/**
 * @file main.cpp
 * @brief This file implements the main thread of execution.
 * @author Nawab Ali
 */

#include<vector>
#include<string>
#include "util.h"
#include "Point.h"
#include "KMeans.h"
#include "Cluster.h"
#include "boost/program_options.hpp"

typedef struct {
	unsigned int k;
	unsigned int num_features;
	string filename;
	unsigned int max_iterations;
} cmd_line_args_t;

/**
 * @brief This function parses command line arguments.
 * @param argc Number of command line arguments
 * @param argv Command line arguments
 * @return Parsed command line arguments
 */
cmd_line_args_t parse_command_line(int argc, char** argv) {
	cmd_line_args_t cmd_line_args;
	namespace po = boost::program_options;
	po::variables_map vm;
	po::options_description desc("Options");

	desc.add_options()
		("help", "help message")
		("k", po::value<unsigned int>(&cmd_line_args.k)->required(), "number of clusters")
		("num_features", po::value<unsigned int>(&cmd_line_args.num_features)->required(),
		 "number of data features")
		("max_iterations", po::value<unsigned int>(&cmd_line_args.max_iterations)->default_value(1000),
		 "maximum iterations")
		("filename", po::value<string>(&cmd_line_args.filename)->required(), "data file");

	po::store(po::parse_command_line(argc, argv, desc), vm);

	if (vm.count("help")) {
		cout << desc << endl;
		exit(1);
	}

	po::notify(vm);
	return(cmd_line_args);
}

int main(int argc, char** argv) {
	vector<Point<double>> points;
	vector<Cluster<double>> clusters;
	vector<vector<double>> training_data;

	cmd_line_args_t cmd_line_args = parse_command_line(argc, argv);
	training_data = read_training_data<double>(cmd_line_args.filename, cmd_line_args.num_features);

	for_each(training_data.begin(), training_data.end(), [&points](vector<double> coordinates) {
		points.push_back(Point<double>(coordinates));
	});

	clusters = KMeans<double>(cmd_line_args.k, points, cmd_line_args.max_iterations);

	for_each(clusters.begin(), clusters.end(), [](Cluster<double> cluster) {
		cout << cluster << endl;
	});

	return(0);
}
