/**
 * @file Point.h
 * @brief This file implements the Point class.
 * @author Nawab Ali
 */

#ifndef POINT_H
#define POINT_H

#include<vector>
#include<iostream>

using namespace std;

/**
 * @class Point
 * @brief This class implements a point in a n-dimensional space.
 */
template <class T>
class Point {
	public:
		/**
		 * @brief Constructor for the Point class.
		 * @param dimensions Number of dimensions in space.
		 * @return None Constructor does not return.
		 */
		Point(int dimensions=0);

		/**
		 * @brief Constructor for the Point class.
		 * @param coordinates Coordinates of the point in n-dimensional space.
		 * @return None Constructor does not return.
		 */
		Point(vector<T> coordinates);

		/**
		 * @brief Copy Constructor for the Point class.
		 * @param point Point object.
		 * @return None Constructor does not return.
		 */
		Point(const Point<T>& point);

		/**
		 * @brief Overload the assignment operator for the Point class.
		 * @param point Point object.
		 * @return Reference to a Point object.
		 */
		Point<T>& operator=(const Point<T>& point);

		/**
		 * @brief Overload the == operator for the Point class.
		 * @param point Point object.
		 * @return True or False.
		 */
		bool operator==(const Point<T>& point) const;

		/**
		 * @brief Overload the << operator for the Point class.
		 * @param ostream ostream object.
		 * @param point Point object.
		 * @return Reference to ostream object.
		 */
		friend ostream& operator<<(ostream& out, const Point<T>& point) {
			for_each(point.coordinates.begin(), point.coordinates.end(),
							[&out](const T e) {
				out << e << " ";
			});

			return(out);
		}

		/**
		 * @brief Getter method for number of dimensions.
		 * @param void
		 * @return Number of dimensions in space.
		 */
		int get_dimensions(void) const;

		/**
		 * @brief Getter method for coordinates of point.
		 * @param void
		 * @return Coordinates of point in n-dimensional space.
		 */
		vector<T> get_coordinates(void) const;

		/**
		 * @brief Setter method for coordinates of point.
		 * @param coordinates Coordinates of point in n-dimensional space.
		 * @return void
		 */
		void set_coordinates(const vector<T> coordinates);

	private:
		vector<T> coordinates;
};

template <class T>
Point<T>::Point(int dimensions) {
	coordinates = vector<T>(dimensions);
}

template <class T>
Point<T>::Point(vector<T> coordinates): coordinates(coordinates) {}

// Copy Constructor
template <class T>
Point<T>::Point(const Point<T>& point): Point(point.coordinates) {}

// Overload assignment operator
template <class T>
Point<T>& Point<T>::operator=(const Point<T>& point) {
 	coordinates = point.coordinates;

	return(*this);
}

// Overload == operator
template <class T>
bool Point<T>::operator==(const Point<T>& point) const {
	return(coordinates == point.coordinates);
}

template <class T>
int Point<T>::get_dimensions(void) const {
	return(coordinates.size());
}

template <class T>
vector<T> Point<T>::get_coordinates(void) const {
	return(coordinates);
}

template <class T>
void Point<T>::set_coordinates(const vector<T> coordinates) {
	this->coordinates = coordinates;
}

#endif //POINT_H
