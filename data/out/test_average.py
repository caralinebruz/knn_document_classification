#!/usr/bin/python3

import logging
import math
from math import sqrt
import random
import string




a = [
	[100,	0.1,	0.0001],
	[102,	0.1, 	0.002],
	[103,	0.2,	0.0006],
	[104,	0.155,	0.008],
]

b = [a]


def average(seq):
	return sum(seq) / len(seq)



def update_centroids():


	new_centroids = []
	# for each cluster
	#	 there will be a new centroid

	for cluster in b:

		new_centroid = [0] * len(a[0])

		for item in range(len(a[0])):

			the_column_values = []
			print("col:%s" % item)

			for row in a:
				value = row[item]

				print("cell value: %s" % value)

				the_column_values.append(value)


			# once we've gone through all the row-values, 
			# take the average and append it to new new_centroid
			avg_value = average(the_column_values)
			print("the average of column %s is %s" % (item, avg_value))

			new_centroid[item] = avg_value

		new_centroids.append(new_centroid)


		for c in new_centroids:
			print(c)



if __name__ == '__main__':

	update_centroids()