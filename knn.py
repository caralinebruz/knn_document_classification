#!/usr/bin/python3
import os
import sys
import logging
import math
from math import sqrt
import random
import string

import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

import itertools

# for average per row
import numpy as np
import pandas as pd

import spacy
from spacy import displacy

# new for hw 2
import sys


## ******************************************
##
## USAGE:
## python3 knn.py data/input_tf_idf_labels_headers.csv data/unknown/unknown01.txt
##
## ******************************************


logging.basicConfig(
	format='%(asctime)s %(levelname)-8s %(message)s',
	level=logging.INFO,
	datefmt='%Y-%m-%d %H:%M:%S')
logger = logging

seed_tf_idf_matrix_file = "./data/input_tf_idf_labels_headers.csv"
outfile_path = "data/out/"
my_process_objects = []

labels = []
indices = []
label_index = {}
tf_idf_matrix = []
input_keywords_concepts = []


class Preprocessor():

	# easy way to increment an index of the class instance
	# https://stackoverflow.com/questions/1045344/how-do-you-create-an-incremental-id-in-a-python-class
	index = itertools.count()

	def __init__(self, file):
		self.file_basename = file
		# self.filename = "./" + infile_path + file
		self.filename = os.path.abspath(file)
		self.out_filename = "./" + outfile_path + self.file_basename
		self.lines = []
		self.stop_words = set(stopwords.words('english'))
		self.lemmatizer = WordNetLemmatizer()
		self.document = []
		self.document_text = ""
		self.keywords_concepts = []
		self.ngrams = []
		self.ngrams_frequency = {}
		self.index = next(Preprocessor.index)

	def read_file(self):
		# read the file into an array of lines
		logger.info("opening file %s ...", self.filename)

		with open(self.filename) as f:
			self.lines = f.readlines()

			logger.debug("line 0: %s", self.lines[0])

	def filter_stopwords_lemmatize(self):
		logger.info("removing stopwords ...")
		new_lines = []

		for line in self.lines:
			if line is not None:

				# split and tokenize
				old_sentence = word_tokenize(line)

				for word in old_sentence:

					# remove punctuation
					# https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
					exclude = set(string.punctuation)
					word = ''.join(ch for ch in word if ch not in exclude)

					# if its not empty 
					if word is not None and len(word) > 0:

						# remove stopwords
						if word not in self.stop_words:

							# lemmatize 
							# best guess here to treat anything ending in 's'
							#	as a noun, anything else gets verb treatment
							new_word = word
							if word.endswith('s'):
								new_word = self.lemmatizer.lemmatize(word)
							else:
								new_word = self.lemmatizer.lemmatize(word, "v")

							## not entirely sure if i should be lowercasing everything
							# new_word = new_word.lower()

							# and add it to the text document
							self.document.append(new_word)
							# logger.info("%s => %s" % (word,new_word))	

		self.document_text = ' '.join(self.document)

	def apply_ner(self):
		logger.info("applying NER ...")
		NER = spacy.load('en_core_web_sm')

		mytext = NER(self.document_text)

		logger.debug("Found the following entities:")
		for ent in mytext.ents:
			# print(ent.text, ent.start_char, ent.end_char, ent.label_)
			logger.debug("\t %s : %s" % (ent.text, ent.label_))
			this_ent = ent.text

			# if there is one or more spaces in the ENT
			if " " in this_ent:
				# then convert them to underscores in the document text
				new_ent = this_ent.replace(" ","_")

				# save the ENT for later matrix
				self.keywords_concepts.append(new_ent)

				# then also replace the original text document
				self.document_text = self.document_text.replace(this_ent, new_ent)

		# also update the tokenized array
		self.document = word_tokenize(self.document_text)


	# https://www.geeksforgeeks.org/python-bigrams-frequency-in-string/
	def _find_bi_grams(self, text):

		bigrams = zip(text, text[1:])
		for gram in bigrams:

			bigram_string = ' '.join(gram)
			self.ngrams.append(bigram_string)

	def _find_tri_grams(self, text):
		# this doesnt seem to be producing as meaningful result as the bigram :/

		trigrams = zip(text, text[1:], text[2:])
		for gram in trigrams:

			trigram_string = ' '.join(gram)
			self.ngrams.append(trigram_string)

	def sliding_window_merge(self):
		logger.info("using a sliding window to merge remaining phrases ...")

		# ****************************************************
		# BI-GRAMS VS TRI-GRAMS ::
		# 
		# 	I won't use trigrams bc frequencies arent as good
		#		but logic for it is here in this block
		#
		#
		# self.ngrams = []
		#
		# self._find_tri_grams(self.document)
		#
		# for ngram in self.ngrams:
		# 	frequency = self.document_text.count(ngram)
		#
		# 	self.ngrams_frequency['ngram'] = frequency
		# 	print("%s : %s "% (ngram, frequency))
		#
		# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

		logger.info("using bi-grams for this, there are more matches...")
		# ngram_dist = nltk.FreqDist(nltk.bigrams(self.document))
		# print(ngram_dist.most_common())

		# i will pick everything with freq > 1 for the merge

		self._find_bi_grams(self.document)
		# print(self.ngrams)

		# dedupe the ngrams
		self.ngrams = list(dict.fromkeys(self.ngrams))

		for ngram in self.ngrams:
			frequency = self.document_text.count(ngram)
		
			self.ngrams_frequency[ngram] = frequency
			#print("%s : %s "% (ngram, frequency))

			# if frequency > 1, merge
			if frequency > 1:
				new_ngram = ngram.replace(" ","_")

				# save the NGRAM for later matrix
				self.keywords_concepts.append(new_ngram)

				# then also replace the original text document
				self.document_text = self.document_text.replace(ngram, new_ngram)	

				# print("\t\t %s : %s "% (ngram, frequency))


	def cleanup(self):

		for i in range(len(self.keywords_concepts)):
			self.keywords_concepts[i] = self.keywords_concepts[i].replace("_"," ").lower()
			# print(self.keywords_concepts[i])

		self.ngrams_frequency = {k.replace("_"," ").lower() : v for k, v in self.ngrams_frequency.items()}
		# print(self.ngrams_frequency.items())

		self.document_text = self.document_text.replace("_"," ").lower()
		# print(self.document_text)

				
	def write_output(self):

		# WRITE THIS FILE WITHOUT ANY UNDERSCORES

		outfilename = self.filename + "_out"
		logger.info("Writing output file "  + outfilename);

		with open(outfilename, "w") as outfile:
			for word in self.document_text:

				outfile.write(word.lower())


def write_keywords_concepts_file(P):

	logger.info("Appending to concepts file ...")
	concepts_file = "./" + outfile_path + "concepts.txt"
	
	with open(concepts_file, "a") as f:

		lines = P.keywords_concepts
		for line in lines:
			# print(line)
			f.write(line)
			f.write("\n")



def parse_file_tfidf(seed_tf_idf_labelled_file):

	lines = []
	logger.info("opening file %s ...", seed_tf_idf_labelled_file)
	with open(seed_tf_idf_labelled_file, "r") as f:

		lines = f.readlines()


	# now that youre done reading, split out into tf_idf cevtors and the labels
	# get the keywords_concepts array
	logger.info("parsing header record for tf_idf file (aka the keywords_concepts ...")
	header = lines[0].rstrip('\n')
	header = header.split(',')


	# https://stackoverflow.com/questions/423379/using-global-variables-in-a-function
	global input_keywords_concepts 

	input_keywords_concepts = header[2:]
	#V print(input_keywords_concepts)

	for line in lines[1:]:

		line = line.rstrip('\n')
		line = line.split(',')


		index = line[0]
		label = line[1]
		data = line[2:]

		labels.append(label)
		indices.append(index)
		label_index[index] = label


		tf_idf_matrix.append(data)

		logger.info("parsing tf_idf data input row %s" % (len(tf_idf_matrix)))
		# print(label_index.items())

	return True



def v2_do_preprocessing(file):

	P = Preprocessor(file)

	# and add that object to the processed objects list
	my_process_objects.append(P)

	# read the file
	P.read_file()

	# 2 - remove stopwords, lemmatize, and tokenize
	# https://www.geeksforgeeks.org/python-lemmatization-with-nltk/
	P.filter_stopwords_lemmatize()

	# 3 - apply NER 
	# https://www.analyticsvidhya.com/blog/2021/06/nlp-application-named-entity-recognition-ner-in-python-with-spacy/#:~:text=Named%20Entity%20Recognition%20is%20the,%2C%20money%2C%20time%2C%20etc.
	P.apply_ner()

	# 4 - use sliding window approach to merge remaining phrases
	P.sliding_window_merge()

	# clean up my findings:
	# 	removes underscores from document text, lowercases
	#	removes underscores from frequency keys, lowercases
	P.cleanup()

	# 5 - at the end, write to out_file for each document for safety
	P.write_output()

	# also write the keywords concepts file
	write_keywords_concepts_file(P)

	return P



def v2_add_to_tf_idf_matrix(tf_idf_matrix, processed_new_object):

	# first we need to 'back into' the original IDF
	array = np.array(tf_idf_matrix)
	tf_idf_df = pd.DataFrame(
							data=array, 
							index=indices, 
							columns=input_keywords_concepts
							)

	#print(tf_idf_df)
	print(processed_new_object.document_text)


	# initialize vector for this new document
	# 	the length will be same as the existing matrix length
	new_document_tf_idf_vector = [0] * len(input_keywords_concepts)

	for w in range(len(input_keywords_concepts)):

		word = input_keywords_concepts[w]
		# logger.info(word)

		if word in processed_new_object.document_text:
			# count the frequency
			frequency = processed_new_object.document_text.count(word)
			logger.info("%s : %s" % (word, frequency))

			# next get the term frequency of this word at it's own document
			# word count of the current document
			this_document_wordcount = len(processed_new_object.document_text)
			# make the tf calculation 
			tf = float(frequency / this_document_wordcount)

			# save this value for now in the vector, add IDF next
			new_document_tf_idf_vector[w] = tf


			# IDF
			# for each document row in the tf_idf matrix
			# count the number of rows which have a value > 0
			# 			means we can back into IDF
			counter = 0
			for i in range(len(tf_idf_matrix)):

				# print(tf_idf_matrix[i][w])

				if float(tf_idf_matrix[i][w]) > 0:
					counter+=1
					# increment here means the document contains this keyword

			# at the end of the whole column in tf_idf matrix, we have now:
			# count of documents containing said keyword
			#	we can compute the IDF (!)

			# print("Num documents containing this keyword (incl. new one): %s" % counter)
			# if we want to include this one its 
			counter+=1 #otherwise comment that one out

			#IDF
			total_docs_including_new_one = len(tf_idf_matrix)
			idf = float(math.log(total_docs_including_new_one / counter))

			# print("IDF for keywd: %s : %s" % (word, idf))

			# now that we have IDF, we can multiply existing TF value
			new_document_tf_idf_vector[w] = float(new_document_tf_idf_vector[w] * idf)

			# print("final value after multiplying: %s" % new_document_tf_idf_vector[w])

	# lastly, after all, append this new document's tf_idf vector to the main matrix
	tf_idf_matrix.append(new_document_tf_idf_vector)
	return tf_idf_matrix


def dot_prod(vector_1, vector_2):
	return float(sum(float(x) * float(y) for x, y in zip(vector_1, vector_2)))

def mag(vec):
	return float(sqrt(dot_prod(vec, vec)))

def cosine_similarity(vector_1, vector_2):
	# define the cosine similarity between the two vectors
	cosine_similarity = dot_prod(vector_1,vector_2) / ( mag(vector_1) * mag(vector_2) + 0.000000001  ) 

	# print("cosine similarity: %s" % cosine_similarity)
	return cosine_similarity


def do_knn(new_tf_idf):
	#
	# PART 1 : get the distances and rank them
	#
	# find distance between the new object and all other vectors
	#			cosine similarity -> highest number is more similar
	# 			when cos(\theta) == 1 means they are the same 

	distances = {}
	new_document_vector = new_tf_idf[-1]

	# go through all of the vectors except the last one (which is itself)
	for i in range(len(new_tf_idf) - 1):

		vector = new_tf_idf[i]

		# get the cosine similarity
		cosine_sim_score = cosine_similarity(new_document_vector, vector)
		distances[i] = cosine_sim_score


	# reverse sorted list of the values of this dictionary
	sorted_keys = sorted(list(distances.values()), reverse=True)
	# print(sorted_keys)

	#
	# PART 2
	#
	# decide which group this document belongs in based off the rankings
	# i decided to use a weighted knn so that documents closer
	#	are ranked with more weight
	logger.info("Moving on to voting rounds...")

	# make a local copy of the distances hash
	this_round_distances = distances
	logger.info(this_round_distances)
	# print(label_index)

	# initalize empty voting hash
	votes = {}
	for label in labels:
		votes[label] = 0

	# initially we pick k=3 modes
	# i tried k=6 as well and got the same results, FWIW
	k = 3
	logger.info("The closest %d neighbors:" % k)
	for j in range(len(sorted_keys[0:k])):

		neighbor_distance = sorted_keys[j]
		if neighbor_distance == 0:
			logger.warning("Nearest neighbor is perpendicular to this vector")

		# print(neighbor_distance)

		# look up the label for this one:
		# takes the first value found forthe distance
		for index in list(this_round_distances):
			if this_round_distances[index] == neighbor_distance:

				# look up that item's label
				this_label = label_index[str(index)]
				logger.info("\tkey %s, distance: %s, label: %s" % (index, neighbor_distance, this_label))

				# add it's label (vote) to the dictionary
				votes[this_label] += 1

				# and pop it from the eligible candidates for this round of knn
				del this_round_distances[index]

	logger.info("After voting, results:")
	logger.info(votes)




if __name__ == '__main__':

	logger.info("starting ...");

	unknown_file = ""
	if not len(sys.argv) > 2:
		logger.error("Expected file argument. None provided")
	else:
		seed_tf_idf_labelled_file = sys.argv[1]
		logger.info("Using this seed file for tf_idf with labels: %s" % seed_tf_idf_labelled_file)

		unknown_file = sys.argv[2]
		logger.info("Using this unknown file for classification: %s" % unknown_file)



		# **************************************************************
		#
		# PART 1: 
		#	For this portion I just parse the TF-IDF matrix from input
		#		capture keywords_concepts and capture labels for each row
		#
		# **************************************************************

		# does preprocessing on the files
		logger.info("Parse the inputu tf_idf file to be useable....")
		parse_file_tfidf(seed_tf_idf_labelled_file)

		logger.info("Done parsing out ye ole tf-idf matrix")


		# **************************************************************
		#
		# PART 2
		#	Take the new file and create the tf-idf matrix
		#
		# **************************************************************

		# create a process object of the new file
		logger.info("parsing the unknown file ...")
		processed_new_object = v2_do_preprocessing(unknown_file)

		# add this new file to the existing tf-idf matrix
		logger.info("integrating this unknown vector with the input tf_idf")
		new_tf_idf = v2_add_to_tf_idf_matrix(tf_idf_matrix, processed_new_object)


		# **************************************************************
		#
		# PART 3
		#	Find KNN for new document - use cosine as in hw1
		#
		# **************************************************************

		neighbors = do_knn(new_tf_idf)























