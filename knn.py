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
## python3 knn.py unknown/unknown01.txt data/input_tf_idf_labels_headers.csv
##
## ******************************************


logging.basicConfig(
	format='%(asctime)s %(levelname)-8s %(message)s',
	level=logging.INFO,
	datefmt='%Y-%m-%d %H:%M:%S')
logger = logging

seed_tf_idf_matrix_file = "./data/input_tf_idf_labels_headers.csv"
# infile_path = "data/"
outfile_path = "data/out/"
my_process_objects = []

# files = []

# files = [
# 	"data/C1/article01.txt",
# 	"data/C1/article02.txt",
# 	"data/C1/article03.txt",
# 	"data/C1/article04.txt",
# 	"data/C1/article05.txt",
# 	"data/C1/article06.txt",
# 	"data/C1/article07.txt",
# 	"data/C1/article08.txt",
# 	"data/C4/article01.txt",
# 	"data/C4/article02.txt",
# 	"data/C4/article03.txt",
# 	"data/C4/article04.txt",
# 	"data/C4/article05.txt",
# 	"data/C4/article06.txt",
# 	"data/C4/article07.txt",
# 	"data/C4/article08.txt",
# 	"data/C7/article01.txt",
# 	"data/C7/article02.txt",
# 	"data/C7/article03.txt",
# 	"data/C7/article04.txt",
# 	"data/C7/article05.txt",
# 	"data/C7/article06.txt",
# 	"data/C7/article07.txt",
# 	"data/C7/article08.txt",
# ]



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



# # # collect keywords and terms across all the files
# # def generate_term_document_matrix():
# class DocuTermMatrix():

# 	def __init__(self):
# 		self.keywords_concepts = []
# 		self.matrix = []
# 		self.tf_idf_matrix = []
# 		self.total_documents = len(my_process_objects)
# 		self.docs_with_keyword = {}

# 	def consolidate_keywords_concepts(self):
# 		# read the file into an array of lines
# 		logger.info("Collecting all of the keywords concepts ...")
# 		for file_object in my_process_objects:

# 			for keyword in file_object.keywords_concepts:
# 				if keyword not in self.keywords_concepts:

# 					self.keywords_concepts.append(keyword.lower())


# 	def initialize_matrix(self):
# 		logger.info("initializing the zero matrix ...")

# 		# fill with 0s for the correct size matrix
# 		num_rows = len(my_process_objects)
# 		num_cols = len(self.keywords_concepts)

# 		# https://intellipaat.com/community/63426/how-to-create-a-zero-matrix-without-using-numpy
# 		self.matrix = [([0]*num_cols) for i in range(num_rows)]


# 	def fill_matrix(self):

# 		logger.info("Creating the document term matrix ...")
# 		i = 0
# 		for i in range(len(my_process_objects)):

# 			# print(i)
# 			# print(my_process_objects[i].index)
# 			# print(my_process_objects[i].filename)

# 			file_object = my_process_objects[i]

# 			# convert all the keys to lowercase for now
# 			the_files_ngrams =  {k.lower(): v for k, v in file_object.ngrams_frequency.items()}
# 			# print(file_object.ngrams_frequency)


# 			# iterate over the keywords_concepts list
# 			for j in range(len(self.keywords_concepts)):

# 				# if a keyword_concept is in the document_text of the document
# 				# count the number of times the substring appears
# 				if self.keywords_concepts[j] in file_object.document_text:

# 					# https://stackoverflow.com/questions/8899905/count-number-of-occurrences-of-a-substring-in-a-string
# 					frequency = file_object.document_text.count(self.keywords_concepts[j])
# 					self.matrix[i][j] = frequency

# 				# else:
# 				# 	print("%s not in document text" % self.keywords_concepts[j])
# 				# 	print(file_object.document_text)


# 	def _get_tf(self, document_object, row_index, col_index):
# 		# logger.debug("getting TF for a keyword in %s" % document_object.filename)

# 		# number of times the term occurs in the current document
# 		keywd_occurrence_this_document = self.matrix[row_index][col_index]

# 		# word count of the current document
# 		this_document_wordcount = len(document_object.document_text)

# 		# logger.info("number of words in %s : %d" % (document_object.filename, this_document_wordcount))

# 		# make the tf calculation 
# 		tf = float(keywd_occurrence_this_document / this_document_wordcount)

# 		return tf


# 	def _get_idf(self, keyword, row_index, col_index):
# 		# logger.info("getting IDF on keyword %s ... " % keyword)
# 		# math.log uses base e
# 		# test1 = math.log(20)
# 		# print(test1)

# 		num_documents_this_keyword = self.docs_with_keyword[keyword]
# 		idf = float(math.log(self.total_documents / num_documents_this_keyword))

# 		return idf

# 	def _get_num_documents_containing_keyword(self):

# 		for col_index in range(len(self.keywords_concepts)):

# 			keyword = self.keywords_concepts[col_index]
# 			counter = 0

# 			# for each row of the current column
# 			for row_index in range(len(my_process_objects)):

# 				# is the value in the matrix > 0 ?
# 				if self.matrix[row_index][col_index] > 0:
# 					counter += 1

# 			# logger.info("current keyword: %s num_documents %s" % (keyword, counter))

# 			# add it to the dictionary
# 			self.docs_with_keyword[keyword] = counter



# 	def create_tf_idf(self):
# 		# start with a copy of the document term matrix
# 		self.tf_idf_matrix = self.matrix

# 		self._get_num_documents_containing_keyword()

# 		# for column (keyword) in the matrix
# 		for col_index in range(len(self.keywords_concepts)):

# 			keyword = self.keywords_concepts[col_index]
# 			counter = 0

# 			# logger.info("current keyword: %s" % keyword)

# 			# for each row of the current column
# 			for row_index in range(len(my_process_objects)):

# 				document = my_process_objects[row_index]

# 				# logger.info("Column: %s | Row: %s" % (keyword, document.index))

# 				tf = self._get_tf(document, row_index, col_index)
# 				# logger.info("\tTF for document %s on current keyword is : %.8f" % (document.filename, tf))

# 				# now we make get the idf calculation
# 				idf = self._get_idf(keyword, row_index, col_index)
# 				# logger.info("\tIDF for this keyword is %.8f" % idf)

# 				# then combine them to get the TF-IDF weight
# 				tf_idf = float(tf * idf)
# 				# logger.info("\t\tFinal TF-IDF: %.8f" % tf_idf)

# 				# next, populate the new cell with the final valye
# 				self.tf_idf_matrix[row_index][col_index] = tf_idf



# # preprocess the raw data
# def v1_do_preprocessing():

# 	#for file in files[0:6]:
# 	for file in files:

# 		P = Preprocessor(file)

# 		# and add that object to the processed objects list
# 		my_process_objects.append(P)

# 		# read the file
# 		P.read_file()

# 		# 2 - remove stopwords, lemmatize, and tokenize
# 		# https://www.geeksforgeeks.org/python-lemmatization-with-nltk/
# 		P.filter_stopwords_lemmatize()

# 		# 3 - apply NER 
# 		# https://www.analyticsvidhya.com/blog/2021/06/nlp-application-named-entity-recognition-ner-in-python-with-spacy/#:~:text=Named%20Entity%20Recognition%20is%20the,%2C%20money%2C%20time%2C%20etc.
# 		P.apply_ner()

# 		# 4 - use sliding window approach to merge remaining phrases
# 		P.sliding_window_merge()

# 		# clean up my findings:
# 		# 	removes underscores from document text, lowercases
# 		#	removes underscores from frequency keys, lowercases
# 		P.cleanup()

# 		# 5 - at the end, write to out_file for each document for safety
# 		P.write_output()

# 		# also write the keywords concepts file
# 		write_keywords_concepts_file(P)

# 	return my_process_objects


# def v1_generate_document_term_matrix():

# 	M = DocuTermMatrix()

# 	# firt, consolidate and dedupe all keywords across the files
# 	M.consolidate_keywords_concepts()
# 	# print(M.keywords_concepts)

# 	# second, create the matrix
# 	M.initialize_matrix()
# 	M.fill_matrix()

# 	print("\n~~~~ Moving on to TF-IDF section ~~~~\n")
# 	M.create_tf_idf()

# 	print("\nTF IDF HERE:\n")

# 	tf_idf_input_file = "./" + outfile_path + "input_tfidf_labels.txt"
	
# 	with open(tf_idf_input_file, "w") as f:
# 		f.write(str(M.keywords_concepts))
# 		f.write("\n")
# 		for row in M.tf_idf_matrix:
# 			f.write(str(row))
# 			f.write("\n")


# 	# print(M.keywords_concepts)
# 	# print(M.tf_idf_matrix)

# 	return M


labels = []
indices = []
label_index = {}
tf_idf_matrix = []
keywords_concepts = []


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


	keywords_concepts = header[2:]
	print(keywords_concepts)

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




def sort_topics(keywords_concepts, folder_aggregate_vector):
	# Overall processing 
	zipped_aggregate = list(zip(keywords_concepts, folder_aggregate_vector))

	# dedupe idk why there are duplicates
	zipped_aggregate = set(zipped_aggregate)

	# https://www.geeksforgeeks.org/python-ways-to-sort-a-zipped-list-by-values/
	# Using sorted and lambda
	sorted_topics = sorted(zipped_aggregate, key = lambda x: x[1], reverse=True)

	for x in sorted_topics:
		print(x)

	return sorted_topics


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

	print(tf_idf_matrix)

	# first we need to 'back into' the original IDF
	array = np.array(tf_idf_matrix)
	tf_idf_df = pd.DataFrame(
							data=array, 
							index=indices, 
							columns=keywords_concepts
							)

	print(tf_idf_df)

	print(processed_new_object.document_text)


	# matrix_object.keywords_concepts is current feature list
	# processed_new_object.keywords)c

	# initialize vector for this new document
	# 	the length will be same as the existing matrix length
	new_document_tf_idf_vector = [0] * len(keywords_concepts)

	for w in range(len(keywords_concepts)):

		word = keywords_concepts[w]
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

				if tf_idf_matrix[i][w] > 0:
					counter+=1
					# increment here means the document contains this keyword

			# at the end of the whole column in tf_idf matrix, we have now:
			# count of documents containing said keyword
			#	we can compute the IDF (!)

			# print("Num documents containing this keyword (incl. new one): %s" % counter)
			# if we want to include this one its 
			counter+=1 #otherwise comment that one out

			#IDF
			total_docs_including_new_one = total_documents + 1
			idf = float(math.log(total_docs_including_new_one / counter))

			# print("IDF for keywd: %s : %s" % (word, idf))

			# now that we have IDF, we can multiply existing TF value
			new_document_tf_idf_vector[w] = float(new_document_tf_idf_vector[w] * idf)

			# print("final value after multiplying: %s" % new_document_tf_idf_vector[w])

	# lastly, after all, append this new document's tf_idf vector to the main matrix
	tf_idf_matrix.append(new_document_tf_idf_vector)
	return matrix_object


def dot_prod(vector_1, vector_2):
	return float(sum(x * y for x, y in zip(vector_1, vector_2)))

def mag(vec):
	return float(sqrt(dot_prod(vec, vec)))

def cosine_similarity(vector_1, vector_2):
	# define the cosine similarity between the two vectors
	cosine_similarity = dot_prod(vector_1,vector_2) / ( mag(vector_1) * mag(vector_2) + 0.000000001  ) 

	# print("cosine similarity: %s" % cosine_similarity)
	return cosine_similarity


def do_knn(matrix_object):
	# find distance between the new object and all other vectors
	#			cosine similarity -> highest number is more similar
	# 			when cos(\theta) == 1 means they are the same 

	distances = {}
	new_document_vector = matrix_object.tf_idf_matrix[-1]

	# go through all of the vectors except the last one (which is itself)
	for i in range(len(matrix_object.tf_idf_matrix) - 1):

		vector = matrix_object.tf_idf_matrix[i]

		# get the cosine similarity
		cosine_sim_score = cosine_similarity(new_document_vector, vector)
		distances[i] = cosine_sim_score


	# reverse sorted list of the values of this dictionary
	sorted_keys = sorted(list(distances_dict.values()), reverse=True)
	print(sorted_keys)






if __name__ == '__main__':

	logger.info("starting ...");

	unknown_file = ""
	if not len(sys.argv) > 2:
		logger.error("Expected file argument. None provided")
	else:
		unknown_file = sys.argv[1]
		logger.info("Using this unknown file for classification: %s" % unknown_file)

		seed_tf_idf_labelled_file = sys.argv[2]
		logger.info("Using this seed file for tf_idf with labels: %s" % seed_tf_idf_labelled_file)




		# # arg parse the file I am supposed to be using
		# unknown_file = sys.argv[1]
		# logger.info("Using this unknown file for classification: %s" % unknown_file)

		# with open(sys.argv[1], 'r') as f:
		# 	contents = f.read()	
		# 	print contents


		# **************************************************************
		#
		# PART 1: 
		#	For this portion I just recreate the TF-IDF matrix from HW 1
		#		I don't do anything new here
		#
		# **************************************************************

		# does preprocessing on the files
		# returns a list of preprocessed file objects for each file
		# logger.info("First: Do preprocessing on the files")
		# processed_objects = v1_do_preprocessing()

		# # then give it the matrix class here
		# logger.info("Next: Generating Document Term Matrix")
		# matrix_object = v1_generate_document_term_matrix()


		logger.info("Parse the inputu tf_idf file to be useable....")
		parse_file_tfidf(seed_tf_idf_labelled_file)

		logger.info("Done parsing out the ole tf-idf matrix")


		# **************************************************************
		#
		# PART 2
		#	Take the new file and create the tf-idf matrix
		#
		# **************************************************************


		logger.info("parsing the unknown file ...")
		# create a process object of the new file
		processed_new_object = v2_do_preprocessing(unknown_file)

		logger.info("integrating this unknown vector with the input tf_idf")



		# add this new file to the existing tf-idf matrix
		new_matrix_object = v2_add_to_tf_idf_matrix(tf_idf_matrix, processed_new_object)


		# **************************************************************
		#
		# PART 3
		#	Find KNN for new document - use cosine as in hw1
		#
		# **************************************************************

		neighbors = do_knn(new_matrix_object)






















