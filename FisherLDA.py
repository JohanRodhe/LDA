from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import pandas as pd


def loadIris():
	data = pd.read_csv(filepath_or_buffer='iris.data',  header=None,  sep=',')

	data.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
	data.dropna(how="all", inplace=True)

	data.tail()
	feature_matrix = data.ix[0:99,0:4].values
	labels = data.ix[0:99, 4].values
	return feature_matrix, labels



def fisherLDA(feature_matrix):
	class1 = feature_matrix[0:50, 0:4]
	class2 = feature_matrix[50:100, 0:4]
	mean_vector1 = class1.mean(axis=0)[np.newaxis].T
	mean_vector2 = class2.mean(axis=0)[np.newaxis].T
	cov_matrix1 = np.cov(class1, rowvar=0)
	cov_matrix2 = np.cov(class2, rowvar=0)
	S_W = cov_matrix1 + cov_matrix2
	S_B = np.dot((mean_vector1 - mean_vector2), (mean_vector1 - mean_vector2).T)
	A = np.dot(la.inv(S_W), S_B)
	eig_values, eig_vectors = la.eig(A)
	eig_pairs = [(np.abs(eig_values[i]), eig_vectors[:,i]) for i in range(len(eig_values))]
	eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
	eig_vec1 = np.array(eig_pairs[0][1])
	return eig_vec1, mean_vector1, mean_vector2

def main():
	feature_matrix, labels = loadIris()
	w, mean_vector1, mean_vector2 = fisherLDA(feature_matrix)
	Y1 = np.dot(np.array(w), np.array(feature_matrix[0:50]).T)
	Y2 = np.dot(np.array(w), np.array(feature_matrix[50:100]).T)
	w0 = 0
	error1 = 0
	error2 = 0
	for i in range(len(Y1)):
		if Y1[i] < -w0: error1 += 1
		if Y2[i] > -w0: error2 += 1
	print error1, error2
	plt.plot(Y1, [0]*  Y1.shape[0], 'ob')
	plt.plot(Y2, [0]*  Y2.shape[0], 'og')
	plt.show()

if __name__ == "__main__":
	main()
