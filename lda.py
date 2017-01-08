from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import pandas as pd
import math


def loadIris():
	testSet = np.zeros(shape=(75, 4))
	data = pd.read_csv(filepath_or_buffer='iris.data',  header=None,  sep=',')
	data.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
	data.dropna(how="all", inplace=True)
	data.tail()
	feature_matrix = data.ix[:,0:4].values
	labels = data.ix[:, 4].values
	testSet[0:25, 0:4] = feature_matrix[25:50, 0:4]
	testSet[25:50, 0:4] = feature_matrix[50:75, 0:4]
	testSet[50:75, 0:4] = feature_matrix[125:150, 0:4]

	return feature_matrix, labels, testSet

def euclideanDistance(instance1, instance2):
	distance = 0
	for x in range(len(instance1)):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def mahalanobisDistance(x, mean, cov):
	diff = x - mean
	m_dist = np.dot(np.dot(diff.T, la.inv(cov)), diff)
	
	return m_dist

def minDistCl(feature_matrix, testSet):
	class1 = feature_matrix[0:25, 0:4]
	class2 = feature_matrix[50:75, 0:4]
	class3 = feature_matrix[100:125, 0:4]
	mean_vector1 = class1.mean(axis=0).T
	mean_vector2 = class2.mean(axis=0).T
	mean_vector3 = class3.mean(axis=0).T
	y_pred = []
	y_actual = []
	for i in range(25):
		y_actual.append("Iris-setosa")
	for i in range(25):
		y_actual.append("Iris-versicolor")
	for i in range(25):
		y_actual.append("Iris-virginica")


	for i in range(len(testSet)):
		e_dist1 = euclideanDistance(testSet[i], mean_vector1)
		e_dist2 = euclideanDistance(testSet[i], mean_vector2)
		e_dist3 = euclideanDistance(testSet[i], mean_vector3)
		if e_dist1 < e_dist2 and e_dist1 < e_dist3:	y_pred.append("Iris-setosa")
		elif e_dist2 < e_dist1 and e_dist2 < e_dist3: y_pred.append("Iris-versicolor")
		elif e_dist3 < e_dist2 and e_dist3 < e_dist1: y_pred.append("Iris-virginica")


	y_a = pd.Series(y_actual, name='Actual')
	y_p = pd.Series(y_pred, name='Predicted')
	df_confusion = pd.crosstab(y_a,y_p)
	print df_confusion
	

def linGausCl(feature_matrix, testSet):
	class1 = feature_matrix[0:25, 0:4]
	class2 = feature_matrix[50:75, 0:4]
	class3 = feature_matrix[100:125, 0:4]
	mean_vector1 = class1.mean(axis=0).T
	mean_vector2 = class2.mean(axis=0).T
	mean_vector3 = class3.mean(axis=0).T
	cov1 = np.cov(class1, rowvar=0)
	cov2 = np.cov(class2, rowvar=0)
	cov3 = np.cov(class3, rowvar=0) 
	cov = np.cov(feature_matrix, rowvar=0)
	y_pred = []
	y_actual = []

	S = (1/3) * cov1 + (1/3) * cov2 + (1/3) * cov3
	for i in range(25):
		y_actual.append("Iris-setosa")
	for i in range(25):
		y_actual.append("Iris-versicolor")
	for i in range(25):
		y_actual.append("Iris-virginica")
	for i in range(len(testSet)):
		m_dist1 = mahalanobisDistance(testSet[i], mean_vector1, S)		
		m_dist2 = mahalanobisDistance(testSet[i], mean_vector2, S)		
		m_dist3 = mahalanobisDistance(testSet[i], mean_vector3, S)		
		if m_dist1 < m_dist2 and m_dist1 < m_dist3: y_pred.append("Iris-setosa")
		elif m_dist2 < m_dist1 and m_dist2 < m_dist3: y_pred.append("Iris-versicolor")
		elif m_dist3 < m_dist2 and m_dist3 < m_dist1: y_pred.append("Iris-virginica")



	y_a = pd.Series(y_actual, name='Actual')
	y_p = pd.Series(y_pred, name='Predicted')
	df_confusion = pd.crosstab(y_a,y_p)
	print df_confusion
	

def quadGausCl(feature_matrix, testSet):
	class1 = feature_matrix[0:25, 0:4]
	class2 = feature_matrix[50:75, 0:4]
	class3 = feature_matrix[100:125, 0:4]
	mean_vector1 = class1.mean(axis=0).T
	mean_vector2 = class2.mean(axis=0).T
	mean_vector3 = class3.mean(axis=0).T
	y_pred = []
	y_actual = []
	for i in range(25):
		y_actual.append("Iris-setosa")
	for i in range(25):
		y_actual.append("Iris-versicolor")
	for i in range(25):
		y_actual.append("Iris-virginica")
	cov1 = np.cov(class1, rowvar=0)
	cov2 = np.cov(class2, rowvar=0)
	cov3 = np.cov(class3, rowvar=0)


	W1 = -(1/2) * la.inv(cov1)
	W2 = -(1/2) * la.inv(cov2)
	W3 = -(1/2) * la.inv(cov3)
	w1 = np.dot(la.inv(cov1), mean_vector1)
	w2 = np.dot(la.inv(cov2), mean_vector2)
	w3 = np.dot(la.inv(cov3), mean_vector3)
	w1_0 = -(1/2) * np.dot(np.dot(mean_vector1.T, la.inv(cov1)), mean_vector1)
	w2_0 = -(1/2) * np.dot(np.dot(mean_vector2.T, la.inv(cov2)), mean_vector2)
	w3_0 = -(1/2) * np.dot(np.dot(mean_vector3.T, la.inv(cov3)), mean_vector3)
	for i in range(len(testSet)):
		g1 = np.dot(np.dot(testSet[i].T, W1), testSet[i]) + np.dot(w1.T, testSet[i]) + w1_0
		g2 = np.dot(np.dot(testSet[i].T, W2), testSet[i]) + np.dot(w2.T, testSet[i]) + w2_0
		g3 = np.dot(np.dot(testSet[i].T, W3), testSet[i]) + np.dot(w3.T, testSet[i]) + w3_0
		if g1 > g2 and g1 > g3: y_pred.append("Iris-setosa")
		elif g2 > g1 and g2 > g3: y_pred.append("Iris-versicolor")
		elif g3 > g2 and g3 > g1: y_pred.append("Iris-virginica")
	y_a = pd.Series(y_actual, name='Actual')
	y_p = pd.Series(y_pred, name='Predicted')
	df_confusion = pd.crosstab(y_a,y_p)
	plot_confusion_matrix(df_confusion)

	print df_confusion
	

def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    plt.show()


def gausian(x, mean, cov):
	xm = x - mean
	p_x = 1/(math.pow(2*math.pi, 2) * math.sqrt(la.det(cov))) * math.exp((-1/2) * np.dot(np.dot(xm.T, la.inv(cov)), xm))
	return p_x
	  
def main():
	feature_matrix, labels, testSet = loadIris()
	minDistCl(feature_matrix, testSet)	
	linGausCl(feature_matrix, testSet)
	quadGausCl(feature_matrix, testSet)

if __name__ == "__main__":
	main()
