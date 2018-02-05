#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 18:51:15 2018

@author: brihat
"""

#import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
#import numpy as np

class MovieReviews:
    
    def readText(filename):
        DF = pd.read_csv(filename, sep = "\n", header = None)
        DF.columns = ['Reviews']
        return DF
    
 
    def splitDataframe(pandasDataFrame):
        train, test = train_test_split(pandasDataFrame, test_size = 0.15)
        return train, test
    
    def mergeDataFrame(pandasDataFrame1, pandasDataFrame2, pos, neg):
        DF1 = pandasDataFrame1.assign(label = pos)
        DF2 = pandasDataFrame2.assign(label = neg)
        df_new = pd.concat([DF1, DF2])
        return df_new
    
    def usePipeline(pandasDataFrame):
        stop = set(stopwords.words('english'))
        text_clf = Pipeline([('vect', TfidfVectorizer(ngram_range = (1, 2), min_df = 2, stop_words = stop)),
                              ('clf', SGDClassifier()),])

        return text_clf
       
        
    def fitTrainData(text_clf, pandasDataFrame):
        text_clf.fit(pandasDataFrame.Reviews, pandasDataFrame.label)
        return text_clf
    
    def predictionWithAccuracy(text_clf, pandasDataFrame):
        predicted = text_clf.predict(pandasDataFrame.Reviews)
        #accuracy = np.mean(predicted == pandasDataFrame.label)
        return predicted
    
    def getMetrics(gs_clf, pandasDataFrame):
        target_names = ['PR', 'NR']
        predictionList = gs_clf.predict(pandasDataFrame.Reviews)
        result = metrics.classification_report(pandasDataFrame.label, predictionList, target_names = target_names)
        return result, predictionList
        
    def getConfusionMatrix(pandasDataFrame, predictionList):
        conf_matrix = metrics.confusion_matrix(pandasDataFrame.label, predictionList)
        return conf_matrix
        
    def useGridSearch(text_clf, pandasDataFrame):
        C = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        param_grid = [{'clf__alpha': C, 'clf__loss': ['hinge', 'squared_hinge', 'log'], 'clf__max_iter': [100]}]
        gs_clf = GridSearchCV(text_clf, param_grid = param_grid, cv = 10, n_jobs = -1)
        gs_clf = gs_clf.fit(pandasDataFrame.Reviews, pandasDataFrame.label)
        best_score = gs_clf.best_score_
        best_param = gs_clf.best_params_
        return gs_clf, best_score, best_param
    
    def getROCCurve(gs_clf, pandasDataFrame):
        score_roc = gs_clf.decision_function(pandasDataFrame.Reviews)
        fpr, tpr, thresholds = metrics.roc_curve(pandasDataFrame.label, score_roc)
        roc_auc = metrics.auc(fpr, tpr)
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.plot([0,1], [0, 1], 'r--')
        plt.show()
        
if __name__ == '__main__':
    RD = MovieReviews
    posReviewDF = RD.readText("rt-polarity.pos")
    negReviewDF = RD.readText("rt-polarity.neg")
    #print (negReviewDF) 
    #posReviewDF1 = RD.removeStopwords(posReviewDF)
    #negReviewDF1 = RD.removeStopwords(negReviewDF)
    #print(posReviewDF1)
    posReviewDFTrain, posReviewDFTest = RD.splitDataframe(posReviewDF)
    negReviewDFTrain, negReviewDFTest = RD.splitDataframe(negReviewDF)
    #print(negReviewDFTrain)  
    reviewTrainDF = RD.mergeDataFrame(posReviewDFTrain, negReviewDFTrain, 1, 0)
    reviewTestDF = RD.mergeDataFrame(posReviewDFTest, negReviewDFTest, 1, 0)
    print(negReviewDFTest)
    print(posReviewDFTest)
    print(posReviewDFTest.shape, negReviewDFTest.shape)
    print(reviewTestDF.shape)
    text_clf = RD.usePipeline(reviewTrainDF)    
    gs_clf, best_score, best_param = RD.useGridSearch(text_clf, reviewTrainDF)
    print("best_score: ")
    print(best_score)
    print("Best Parameter: ")
    print(best_param)
    MetricsF1Scores, predictionList = RD.getMetrics(gs_clf, reviewTestDF)
    print("Metrics: ")
    print(MetricsF1Scores)
    conf_matrix = RD.getConfusionMatrix(reviewTestDF, predictionList)
    print("Confusion Matrix :")
    print(conf_matrix)
    
    RD.getROCCurve(gs_clf, reviewTestDF)
    
    