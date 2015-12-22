#!flask/bin/python

from __future__ import division
from flask import Flask, jsonify, abort, request, make_response, url_for
import json
import pickle
import base64
import numpy
import math
import scipy
from copy import deepcopy
from sklearn.cross_decomposition import PLSCanonical, PLSRegression, CCA
from sklearn import linear_model
from numpy  import array, shape, where, in1d
import ast
import threading
import Queue
import time
import random
from random import randrange
import sklearn
from sklearn import cross_validation
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import confusion_matrix
import cStringIO
from numpy import random
import scipy
from scipy.stats import chisquare
from copy import deepcopy
import operator 
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt
from operator import itemgetter
#from PIL import Image ## Hide for production
app = Flask(__name__, static_url_path = "")

"""
    JSON Parser for interlabtest
"""
def getJsonContents (jsonInput):
    try:
        dataset = jsonInput["dataset"]

        # type, number_of_variables, predictionFeature,predictedFeature
        parameters = jsonInput["parameters"]
        type = parameters.get("type", None)
        number_of_variables = parameters.get("variables", None)
        predictionFeature = parameters.get("predictionFeature", None)
        predictedFeature = parameters.get("predictedFeature", None)

        dataEntry = dataset.get("dataEntry", None)
        variables = dataEntry[0]["values"].keys() 

        real = [] 
        predicted = []

        for i in range(len(dataEntry)):
            for j in variables:
                temp = dataEntry[i]["values"].get(j)

                #if isinstance (temp, float):
                #    temp = round(temp, 2)

                if j == predictionFeature: 
                    real.append(temp)
                elif j == predictedFeature:
                    predicted.append(temp)
        
    except(ValueError, KeyError, TypeError):
        print "Error: Please check JSON syntax... \n"

    return real, predicted, type, number_of_variables, predictionFeature, predictedFeature


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    #tick_marks = np.arange(len(iris.target_names))
    #plt.xticks(tick_marks, iris.target_names, rotation=45)
    #plt.yticks(tick_marks, iris.target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def stats_regression(Y, predY, num_predictors):

    meanY4r2 = numpy.mean(Y)
    meanYpred4r2 = numpy.mean(predY)

    RMSD_tem1 = 0
    for i in range (len(Y)):   
        RMSD_tem1 +=  numpy.power ((predY[i] - Y[i]), 2)

    RMSD = math.sqrt( RMSD_tem1/len(Y) )

    SSXX = 0
    SSYY = 0
    SSXY = 0
    for i in range (len(Y)):
        SSXX += numpy.power ((Y[i] - meanY4r2), 2)
        SSYY += numpy.power ((predY[i] - meanYpred4r2), 2) 
        SSXY += (Y[i] - meanY4r2)*(predY[i] - meanYpred4r2)
    
    if SSXX ==0 or SSYY ==0:
        R2wolfram = 0
    else:
        R2wolfram = numpy.power(SSXY, 2)/(SSXX*SSYY)

    R2adjusted = 1 -((1-R2wolfram)*((len(Y)-1)/(len(Y)-num_predictors-1)))

    RSS = 0 # residual sum of sq
    for i in range (len(Y)):
        RSS +=  numpy.power ((Y[i] - predY[i]), 2)

    SSR = 0 # sum sq regression
    for i in range (len(Y)):
        SSR +=  numpy.power ((Y[i] - meanYpred4r2), 2)

    StdError = numpy.sqrt(abs(RSS/(len(Y)-num_predictors-1)))
    Fvalue = (SSR/num_predictors)/(RSS/(len(Y)-num_predictors-1))

    #add errors from sklearn

    return R2wolfram, R2adjusted, RMSD, Fvalue, StdError


def stats_classification(Y, predY):
    Accuracy = sklearn.metrics.accuracy_score(Y, predY) #, normalize=True, sample_weight=None) #pos label 1 deleted
    Precision = sklearn.metrics.precision_score(Y, predY)#, labels=None, average='binary', sample_weight=None)
    Recall = sklearn.metrics.recall_score(Y, predY)#, labels=None, average='binary', sample_weight=None)
    F1_score = sklearn.metrics.f1_score(Y, predY)#, labels=None, average='binary', sample_weight=None)
    Jacc = sklearn.metrics.jaccard_similarity_score(Y, predY) 
    #TP.TN/FP/FN - > MCC etc ?

    """
    # Check if Y is binary and convert Y, predY to True/False
    AUC_decision = []
    for i in range (len(Y)):
        if Y[i] == predY[i]:
            AUC_decision.append(True)
        else: 
            AUC_decision.append(False)
    AUC = sklearn.metrics.roc_auc_score(Y, AUC_decision),  average='macro', sample_weight=None)
    """

    cm = confusion_matrix(Y, predY)
    numpy.set_printoptions(precision=2)

    myFIGA = plt.figure()
    plot_confusion_matrix(cm, title='Confusion matrix')

    sio = cStringIO.StringIO()
    myFIGA.savefig(sio, dpi=300, format='png') 
    saveas = pickle.dumps(sio.getvalue())
    cm_encoded = base64.b64encode(saveas)

    #plt.show()
    plt.close()

    ## can plot normalized cm as well
    #cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
    #plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    return Accuracy, Precision, Recall, F1_score, Jacc, cm_encoded

"""
    [[],[]] Transposed Matrix to dictionary
"""
def mat2dic(matrix):
    myDict = {}
    for i in range (len (matrix)):
        myDict["Row_" + str(i+1)] = [matrix[i][0], matrix[i][1]]
    return myDict

@app.route('/pws/validation', methods = ['POST'])
def create_task_interlabtest():

    if not request.json:
        abort(400)

    real, predicted, type, number_of_variables, predictionFeature, predictedFeature = getJsonContents(request.json)
    #print real,"\n", predicted,"\n",  type,"\n",  number_of_variables,"\n",  predictionFeature, "\n", predictedFeature

    full_table = [real, predicted]
    full_table_transposed = map(list, zip(*full_table)) 
    #print full_table,full_table_transposed
    full_table_dict = mat2dic(full_table_transposed)
    #print full_table_dict

    if type == "REGRESSION":
        R2wolfram, R2adjusted, RMSD, Fvalue, StdError = stats_regression(real, predicted, number_of_variables)
        task = {
        "singleCalculations": {"Algorithm Type": type, 
                               "Number of predictor variables": number_of_variables,
                               "R^2" : R2wolfram,
                               "R^2 Adjusted (if applicable)" : R2adjusted,
                               "RMSD" : RMSD,
                               "F-Value" : Fvalue,
                               "StdError" : StdError
                              },
        "arrayCalculations": {"All Data":
                               {"colNames": ["Real", "Predicted"],
                                "values": full_table_dict
                               }
                             },
        "figures": {
                   }
        }
    elif type == "CLASSIFICATION":
        Accuracy, Precision, Recall, F1_score, Jaccard, cm_encoded = stats_classification(real, predicted)
        task = {
        "singleCalculations": {"Algorithm Type": type, 
                               "Number of predictor variables": number_of_variables,
                               "R^2" : Accuracy,
                               "Precision" : Precision,
                               "Recall" : Recall,
                               "F1_score" : F1_score,
                               "Jaccard" : Jaccard
                              },
        "arrayCalculations": {"All Data":
                               {"colNames": ["Real", "Predicted"],
                                "values": full_table_dict
                               }
                             },
        "figures": {"Confusion Matrix" : cm_encoded
                   }
        }

    jsonOutput = jsonify( task )

    return jsonOutput, 201 

if __name__ == '__main__': 
    app.run(host="0.0.0.0", port = 5000, debug = True)

# curl -i -H "Content-Type: application/json" -X POST -d @C:/Python27/Flask-0.10.1/python-api/val.json http://localhost:5000/pws/validation
# C:\Python27\Flask-0.10.1\python-api 
# C:/Python27/python valid_service.py