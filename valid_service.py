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
import io
from io import BytesIO
#matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt
from operator import itemgetter

#from PIL import Image ## Hide for production

app = Flask(__name__, static_url_path = "")

"""
    JSON Parser for Validation
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
    #print len(real), len(predicted)
    return real, predicted, type, number_of_variables, predictionFeature, predictedFeature

"""
    Matplotlib default Confusion Matrix
"""
def plot_confusion_matrix(cm, labels, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    plt.xlabel('True label')
    plt.ylabel('Predicted label')
	
    plt.tight_layout()

"""
    QQ plot of Real Quantiles vs Predicted Quantiles
"""
def qq_plot(real, predicted):
    rp = zip (real, predicted)
    rp.sort()
    real_sorted = [r for r,p in rp]
    predicted_sorted = [p for r,p in rp]

    real_sorted = 1.0*numpy.array(real_sorted)
    predicted_sorted = 1.0*numpy.array(predicted_sorted)

    # Each of Real/Predicted with Theoretical Quantiles
    """
    myFIGA = plt.figure()
    ax = myFIGA.add_subplot(111)
    scipy.stats.probplot(real_sorted, dist="norm", plot=ax)
    #plt.show()
    plt.close()
	
    myFIGA = plt.figure()
    ax = myFIGA.add_subplot(111)
    scipy.stats.probplot(predicted_sorted, dist="norm", plot=ax)
    #plt.show()
    plt.close()
    """

    myFIGA = plt.figure()

    real_intervals = [(x-min(real_sorted))*100/(max(real_sorted)-min(real_sorted)) for x in real_sorted]
    #predicted_intervals = [(x-min(predicted_sorted))*100/(max(predicted_sorted)-min(predicted_sorted)) for x in predicted_sorted]
    #print real_intervals, "\n", predicted_intervals, "\n"

    real_percentile = numpy.percentile(real_sorted, real_intervals)
    #predicted_percentile = numpy.percentile(predicted_sorted, predicted_intervals) ####
    predicted_percentile = numpy.percentile(predicted_sorted, real_intervals) ####
    #print real_percentile, "\n", predicted_percentile
	
    plt.plot(real_percentile, predicted_percentile, 'ro', c="red")
    straight, = plt.plot(real_percentile, real_percentile, 'r', c="green", label = "Quantile Identity Line")

    #adjustment1 = abs(max(real_percentile) - min (real_percentile))*0.05 # +/- 5%
    #adjustment2 = abs(max(predicted_percentile) - min (predicted_percentile))*0.05 # +/- 5%
    #plt.xlim([round(min(real_percentile),2) - adjustment1, round(max(real_percentile),2) + adjustment1,])
    #plt.ylim([round(min(predicted_percentile),2) - adjustment2, round(max(predicted_percentile),2) + adjustment2,])

    plt.xlabel("Quantiles for Real Values")
    plt.ylabel("Quantiles for Predicted  Values")
	
    plt.title('QQ Plot')
    myLegend = plt.legend(handles = [straight], loc=2, fontsize = 'small')
	
    plt.tight_layout()
    plt.show() ## HIDE show on production

    ##sio = cStringIO.StringIO()
    #sio = BytesIO()
    #myFIGA.savefig(sio, dpi=300, format='png') # (sio, dpi=300, format='png', bbox_extra_artists=(myLegend,), bbox_inches='tight')
    #saveas = pickle.dumps(sio.getvalue())
    #fig_encoded = base64.b64encode(saveas)

    figfile = BytesIO()
    myFIGA.savefig(figfile, dpi=300, format='png')
    figfile.seek(0)  # rewind to beginning of file
    fig_encoded = base64.b64encode(figfile.getvalue())

    plt.close()
    return fig_encoded

"""
    Get % Confidence Interval for predictions
"""
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*numpy.array(data)
    n = len(a)
    m, se = numpy.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
    return m, h, -h

"""
    Plot Real vs. predicted values
"""
def plot_norm (real, predicted):
    rp = zip (real, predicted)
    rp.sort()
    real_sorted = [r for r,p in rp]
    predicted_sorted = [p for r,p in rp]
    #print real_sorted, "\n", predicted_sorted
    

    myFIGA = plt.figure()
	
    m, m_plus, m_minus = mean_confidence_interval(real_sorted)
    plus = [x + m_plus for x in real_sorted]
    minus = [x + m_minus for x in real_sorted]
	
    plt.plot(real_sorted, predicted_sorted, 'ro', c="red")
    straight, = plt.plot(real_sorted, real_sorted, 'r', c="green", label = "Identity Line (Real = Predicted)")
    dashed, = plt.plot(real_sorted, plus, 'r--', c="green", label = "95% Confidence Level")
    plt.plot(real_sorted, minus, 'r--', c="green")
	
    #adjustment = abs(max(real_sorted) - min (real_sorted))*0.05 # +/- 5%
    #adjustment2 = abs(max(predicted_sorted) - min (predicted_sorted))*0.05 # +/- 5%
    #plt.xlim([round(min(real_sorted),2) - adjustment, round(max(real_sorted),2) + adjustment,])
    #plt.ylim([round(min(real_sorted)) - adjustment, round(max(real_sorted)) + adjustment,])
    #plt.ylim([round(min(predicted_sorted),2) - adjustment2, round(max(predicted_sorted),2) + adjustment2,])

    plt.xlabel("Real Values")
    plt.ylabel("Predicted Values")
	
    plt.title('Real vs Predicted Values')
    myLegend = plt.legend(handles = [straight, dashed], loc=2, fontsize = 'small')
	
    plt.tight_layout()
    plt.show() ## HIDE show on production

    ###sio = cStringIO.StringIO()
    #sio = BytesIO()
    #myFIGA.savefig(sio, dpi=300, format='png') # myFIGA1a.savefig(sio, dpi=300, format='png', bbox_extra_artists=(myLegend,), bbox_inches='tight')
    #saveas = pickle.dumps(sio.getvalue())
    #fig_encoded = base64.b64encode(saveas)

    figfile = BytesIO()
    myFIGA.savefig(figfile, dpi=300, format='png')
    figfile.seek(0)  # rewind to beginning of file
    fig_encoded = base64.b64encode(figfile.getvalue())

    plt.close()
    return fig_encoded

"""
    Get Regression Stats
"""
def stats_regression(Y, predY, num_predictors):
    fig1 = plot_norm(Y, predY)
    fig2 = qq_plot(Y, predY) ##
	
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
    
    ## R2 by Wolfram
    if SSXX ==0 or SSYY ==0:
        R2wolfram = 0
    else:
        R2wolfram = numpy.power(SSXY, 2)/(SSXX*SSYY)

    #slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Y, predY)
    #print R2wolfram, sklearn.metrics.r2_score(Y, predY), numpy.power(r_value,2)
	
    R2wolfram = sklearn.metrics.r2_score(Y, predY), numpy.power(r_value,2)

    if len(Y) == num_predictors+1:
        R2adjusted = 0
    else:
        R2adjusted = 1 -((1-R2wolfram)*((len(Y)-1)/(len(Y)-num_predictors-1)))

    RSS = 0 # residual sum of sq
    for i in range (len(Y)):
        RSS +=  numpy.power ((Y[i] - predY[i]), 2)

    SSR = 0 # sum sq regression
    for i in range (len(Y)):
        SSR +=  numpy.power ((Y[i] - meanYpred4r2), 2)

    if len(Y) == num_predictors+1:
        StdError = 0
        Fvalue = 0
    else:
        StdError = numpy.sqrt(abs(RSS/(len(Y)-num_predictors-1)))
        Fvalue = (SSR/num_predictors)/(RSS/(len(Y)-num_predictors-1))

    if R2wolfram<0:
        R2wolfram = 0
    if R2adjusted<0:
        R2adjusted = 0
    if StdError<0:
        StdError = 0
    if Fvalue<0:
        Fvalue = 0
    return round(R2wolfram,2), round(R2adjusted,2), round(RMSD,2), round(Fvalue,2), round(StdError,2), fig1, fig2

"""
    Get Classification Stats
"""
def stats_classification(Y, predY):
    Accuracy = sklearn.metrics.accuracy_score(Y, predY) #, normalize=True, sample_weight=None) #pos label 1 deleted
    Precision = sklearn.metrics.precision_score(Y, predY, pos_label=None)#, labels=None, average='binary', sample_weight=None)
    Recall = sklearn.metrics.recall_score(Y, predY, pos_label=None)#, labels=None, average='binary', sample_weight=None)
    F1_score = sklearn.metrics.f1_score(Y, predY, pos_label=None)#, labels=None, average='binary', sample_weight=None)
    Jacc = sklearn.metrics.jaccard_similarity_score(Y, predY) 

    ## General case for roc/auc
    """
    AUC_decision = []
    editedY = []
    indices = list(set(Y)) 
    for i in range (len(indices)):
        editedY.append([])
        AUC_decision.append([])

    for i in range (len(Y)):
        for j in range (len(indices)):
            if Y[i] == indices[j]: 
                editedY[j].append(0) 
                if predY[i] == indices[j]: 
                    AUC_decision[j].append(0) 
                else: 
                    AUC_decision[j].append(1) 
            else:
                editedY[j].append(1) 
                if predY[i] == indices[j]: 
                    AUC_decision[j].append(0) 
                else: 
                    AUC_decision[j].append(1) 

    fpr = []
    tpr = []
    thresholds = []
    for i in range (len(indices)):
            fpr.append([])
            tpr.append([])
            thresholds.append([])


    for i in range (len(indices)):
        fpr[i], tpr[i], thresholds[i] = sklearn.metrics.roc_curve(editedY[i], AUC_decision[i])

    myROC = plt.figure()

    for i in range(len(indices)):
        plt.plot(fpr[i], tpr[i], label='TP/FP rates for Class: '+str(indices[i])) ## Change Message if ROC

    #all_fpr = numpy.unique(numpy.concatenate([fpr[i] for i in range(len(indices))]))
    #all_tpr = numpy.unique(numpy.concatenate([tpr[i] for i in range(len(indices))]))
	
    #all_fpr = numpy.sort(numpy.concatenate([fpr[i] for i in range(len(indices))]))
    #all_tpr = numpy.sort(numpy.concatenate([tpr[i] for i in range(len(indices))]))
	
    #plt.plot(all_fpr, all_tpr, label='Max ROC')

    plt.plot([0, 1], [0, 1], 'k--') # y = x
    plt.xlim([-0.05, 1.05]) # +/- 0.05
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-like Representation of Confusion Matrix') ## Change Message if ROC
    plt.legend(loc="lower right")

    plt.tight_layout()

    figfileROC = BytesIO()
    myROC.savefig(figfileROC, dpi=300, format='png')
    saveas = pickle.dumps(figfileROC.getvalue())
    roc_encoded = base64.b64encode(saveas)

    #figfileROC = BytesIO()
    #plt.savefig(figfileROC, format='png')
    #figfileROC.seek(0)  # rewind to beginning of file
    #roc_encoded = base64.b64encode(figfileROC.getvalue())

    #plt.show() ##
    plt.close()
    ## end general case roc/auc
    """

    # DEBUG Conf Mat
    #from collections import Counter
    #print Y, "\n", predY, "\n", list(set(Y))
    #print Counter(Y), Counter(predY)

    cm = confusion_matrix(Y, predY, labels = list(set(Y)))
    numpy.set_printoptions(precision=2)

    myFIGA = plt.figure()
    plot_confusion_matrix(cm, list(set(Y)), title='Confusion matrix')

    plt.tight_layout()

    ## String IO
    #sio = cStringIO.StringIO()
    #myFIGA.savefig(sio, dpi=300, format='png') 
    #saveas = pickle.dumps(sio.getvalue())
    #cm_encoded = base64.b64encode(saveas)

    ## Bytes IO
    #bio = BytesIO()
    #myFIGA.savefig(bio, dpi=300, format='png')
    #saveas = pickle.dumps(bio.getvalue())
    #cm_encoded = base64.b64encode(saveas)

    ## Bytes IO v2
    figfile = BytesIO()
    plt.savefig(figfile, dpi=300, format='png')
    figfile.seek(0)  # rewind to beginning of file
    cm_encoded = base64.b64encode(figfile.getvalue())

    #plt.show() ## show CM
    plt.close()

    ## can plot normalized cm as well
    #cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
    #plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

    return round(Accuracy,2), round(Precision,2), round(Recall,2), round(F1_score,2), round(Jacc,2), cm_encoded #, roc_encoded


"""
    [[],[]] Transposed Matrix to dictionary
"""
def mat2dic(matrix):
    myDict = {}
    for i in range (len (matrix)):
        myDict["Row_" + str(i+1)] = [matrix[i][0], matrix[i][1]]
    return myDict

@app.route('/pws/validation', methods = ['POST'])
def create_task_validation():

    if not request.environ['body_copy']:
        abort(500)

    readThis = json.loads(request.environ['body_copy'])

    real, predicted, type, number_of_variables, predictionFeature, predictedFeature = getJsonContents(readThis)
    #print real,"\n", predicted,"\n",  type,"\n",  number_of_variables,"\n",  predictionFeature, "\n", predictedFeature

    full_table = [real, predicted]
    full_table_transposed = map(list, zip(*full_table)) 
    #print full_table,full_table_transposed
    full_table_dict = mat2dic(full_table_transposed)
    #print full_table_dict

    if type == "REGRESSION" and (max(real)-min(real)!=0) and (max(predicted)-min(predicted)!=0):
        R2wolfram, R2adjusted, RMSD, Fvalue, StdError, fig1, fig2 = stats_regression(real, predicted, number_of_variables)
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
                   "Real Vs Predicted" : fig1,
                   "QQ Plot" : fig2
                   }
        }
    elif type == "CLASSIFICATION":
        Accuracy, Precision, Recall, F1_score, Jaccard, cm_encoded = stats_classification(real, predicted)
        task = {
        "singleCalculations": {"Algorithm Type": type, 
                               "Number of predictor variables": number_of_variables,
                               "Accuracy" : Accuracy,
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
        "figures": {
                   #"ROC-like Curve for TP/FP rates" : roc_encoded,
                   "Confusion Matrix" : cm_encoded
                   }
        }
    else:
        task = {
        "singleCalculations": {"Validation Failed": "Check Dataset"
                              },
        "arrayCalculations": {"Reasons Include" : 
                               {"colNames": ["Real", "Predicted"],
                                "values": {
                                           "Row_1" : ["1", "Empty Dataset"],
                                           "Row_2" : ["2", "Prediction Feature Values Identical"],
                                           "Row_3" : ["3", "Predicted Feature Values Identical"]
                                          }
                               }
                             },
        "figures": {
                   }
        }

    jsonOutput = jsonify( task )

    ## DEBUG 
    #print fig1
    """
    with open("C:/Python27/delete_this", "rb") as b64_file:
         content = b64_file.read()

    decc = base64.standard_b64decode(content) 
    print decc
    mystr = pickle.loads(decc)
    stb = io.BytesIO(mystr)
    img = Image.open(stb)
    img.seek(0)
    img.save('C:/Python27/delete_this_too.png', 'png')
    """
    # REGRESSION IMAGES
    """
    decc = base64.standard_b64decode(fig1) 
    mystr = pickle.loads(decc)
    stb = io.BytesIO(mystr)
    img = Image.open(stb)
    img.seek(0)
    img.save('C:/Python27/Flask-0.10.1/python-api/Val/fig1W.png', 'png')

    decc = base64.standard_b64decode(fig2) 
    mystr = pickle.loads(decc)
    stb = io.BytesIO(mystr)
    img = Image.open(stb)
    img.seek(0)
    img.save('C:/Python27/Flask-0.10.1/python-api/Val/fig2W.png', 'png')
    """
    # CLASSIFICATION IMAGES
    """
    decc = base64.standard_b64decode(cm_encoded) 
    mystr = pickle.loads(decc)
    stb = io.BytesIO(mystr)
    img = Image.open(stb)
    img.seek(0)
    img.save('C:/Python27/Flask-0.10.1/python-api/Val/confmat.png', 'png')
    """
    return jsonOutput, 201 

############################################################

class WSGICopyBody(object):
    def __init__(self, application):
        self.application = application

    def __call__(self, environ, start_response):
        from cStringIO import StringIO
        input = environ.get('wsgi.input')
        length = environ.get('CONTENT_LENGTH', '0')
        length = 0 if length == '' else int(length)
        body = ''
        if length == 0:
            environ['body_copy'] = ''
            if input is None:
                return
            if environ.get('HTTP_TRANSFER_ENCODING','0') == 'chunked':
                size = int(input.readline(),16)
                while size > 0:
                    temp = str(input.read(size+2)).strip()
                    body += temp
                    size = int(input.readline(),16)
        else:
            body = environ['wsgi.input'].read(length)
        environ['body_copy'] = body
        environ['wsgi.input'] = StringIO(body)

        # Call the wrapped application
        app_iter = self.application(environ, 
                                    self._sr_callback(start_response))

        # Return modified response
        #print app_iter
        return app_iter

    def _sr_callback(self, start_response):
        def callback(status, headers, exc_info=None):

            # Call upstream start_response
            start_response(status, headers, exc_info)
        #print callback
        return callback

############################################################

if __name__ == '__main__': 
    app.wsgi_app = WSGICopyBody(app.wsgi_app) ##
    app.run(host="0.0.0.0", port = 5000, debug = True)

# curl -i -H "Content-Type: application/json" -X POST -d @C:/Python27/Flask-0.10.1/python-api/val.json http://localhost:5000/pws/validation
# curl -i -H "Content-Type: application/json" -X POST -d @C:/Python27/Flask-0.10.1/python-api/valW.json http://localhost:5000/pws/validation
# C:\Python27\Flask-0.10.1\python-api 
# C:/Python27/python valid_service.py