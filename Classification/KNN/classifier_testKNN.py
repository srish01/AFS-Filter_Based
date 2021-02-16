#!/usr/bin python 

# Copyright 2021 Gregory Ditzler 
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this 
# software and associated documentation files (the "Software"), to deal in the Software 
# without restriction, including without limitation the rights to use, copy, modify, 
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
# permit persons to whom the Software is furnished to do so, subject to the following 
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies 
# or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE 
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT 
# OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
# OTHER DEALINGS IN THE SOFTWARE.

import numpy as np 
import pandas as pd 

import argparse

from utils import kuncheva, jaccard

import sys
sys.path.append("./scikit-feature/")
import skfeature as skf
from skfeature.function.information_theoretical_based import JMI, MIM, MRMR, MIFS
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

print(sys.path)
# -----------------------------------------------------------------------
# # setup program constants 
# percentage of poisoning levels  
POI_RNG = [.01, .025, .05, .075, .1, .125, .15, .175, .2]
# total number of poisoning levels 
NPR = len(POI_RNG)
# percentage of features that we want to select 
SEL_PERCENT = .1
# number of algorithms that we are going to test [JMI, MIM, MRMR, MIFS]
NALG = 4
# used when we select features 
FEAT_IDX = 0
# number of cross validation runs to perform
CV = 5
# dataset names 
# did not run 
#   - miniboone, connect-4
DATA = [
        'conn-bench-sonar-mines-rocks',        
        'ionosphere',
        'bank',
        'oocytes_trisopterus_nucleus_2f', 
        'statlog-german-credit', 
        'molec-biol-promoter', 
        # 'ozone', 
        # 'spambase',
        'parkinsons', 
        'oocytes_merluccius_nucleus_4d',
        # 'musk-1', 
        # 'musk-2', 
        # 'chess-krvkp', 
        # 'twonorm'
]
        
        
BOX = ['0.5', '1','1.5', '2', '2.5', '5']
# -----------------------------------------------------------------------

def KNN_classification(X_tr, y_tr, X_te, y_te):
        scaler = StandardScaler()
        scaler.fit(X_tr)

        X_tr = scaler.transform(X_tr)
        X_te = scaler.transform(X_te)
    
        clf1 = KNeighborsClassifier(n_neighbors=5)
        clf1.fit(X_tr, y_tr)
        y_pred_knn = clf1.predict(X_te)
        return(y_pred_knn)

def experiment(data, box, cv, output):
    """
    Write the results of an experiment.
        This function will run an experiment for a specific dataset for a bounding box. 
        There will be CV runs of randomized experiments run and the outputs will be 
        written to a file. 

        Parameters
        ----------
        data : string
            Dataset name.
            
        box : string 
            Bounding box on the file name.
        cv : int 
            Number of cross validation runs. 
            
        output : string
            If float or tuple, the projection will be the same for all features,
            otherwise if a list, the projection will be described feature by feature.
                    
        Returns
        -------
        None
            
        Raises
        ------
        ValueError
            If the percent poison exceeds the number of samples in the requested data.
    """
    #data, box, cv, output = 'conn-bench-sonar-mines-rocks', '1', 5, 'results/test.npz'

    # load normal and adversarial data 
    path_adversarial_data = 'data/attacks/' + data + '_[xiao][' + box + '].csv'
    df_normal = pd.read_csv('data/clean/' + data + '.csv', header=None).values
    df_adversarial = pd.read_csv(path_adversarial_data, header=None).values
    
    # separate out the normal and adversarial data 
    Xn, yn = df_normal[:,:-1], df_normal[:,-1]
    Xa, ya = df_adversarial[:,:-1], df_adversarial[:,-1]
    
    # change the labels from +/-1 to [0,1]
    ya[ya==-1], yn[yn==-1] = 0, 0

    # calculate the rattios of data that would be used for training and hold out  
    p0, p1 = 1./cv, (1. - 1./cv)
    N = len(Xn)
    # calculate the total number of training and testing samples and set the number of 
    # features that are going to be selected 
    Ntr, Nte = int(p1*N), int(p0*N)                                             ##### [OBS]: Losing one feature in the process
    n_selected_features = int(Xn.shape[1]*SEL_PERCENT)+1
       
    # zero the results out 
    acc_KNN = np.zeros((NPR,6))
    ####################################
    # CLASSIFICATION
    ##################################
    
    # run `cv` randomized experiments. note this is not performing cross-validation, rather
    # we are going to use randomized splits of the data.  
    for _ in range(cv): 
        # shuffle up the data for the experiment then split the data into a training and 
        # testing dataset
        i = np.random.permutation(N)
        Xtrk, ytrk, Xtek, ytek = Xn[i][:Ntr], yn[i][:Ntr], Xn[i][-Nte:], yn[i][-Nte:]

        
        ####### Classification on Normal Data with no FS #######################
        yn_allfeature_KNN = KNN_classification(Xtrk, ytrk, Xtek, ytek)
           
        ####### Classification on JMI-based features on Normal data #############
        sf_base_jmi = JMI.jmi(Xtrk, ytrk, n_selected_features=n_selected_features)[FEAT_IDX]
        #print("\nNOR: JMI features", sf_base_jmi)
        Xtr_jmi = Xtrk[:, sf_base_jmi]
        Xte_jmi = Xtek[:, sf_base_jmi]
        yn_JMI_KNN = KNN_classification(Xtr_jmi, ytrk, Xte_jmi, ytek)
                
        for n in range(NPR): 

            # calucate the number of poisoned data that we are going to need to make sure 
            # that the poisoning ratio is correct in the training data. e.g., if you have 
            # N=100 samples and you want to poison by 20% then the 20% needs to be from 
            # the training size. hence it is not 20. 
            Np = int(len(ytrk)*POI_RNG[n]+1)
            if Np >= len(ya): 
                # shouldn't happen but catch the case where we are requesting more poison
                # data samples than are available. NEED TO BE CAREFUL WHEN WE ARE CREATING 
                # THE ADVERSARIAL DATA
                ValueError('Number of poison data requested is larger than the available data.')

            # find the number of normal samples (i.e., not poisoned) samples in the 
            # training data. then create the randomized data set that has Nn normal data
            # samples and Np adversarial samples in the training data
            Nn = len(ytrk) - Np
            idx_normal, idx_adversarial = np.random.permutation(len(ytrk))[:Nn], \
                                            np.random.permutation(len(ya))[:Np]
            Xtrk_poisoned, ytrk_poisoned = np.concatenate((Xtrk[idx_normal], Xa[idx_adversarial])), \
                                            np.concatenate((ytrk[idx_normal], ya[idx_adversarial]))   
            
            ya_allfeature_KNN = KNN_classification(Xtrk_poisoned, ytrk_poisoned, Xtek, ytek)
            
            # run feature selection with the training data that has adversarial samples
            sf_adv_jmi = JMI.jmi(Xtrk_poisoned, ytrk_poisoned, n_selected_features=n_selected_features)[FEAT_IDX]
            sf_adv_mim = MIM.mim(Xtrk_poisoned, ytrk_poisoned, n_selected_features=n_selected_features)[FEAT_IDX]
            sf_adv_mrmr = MRMR.mrmr(Xtrk_poisoned, ytrk_poisoned, n_selected_features=n_selected_features)[FEAT_IDX]
            sf_adv_misf = MIFS.mifs(Xtrk_poisoned, ytrk_poisoned, n_selected_features=n_selected_features)[FEAT_IDX]
            
            # KNN Classification on JMI selected features
            Xtrk_poisoned_JMI = Xtrk_poisoned[:, sf_adv_jmi]
            Xtest_JMI = Xtek[:, sf_adv_jmi]
            ya_JMI_KNN = KNN_classification(Xtrk_poisoned_JMI, ytrk_poisoned, Xtest_JMI, ytek)
            # KNN Classification on MIM selected features
            Xtrk_poisoned_MIM = Xtrk_poisoned[:, sf_adv_mim]
            Xtest_MIM = Xtek[:, sf_adv_mim]
            ya_MIM_KNN = KNN_classification(Xtrk_poisoned_MIM, ytrk_poisoned, Xtest_MIM, ytek)
            # KNN Classification on MRMR selected features
            Xtrk_poisoned_MRMR = Xtrk_poisoned[:, sf_adv_mrmr]
            Xtest_MRMR = Xtek[:, sf_adv_mrmr]
            ya_MRMR_KNN = KNN_classification(Xtrk_poisoned_MRMR, ytrk_poisoned, Xtest_MRMR, ytek)
            # KNN Classification on MISF selected features
            Xtrk_poisoned_MISF = Xtrk_poisoned[:, sf_adv_misf]
            Xtest_MISF = Xtek[:, sf_adv_misf]
            ya_MISF_KNN = KNN_classification(Xtrk_poisoned_MISF, ytrk_poisoned, Xtest_MISF, ytek)
            """
            ######### KNN Classification on adversarial data with no FS #################
            ya_allfeature_KNN = KNN_classification(Xtrk_poisoned, ytrk_poisoned, Xtek, ytek)
            #print("[ADV] KNN: No FS Confusion Matrix for Poisoning Ratio: ", POI_RNG[n], "\n", confusion_matrix(ytek, ya_allfeature_KNN))
            #print(classification_report(ytek, ya_allfeature_KNN))
            #print("[ADV] KNN: NO FS Accuracy for Poisoning ratio", POI_RNG[n], "\n",  accuracy_score(ytek, ya_allfeature_KNN))
            
            ######### KNN Classification on adversarial data with JMI FS #################
            sf_adv_jmi = JMI.jmi(Xtrk_poisoned, ytrk_poisoned, n_selected_features=n_selected_features)[FEAT_IDX]
            Xtrk_poisoned_JMI = Xtrk_poisoned[:, sf_adv_jmi]
            Xtest_JMI = Xtek[:, sf_adv_jmi]
            
            ya_JMI_KNN = KNN_classification(Xtrk_poisoned_JMI, ytrk_poisoned, Xtest_JMI, ytek)
            #print("\nJMI Features: ", sf_adv_jmi)
            #print("[ADV] KNN: JMI FS Confusion Matrix for Poisoning Ratio: ", POI_RNG[n], "\n", confusion_matrix(ytek, ya_JMI_KNN))
            #print("[ADV] KNN: JMI Accuracy for Poisoning ratio", POI_RNG[n], "\n", accuracy_score(ytek, ya_JMI_KNN))
            
            ######### KNN Classification on adversarial data with MIM FS #################
            sf_adv_mim = MIM.mim(Xtrk_poisoned, ytrk_poisoned, n_selected_features=n_selected_features)[FEAT_IDX]
            Xtrk_poisoned_MIM = Xtrk_poisoned[:, sf_adv_mim]
            Xtest_MIM = Xtek[:, sf_adv_mim]
            
            ya_MIM_KNN = KNN_classification(Xtrk_poisoned_MIM, ytrk_poisoned, Xtest_MIM, ytek)
            #print("\nMIM Features: ", sf_adv_mim)
            #print("[ADV] KNN: MIM FS Confusion Matrix for Poisoning Ratio: ", POI_RNG[n], "\n", confusion_matrix(ytek, ya_MIM_KNN))
            #print("[ADV] KNN: MIM Accuracy for Poisoning ratio", POI_RNG[n], "\n", accuracy_score(ytek, ya_MIM_KNN))
            
            ######### KNN Classification on adversarial data with MRMR FS #################
            sf_adv_mrmr = MRMR.mrmr(Xtrk_poisoned, ytrk_poisoned, n_selected_features=n_selected_features)[FEAT_IDX]
            Xtrk_poisoned_MRMR = Xtrk_poisoned[:, sf_adv_mrmr]
            Xtest_MRMR = Xtek[:, sf_adv_mrmr]
            
            ya_MRMR_KNN = KNN_classification(Xtrk_poisoned_MRMR, ytrk_poisoned, Xtest_MRMR, ytek)
            #print("\nMRMR Features: ", sf_adv_mrmr)
            #print("[ADV] KNN: MRMR FS Confusion Matrix for Poisoning Ratio: ", POI_RNG[n], "\n", confusion_matrix(ytek, ya_MRMR_KNN))
            #print("[ADV] KNN: MRMR Accuracy for Poisoning ratio", POI_RNG[n], "\n", accuracy_score(ytek, ya_MRMR_KNN))
            
            ######### KNN Classification on adversarial data with MISF FS #################
            sf_adv_misf = MIFS.mifs(Xtrk_poisoned, ytrk_poisoned, n_selected_features=n_selected_features)[FEAT_IDX]
            Xtrk_poisoned_MISF = Xtrk_poisoned[:, sf_adv_misf]
            Xtest_MISF = Xtek[:, sf_adv_misf]
            
            ya_MISF_KNN = KNN_classification(Xtrk_poisoned_MISF, ytrk_poisoned, Xtest_MISF, ytek)
            #print("\nMISF Features: ", sf_adv_misf)
            #print("[ADV] KNN: MISF FS Confusion Matrix for Poisoning Ratio: ", POI_RNG[n], "\n", confusion_matrix(ytek, ya_MISF_KNN))
            #print("[ADV] KNN: MISF Accuracy for Poisoning ratio", POI_RNG[n], "\n", accuracy_score(ytek, ya_MISF_KNN))
            """
            # Calculate accumulated accuracy in a matrix of size 9x6
            acc_KNN[n, 0] += accuracy_score(ytek, yn_allfeature_KNN)    # Acc score of normal data without Feature Selection
            acc_KNN[n, 1] += accuracy_score(ytek, ya_allfeature_KNN)    # Acc score of adversarial data without Feature Selection
            acc_KNN[n, 2] += accuracy_score(ytek, ya_JMI_KNN)    # Acc score of adversarial data with JMI Feature Selection algo
            acc_KNN[n, 3] += accuracy_score(ytek, ya_MIM_KNN)    # Acc score of adversarial data with MIM Feature Selection algo
            acc_KNN[n, 4] += accuracy_score(ytek, ya_MRMR_KNN)    # Acc score of adversarial data with MRMR Feature Selection algo
            acc_KNN[n, 5] += accuracy_score(ytek, ya_MISF_KNN)    # Acc score of adversarial data with MISF Feature Selection algo
            
            
    #print(acc_KNN)
    # scale the accuracy statistics by 1.0/cv then write the output file
    acc_KNN = acc_KNN/cv
    print("\n Accuracy matrix of KNN")
    print("[COL]: Norm_noFS, Adv_noFS, Adv_JMI, Adv_MIM, Adv_MRMR, Adv_MISF")
    print("[ROW]: Poisoning ratios: 0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2")
    print("\n", acc_KNN)
    
    np.savez(output, acc_KNN=acc_KNN)
    return None
        
if __name__ == '__main__': 

    for data in DATA: 
        for box in BOX: 
            print('Running ' + data + ' - box:' + box)
            #try: 
            experiment(data, box, CV, 'results/Classification_accuracy/KNN/' + data + '_[xiao][' + box + ']_classification_results.npz')
            #except: 
            #    print(' ... ERROR ...')