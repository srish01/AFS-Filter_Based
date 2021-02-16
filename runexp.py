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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

# -----------------------------------------------------------------------
# # setup program constants
# Use thread pool executor for feature selection
PARALLEL = True
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
#   - miniboone, connect-4, ozone, spambase
DATA = [
        'conn-bench-sonar-mines-rocks',
        'ionosphere',
        # 'bank',
        # 'oocytes_trisopterus_nucleus_2f', 
        # 'statlog-german-credit', 
        'molec-biol-promoter', 
        # #'ozone', 
        # #'spambase',
        # 'parkinsons', 
        'oocytes_merluccius_nucleus_4d',
        'musk-1', 
        # 'musk-2', 
        # 'chess-krvkp', 
        'twonorm'
        
]
BOX = ['0.5', '1', '1.5', '2', '2.5'] # 5]
# -----------------------------------------------------------------------

def run_feature_selection(X, Y, n_selected_features):
    
    lst = []
    
    if PARALLEL:
        # with multiprocessing.Pool(processes=4) as pool:
        #     lst.append(pool.apply(JMI.jmi, args=(X, Y), kwds={'n_selected_features': n_selected_features}))
        #     lst.append(pool.apply(MIM.mim, args=(X, Y), kwds={'n_selected_features': n_selected_features}))
        #     lst.append(pool.apply(MRMR.mrmr, args=(X, Y), kwds={'n_selected_features': n_selected_features}))
        #     lst.append(pool.apply(MIFS.mifs, args=(X, Y), kwds={'n_selected_features': n_selected_features}))
            
        # lst = [l[FEAT_IDX] for l in lst]
        
        with ProcessPoolExecutor(max_workers=4) as executor:
            lst.append(executor.submit(JMI.jmi, X, Y, n_selected_features=n_selected_features))
            lst.append(executor.submit(MIM.mim, X, Y, n_selected_features=n_selected_features))
            lst.append(executor.submit(MRMR.mrmr, X, Y, n_selected_features=n_selected_features))
            lst.append(executor.submit(MIFS.mifs, X, Y, n_selected_features=n_selected_features))   
        lst = [l.result()[FEAT_IDX] for l in lst]
    else:
        lst.append(JMI.jmi(X, Y, n_selected_features=n_selected_features)[FEAT_IDX])
        lst.append(MIM.mim(X, Y, n_selected_features=n_selected_features)[FEAT_IDX])
        lst.append(MRMR.mrmr(X, Y, n_selected_features=n_selected_features)[FEAT_IDX])
        lst.append(MIFS.mifs(X, Y, n_selected_features=n_selected_features)[FEAT_IDX])

    return lst

def err_KNN_classification(X_tr, y_tr, X_te, y_te):
    scaler = StandardScaler()
    scaler.fit(X_tr)

    X_tr = scaler.transform(X_tr)
    X_te = scaler.transform(X_te)
    
    clf1 = KNeighborsClassifier(n_neighbors=5)
    clf1.fit(X_tr, y_tr)
    y_pred_knn = clf1.predict(X_te)
    accuracy_KNN = accuracy_score(y_te, y_pred_knn)
    error = np.mean(y_pred_knn != y_te)
    return(error)

# comb_kuncheva function is used to find the stability among feature sets calculated for at each cv for every npr
# fset is a list of 9 lists(npr=9) that has features for every pass, 
# ex: JMI_fset [[f1,f2,f3], [f1,f2,f3], [f1,f2,f3], [f1,f2,f3]...., [f1,f2,f3]]
# combination() is used to create non repeatable pairs(for r=2), ex: iter = ABCD; comb = combination(iter, 2) will give
# comb = AB, AC, AD, BC, BD, CD
def comb_kuncheva(fset, r, CV, no_of_col):
    fin = []
    for i in fset:
        comb = list(combinations(i, r))     # creates non-repeatable pairs of features, ex: (f1,f2), (f1,f3), (f2,f3)
        kun = 0
        for a in comb:
            kun += kuncheva(a[0], a[1], no_of_col)
        fin.append(kun/len(comb))
    return np.array(fin).T

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

    # calculate the ratios of data that would be used for training and hold out  
    p0, p1 = 1./cv, (1. - 1./cv)
    N = len(Xn)
    # calculate the total number of training and testing samples and set the numbfer of 
    # features that are going to be selected 
    Ntr, Nte = int(p1*N), int(p0*N)
    n_selected_features = int(Xn.shape[1]*SEL_PERCENT)+1

    # zero the results out : err_jaccard and err_kuncheva are 9x4 matrices
    err_jaccard, err_kuncheva = np.zeros((NPR, NALG)), np.zeros((NPR, NALG))
    # For M3(KNN classification error) analysis: err_KNN_norm will have just one row(1x4) because it only only contains normal data
    # err_KNN_pois is a 9x4 matrix
    err_KNN_norm, err_KNN_pois = np.zeros((1,NALG)), np.zeros((NPR,NALG))   
    
    # Empty lists that will hold feature sets for all npr
    MIM_fset = []
    MIFS_fset = []
    MRMR_fset = []
    JMI_fset = []
   
   # creating list of empty lists  
    for n in range (NPR):
        MIM_fset.append([])
        MIFS_fset.append([])
        MRMR_fset.append([])
        JMI_fset.append([])
    
    # run `cv` randomized experiments. note this is not performing cross-validation, rather
    # we are going to use randomized splits of the data.  
    for _ in range(cv): 
        # shuffle up the data for the experiment then split the data into a training and 
        # testing dataset
        i = np.random.permutation(N)
        Xtrk, ytrk, Xtek, ytek = Xn[i][:Ntr], yn[i][:Ntr], Xn[i][Nte:], yn[i][Nte:]
        # run feature selection on the baseline dataset without an adversarial data. this 
        # will serve as the baseline. use a parallel assignment to speed things up. 
        sf_base_jmi, sf_base_mim, sf_base_mrmr, sf_base_mifs = run_feature_selection(Xtrk, ytrk, n_selected_features)
        
        Xtr_mim = Xtrk[:, sf_base_mim]
        Xtr_mifs = Xtrk[:, sf_base_mifs]
        Xtr_mrmr = Xtrk[:, sf_base_mrmr]
        Xtr_jmi = Xtrk[:, sf_base_jmi]
        
        Xte_mim = Xtek[:, sf_base_mim]
        Xte_mifs = Xtek[:, sf_base_mifs]
        Xte_mrmr = Xtek[:, sf_base_mrmr]
        Xte_jmi = Xtek[:, sf_base_jmi]
        
        # err_KNN_norm table gives us the classification accuracy score of feature selection
        # algorithms performed on untainted data, that can be used for further analysis
        err_KNN_norm[0, 0] += err_KNN_classification(Xtr_mim, ytrk, Xte_mim, ytek)
        err_KNN_norm[0, 1] += err_KNN_classification(Xtr_mifs, ytrk, Xte_mifs, ytek)
        err_KNN_norm[0, 2] += err_KNN_classification(Xtr_mrmr, ytrk, Xte_mrmr, ytek)
        err_KNN_norm[0, 3] += err_KNN_classification(Xtr_jmi, ytrk, Xte_jmi, ytek)
                
        # loop over the number of poisoning ratios that we need to evaluate
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
                raise ValueError('Number of poison data requested is larger than the available data.')

            # find the number of normal samples (i.e., not poisoned) samples in the 
            # training data. then create the randomized data set that has Nn normal data
            # samples and Np adversarial samples in the training data
            Nn = len(ytrk) - Np
            idx_normal, idx_adversarial = np.random.permutation(len(ytrk))[:Nn], \
                                          np.random.permutation(len(ya))[:Np]
            Xtrk_poisoned, ytrk_poisoned = np.concatenate((Xtrk[idx_normal], Xa[idx_adversarial])), \
                                           np.concatenate((ytrk[idx_normal], ya[idx_adversarial]))
            
            # run feature selection with the training data that has adversarial samples
            sf_adv_jmi, sf_adv_mim, sf_adv_mrmr, sf_adv_mifs = run_feature_selection(Xtrk_poisoned, ytrk_poisoned, n_selected_features)
            
            Xtrk_poisoned_MIM = Xtrk_poisoned[:, sf_adv_mim]
            Xtrk_poisoned_MIFS = Xtrk_poisoned[:, sf_adv_mifs]
            Xtrk_poisoned_MRMR = Xtrk_poisoned[:, sf_adv_mrmr]           
            Xtrk_poisoned_JMI = Xtrk_poisoned[:, sf_adv_jmi]
            
            Xtest_MIM = Xtek[:, sf_adv_mim]
            Xtest_MIFS = Xtek[:, sf_adv_mifs]
            Xtest_MRMR = Xtek[:, sf_adv_mrmr]
            Xtest_JMI = Xtek[:, sf_adv_jmi]
                       
            # calculate the accumulated jaccard and kuncheva performances for each of the 
            # feature selection algorithms 
            err_jaccard[n, 0] += jaccard(sf_adv_mim, sf_base_mim)
            err_jaccard[n, 1] += jaccard(sf_adv_mifs, sf_base_mifs)
            err_jaccard[n, 2] += jaccard(sf_adv_mrmr, sf_base_mrmr)
            err_jaccard[n, 3] += jaccard(sf_adv_jmi, sf_base_jmi)

            err_kuncheva[n, 0] += kuncheva(sf_adv_mim, sf_base_mim, Xtrk.shape[1])
            err_kuncheva[n, 1] += kuncheva(sf_adv_mifs, sf_base_mifs, Xtrk.shape[1])
            err_kuncheva[n, 2] += kuncheva(sf_adv_mrmr, sf_base_mrmr, Xtrk.shape[1])
            err_kuncheva[n, 3] += kuncheva(sf_adv_jmi, sf_base_jmi, Xtrk.shape[1])
                      
            # err_KNN_pois table gives the classification accuracy score of feature selection
            # algorithms performed on poisoned data
            err_KNN_pois[n, 0] += err_KNN_classification(Xtrk_poisoned_MIM, ytrk_poisoned, Xtest_MIM, ytek)
            err_KNN_pois[n, 1] += err_KNN_classification(Xtrk_poisoned_MIFS, ytrk_poisoned, Xtest_MIFS, ytek)
            err_KNN_pois[n, 2] += err_KNN_classification(Xtrk_poisoned_MRMR, ytrk_poisoned, Xtest_MRMR, ytek)
            err_KNN_pois[n, 3] += err_KNN_classification(Xtrk_poisoned_JMI, ytrk_poisoned, Xtest_JMI, ytek)

            # Storing all the features in corresponding feature selection algo list
            MIM_fset[n].append(sf_adv_mim)
            MIFS_fset[n].append(sf_base_mifs)
            MRMR_fset[n].append(sf_base_mrmr)
            JMI_fset[n].append(sf_adv_jmi)
    
    MIM_stability_score = comb_kuncheva(MIM_fset, 2, cv, Xtrk.shape[1])
    MIFS_stability_score = comb_kuncheva(MIFS_fset, 2, cv, Xtrk.shape[1])        
    MRMR_stability_score = comb_kuncheva(MRMR_fset, 2, cv, Xtrk.shape[1])
    JMI_stability_score = comb_kuncheva(JMI_fset, 2, cv, Xtrk.shape[1])
    
    feature_stability = np.column_stack((MIM_stability_score, MIFS_stability_score, MRMR_stability_score, JMI_stability_score))
                  
    # scale the kuncheva and jaccard statistics by 1.0/cv then write the output file
    err_jaccard,  err_kuncheva  = err_jaccard/cv, err_kuncheva/cv
    err_KNN_pois, err_KNN_norm = err_KNN_pois/cv, err_KNN_norm/cv
    np.savez(output, M1 = feature_stability, err_jaccard=err_jaccard, M2=err_kuncheva, M3_pois=err_KNN_pois, M3_norm = err_KNN_norm)
    
    return None

if __name__ == '__main__': 

    for data in DATA: 
        for box in BOX: 
            print('Running ' + data + ' - box:' + box)
            #try: 
            experiment(data, box, CV, 'results/reduced_Boxsize/' + data + '_[xiao][' + box + ']_results.npz')
            #except:
            #    print(' ... ERROR ...')

