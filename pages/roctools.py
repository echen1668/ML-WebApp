import pandas as pd
import numpy as np
from scipy.stats import norm, ranksums
from scipy import stats
from scipy.special import ndtri
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils.extmath import stable_cumsum
from sklearn.metrics import matthews_corrcoef
import seaborn as sns
import pickle
import datetime
import pprint
import pymongo
from pymongo import MongoClient
import streamlit as st
import io


# for metric confidence intervals
# pip install confidenceinterval@git+https://github.com/jacobgil/confidenceinterval 
# https://pypi.org/project/classification-confidence-intervals/

# AUC comparison adapted from
# https://github.com/Netflix/vmaf/

def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2

def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def delong_roc_variance(ground_truth, predictions):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert len(
        aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov


def auc_test(y_true, y_pred, alternative='two-sided'):
    """
    Compare AUC significantly better than 0.5
    H0 = "The AUC is equal to 0.5".
    H0 = "The distribution of the ranks in the two groups are equal".
    use a Mann-Whitney-Wilcoxon test
    
    Args:
       y_true: np.array of 0 and 1
       y_pred: np.array of prediction scores
    """
    pos_score = y_pred[y_true==1]
    neg_score = y_pred[y_true==0]
    _, p_value = ranksums(pos_score, neg_score)#, alternative=alternative)
    return p_value

def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes log(p-value) for hypothesis that two ROC AUCs are different
    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model,
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack(
        (predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov)


def proportion_confidence_interval(r, n, z):
    """Compute confidence interval for a proportion.

    Follows notation described on pages 46--47 of [1].

    References
    ----------
    [1] R. G. Newcombe and D. G. Altman, Proportions and their differences, in Statisics
    with Confidence: Confidence intervals and statisctical guidelines, 2nd Ed., D. G. Altman,
    D. Machin, T. N. Bryant and M. J. Gardner (Eds.), pp. 45-57, BMJ Books, 2000.
    """

    A = 2*r + z**2
    B = z*np.sqrt(z**2 + 4*r*(1 - r/n))
    C = 2*(n + z**2)
    return ((A-B)/C, (A+B)/C)


def sensitivity_and_specificity_with_confidence_intervals(TP, FP, FN, TN, alpha=0.95):
    """Compute confidence intervals for sensitivity and specificity using Wilson's method.

    This method does not rely on a normal approximation and results in accurate
    confidence intervals even for small sample sizes.

    Parameters
    ----------
    TP : int
        Number of true positives
    FP : int
        Number of false positives
    FN : int
        Number of false negatives
    TN : int
        Number of true negatives
    alpha : float, optional
        Desired confidence. Defaults to 0.95, which yields a 95% confidence interval.

    Returns
    -------
    sensitivity_point_estimate : float
        Numerical estimate of the test sensitivity
    specificity_point_estimate : float
        Numerical estimate of the test specificity
    sensitivity_confidence_interval : Tuple (float, float)
        Lower and upper bounds on the alpha confidence interval for sensitivity
    specificity_confidence_interval
        Lower and upper bounds on the alpha confidence interval for specificity

    References
    ----------
    [1] R. G. Newcombe and D. G. Altman, Proportions and their differences, in Statisics
    with Confidence: Confidence intervals and statisctical guidelines, 2nd Ed., D. G. Altman,
    D. Machin, T. N. Bryant and M. J. Gardner (Eds.), pp. 45-57, BMJ Books, 2000.
    [2] E. B. Wilson, Probable inference, the law of succession, and statistical inference,
    J Am Stat Assoc 22:209-12, 1927.
    """

    #
    z = -ndtri((1.0-alpha)/2)

    # Compute sensitivity using method described in [1]
    sensitivity_point_estimate = TP/(TP + FN)
    sensitivity_confidence_interval = _proportion_confidence_interval(
        TP, TP + FN, z)

    # Compute specificity using method described in [1]
    specificity_point_estimate = TN/(TN + FP)
    specificity_confidence_interval = _proportion_confidence_interval(
        TN, TN + FP, z)

    return sensitivity_point_estimate, specificity_point_estimate, sensitivity_confidence_interval, specificity_confidence_interval


def roc(y_true, y_pred, alpha=0.95):
    """
    Computes ROC AUC and its confidence interval
    Args:
       y_true: np.array of 0 and 1
       y_pred: np.array of floats of the probability of being class 1
       alpha: significance level
    Returns:
       auc: AUROC
       ci: [low, high]
    """
    auc, auc_cov = delong_roc_variance(y_true, y_pred)

    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)

    ci = norm.ppf(lower_upper_q, loc=auc, scale=auc_std)

    ci[ci > 1] = 1
    return auc, ci


def full_roc_curve(y_true, y_pred, alpha=0.95, index='youden'):
    """
    Computes all TPR,TNR,PPV,NPV,f1,DOR at possible thresholds, and the 95%CI of TPR, and PPV, and the AUC, and Youden index based metrics
    Args:
       y_true: np.array of 0 and 1
       y_pred: np.array of floats of the probability of being class 1
       alpha: significance level
    Returns:
       res: a struct with metrics by Youden Index
       res_array: a dataframe of TPR,TNR,PPV,NPV,f1,DOR at possible thresholds, and the 95%CI of TPR, and PPV
    """
    locs1 = np.isnan(y_true)
    locs2 = np.isnan(y_pred)
    locs = np.logical_or(locs1, locs2)
    y_true = y_true[~locs]
    y_pred = y_pred[~locs]
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    
    # MCC
    #mcc = matthews_corrcoef(y_true, y_pred)
    #print("MCC is:",mcc)

    #df = pd.read_excel(r'C:\projects\coagulopathy\features_outcome.xlsx', 1)
    #locs1 = np.isnan(df['SI_Mean'])
    res = {}
    #fpr, tpr, thresholds = roc_curve(
    #    df['fiblt200'].loc[~locs1], df['SI_Mean'].loc[~locs1])
    # calculate P and N
    P = np.sum(list(y_true))
    N = len(y_true) - P
    # calculate fnr, tnr, ppv, npv, f1, dor
    fnr = 1.0 - tpr
    tnr = 1.0 - fpr
    ppv = (tpr * P) / ((tpr * P) + (fpr * N))
    npv = (tnr * N) / ((tnr * N) + (fnr * P))
    f1 = (2.0 * ppv * tpr) / (ppv + tpr)
    dor = (tpr * tnr) / (fpr * fnr)
    # 95% CI for tpr, and ppv
    z = -ndtri((1.0-alpha)/2)
    # Calcuate TP, FP, TN, FN
    TP = tpr * P
    #FP = fpr * N
    #TN = tnr * N
    #FN = fnr * P
    # Calcaute MCC max index
    # MCC index cutoff: maximum (tpr * tnr * ppv * npv)**0.5) - ((1 - ppv) * fnr * fpr * (1 - npv))**0.5)
    max_index_mcc = np.argmax(np.nan_to_num((((tpr * tnr * ppv * npv)**0.5) - (((1 - ppv) * fnr * fpr * (1 - npv))**0.5)), nan=0))
    print("MCC Index:", max_index_mcc)
    cutoff_mcc = thresholds[max_index_mcc]

    cis = proportion_confidence_interval(TP, P, z) # CI for TPR
    n = TP + fpr * N # TP + FP
    cis_ppv = proportion_confidence_interval(TP, n, z)

    # Youden index cutoff: maximum tpr + tnr
    max_index_youden = np.argmax(tpr + tnr)
    print("Youden Index:", max_index_youden)
    cutoff_youden = thresholds[max_index_youden]
    # F1 score cutoff: maximum f1
    max_index_f1 = np.argmax(np.nan_to_num(f1, nan=0))
    print("Max F1 Index: ", max_index_f1)
    cutoff_f1 = thresholds[max_index_f1]
    # JI score cutoff: maximum ji
    max_index_ji = np.argmax(np.nan_to_num(TP/(TP+(fnr * P)+(fpr * N)), nan=0))
    print("Max JI Index: ", max_index_ji)
    cutoff_ji = thresholds[max_index_ji]
    # return full performance metrics, and youden index based
    res_array = pd.DataFrame(np.transpose([tpr, tnr, ppv, npv, f1, dor,cis[0], cis[1],cis_ppv[0],cis_ppv[1]]), columns=[
                             'tpr', 'tnr', 'ppv', 'npv', 'f1', 'dor','tpr_low','tpr_high','ppv_low','ppv_high'])
    auroc, ci = roc(y_true, y_pred)
    locs_ppv_nan = np.isnan(ppv)
    try:
        auprc = auc(tpr[~locs_ppv_nan], ppv[~locs_ppv_nan])
    except:
        auprc = np.nan
    res['auc'] = auroc
    res['p_value'] = auc_test(y_true, y_pred)# compare AUC with 0.5
    res['auc_cilow'] = ci[0]
    res['auc_cihigh'] = ci[1]
    res['auprc'] = auprc
    locs_ppv_nan = np.isnan(cis_ppv[0])
    try:
        res['auprc_cilow'] = auc(tpr[~locs_ppv_nan], cis_ppv[0][~locs_ppv_nan])
    except:
         res['auprc_cilow'] = np.nan
    locs_ppv_nan = np.isnan(cis_ppv[1])
    try:
        res['auprc_cihigh'] = auc(tpr[~locs_ppv_nan], cis_ppv[1][~locs_ppv_nan])
    except:
        res['auprc_cihigh'] = np.nan
    res['cutoff_mcc'] = cutoff_mcc
    res['cutoff_youden'] = cutoff_youden
    res['cutoff_f1'] = cutoff_f1
    res['cutoff_ji'] = cutoff_ji

    max_index = max_index_ji if index == 'ji' else (max_index_mcc if index == 'mcc' else (max_index_f1 if index == 'f1' else max_index_youden))
    print("Max index is:", max_index)

    res['tpr'] = tpr[max_index]
    res['tnr'] = tnr[max_index]
    res['fpr'] = fpr[max_index]
    res['fnr'] = fnr[max_index]
    res['ppv'] = ppv[max_index]
    res['npv'] = npv[max_index]
    res['fscore'] = f1[max_index]
    res['odds'] = dor[max_index]
    res['P'] = P
    res['N'] = N
    # Calcuate TP, FP, TN, FN
    res['TP'] = res['tpr'] * P
    res['TN'] = res['tnr'] * N
    res['FP'] = res['fpr'] * N
    res['FN'] = res['fnr'] * P

    res['precision'] = res['TP']/(res['TP'] + res['FP'])
    res['recall'] = res['TP']/(res['TP'] + res['FN'])

    return res, res_array


def univariable_auc():
    """
    for each column in a dataframe, calculate AUC and other Youden index based metrics
    """
    

 # functions for ROC/PRC plots
def plot_averoc_curve(y_true, y_pred, tp=None, fp=None, color='#FFA500',linestyle='-',label=None,fig=None, ax = None, isLast = False, withCI=False, plot_title=None, fig_name="AUROC_Graph"):
    """
    Plot averaged ROC curves, either through (y_true, y_pred) or through (tp, fp) for a K-fold cross-validation results. 
    Args:
       y_true: np.array of 0 and 1 (n x K). Each column is one fold
       y_pred: np.array of floats of the probability of being class 1 (n x K). Each column is one fold
    """
    print("plot_averoc_curve is called.")
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    if y_true is not None:
        n_fold = len(y_true)
        
        for k in range(n_fold):
            # loc1 = trY.iloc[:,k].isnull()
            # loc2 = pred.iloc[:,k].isnull()
            # locs =np.logical_not(np.logical_or(loc1,loc2))
            # this_trY = trY.iloc[locs, k]
            # this_pred = pred.iloc[locs, k]
            this_true = y_true[k]
            this_pred = y_pred[k]#[:,1]
            
            #st.write(this_true)
            #st.write(type(this_true), this_true.shape)
            #st.write(this_pred)
            #st.write(type(this_pred), this_pred.shape)
            auc_value = roc_auc_score(this_true, this_pred)
            if auc_value < 0.5:
                #this_pred = 1-this_pred
                this_pred = [1 - x for x in this_pred]
            fpr, tpr, thresholds = roc_curve(this_true, this_pred, pos_label=1)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            roc_auc = auc(fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(roc_auc)
    else:
        n_fold = tp.shape[1]
        
        for k in range(n_fold):
            fpr = fp.iloc[:,k].dropna()
            tpr = tp.iloc[:,k].dropna()
            if len(fpr) != len(tpr):
                pass
            else:
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                roc_auc = auc(fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(roc_auc)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.nanstd(aucs)
    std_tpr = np.std(tprs, axis=0)
    se_tpr = std_tpr/np.sqrt(n_fold)
    se_auc = std_auc/np.sqrt(n_fold)
    t_score = stats.t.ppf(1-0.025, n_fold)
    #tprs_upper = np.minimum(mean_tpr + 1.96*std_tpr, 1)
    #tprs_lower = np.maximum(mean_tpr - 1.96*std_tpr, 0)
    tprs_upper = np.minimum(mean_tpr + t_score*se_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - t_score*se_tpr, 0)
    #cilow = np.maximum(mean_auc-1.96*std_auc, 0)
    #cihigh = np.minimum(mean_auc+1.96*std_auc, 1)
    cilow = np.maximum(mean_auc-t_score*se_auc, 0)
    cihigh = np.minimum(mean_auc+t_score*se_auc, 1)
    
    if ax is None:
        fig, ax = plt.subplots()
        isLast = True
    if label is None:
        ax.plot(mean_fpr, mean_tpr, color = color, label=f'AU-ROC: {mean_auc:.2f} CI: [{cilow:.2f}, {cihigh:.2f}]')
        #ax.plot(mean_fpr, mean_tpr, color = color)
    else:
        label = label + ': AUC:' + "%.2f"%mean_auc + ', 95%CI: '+"%.2f"%cilow + '-' + "%.2f"%cihigh
        ax.plot(mean_fpr, mean_tpr, color = color, label=label,linestyle=linestyle)    
    
    if withCI:
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color, alpha=.15)
    
    if isLast:
        # font = font.mgr.FontProperties(family='', size)
        #fig.legend(loc='lower right',bbox_to_anchor=(0.7, 0.15), frameon=False,prop={"family":'Consolas',"size":16}, edgecolor='black')
        fig.legend(loc='upper left', prop={"size":8}, edgecolor='black') 
        ident = [0.0, 1.0]
        ax.plot(ident,ident, color='#C0C0C0', linestyle='dashed')
        ax.set_xlabel('False positive rate', fontsize = 17)
        ax.set_ylabel('True positive rate', fontsize = 17)
    
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_aspect('equal', adjustable='box')
    else:
        fig.legend(loc='upper left', prop={"size":8}, edgecolor='black')
        
    if plot_title == None:
        plt.title("Averaged ROC curves with 95%CI",fontsize=12)   
    else:
        plt.title(plot_title,fontsize=15) 

    # Save the figure to a buffer
    fig = plt.gcf()  # Get current figure
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image_data = buf.read()
        
    # Save the plot, name must end with .png
    plt.savefig(fig_name)
    plt.show()
    st.pyplot(fig)  # Use Streamlit's function to display the plot
    plt.close()

    return np.array([mean_auc, cilow, cihigh]), image_data 
 

def plot_roc_curve(y_true, y_pred, curve_type='roc', color='#ADDFFF',label="",fig=None, ax = None, isLast = False, withci = True, alpha=0.95):
    """
    Plot ROC/PRC curves. 
    Args:
       y_true: np.array of 0 and 1
       y_pred: np.array of floats of the probability of being class 1
       curve_type: roc or prc
       label: name for the classifier that corresponds to this curve
       fig: pass a fig handle, if plotting multiple curves on the same figure
       ax: pass an ax handle, if plotting multiple curved on the same figure
       isLast: False if more curves to be added. True, if current call is for the last curve plotting
       withci: is adding confidence interval to the curve.
       
    Example:
       #matplotlib qt
       import seaborn as sns
       fig, ax = plt.subplots()
       colors = sns.color_palette('tab10', 5)
       plot_roc_curve(y_true, y_pred,curve_type='prc', color='#ADDFFF',label='SI',fig=fig, ax = ax, isLast = True)
    """
    # fig, ax = plt.subplots()  # pass fig and ax to this function before use
    if ax is None:
        fig, ax = plt.subplots()
        isLast = True

    res, res_array = full_roc_curve(y_true, y_pred)

    if curve_type == 'roc':
        xlabel = 'False positive rate'
        ylabel = 'True positive rate'
        specificity = res_array['tnr']
        sensitivity = res_array['tpr']
        auroc = res['auc']
        if np.isnan(res['auc_cilow']):
            res['auc_cilow'] = 1.0
        if np.isnan(res['auc_cihigh']):
            res['auc_cihigh'] = 1.0

        label = label + ': AUC:' + "%.2f"%auroc + ', 95%CI: '+"%.2f"%res['auc_cilow'] + '-' + "%.2f"%res['auc_cihigh']
        ax.plot(1-specificity, sensitivity,color = color, label=label)
    elif curve_type == 'prc':
        xlabel = 'Recall'
        ylabel = 'Precision'
        recall = res_array['tpr']
        precision = res_array['ppv']
        auprc = res['auprc']
        label = label + ': AUC:' + "%.3f"%auprc + ', 95%CI: '+"%.3f"%res['auprc_cilow'] + '-' + "%.3f"%res['auprc_cihigh']
        ax.plot(recall, precision,color = color, label=label)
        
    if withci:
        #z = -ndtri((1.0-alpha)/2)
        #TP = res_array['tpr'] * res['P']
        #n = res['P']
        #cis = proportion_confidence_interval(TP, n, z) # se low, se median, se high
        #ax.fill_between(1-specificity, cis[0], cis[1], color=color, alpha=.29)
        if curve_type=='roc':
            ax.fill_between(1-specificity, res_array['tpr_low'], res_array['tpr_high'], color=color, alpha=.2)
        elif curve_type=='prc':
            ax.fill_between(recall, res_array['ppv_low'], res_array['ppv_high'], color=color, alpha=.2)

    if isLast:
        # font = font.mgr.FontProperties(family='', size)
        fig.legend(loc='lower right',bbox_to_anchor=(0.765, 0.20), frameon=False,prop={"family":'Consolas',"size":23})
        ident = [0.0, 1.0]
        ax.plot(ident,ident, color='#C0C0C0', linestyle='dashed')
        ax.set_xlabel(xlabel, fontsize = 28)
        ax.set_ylabel(ylabel, fontsize = 28)
        ax.tick_params(axis='x', labelsize= 24)
        ax.tick_params(axis='y', labelsize= 24)
    
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_aspect('equal', adjustable='box')
        fig.tight_layout()


def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks
   
    rows1 = np.sum(cf, axis=1)
    cf_pct = (cf / rows1[:, np.newaxis])
   
    if percent:
       
        group_percentages = ["{0:.2%}".format(value) for value in cf_pct.flatten()]#["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks
    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize is None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf_pct,annot=box_labels,annot_kws={"size": 16},fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label', fontsize=26)
        plt.xlabel('Predicted label' + stats_text, fontsize=26)
    else:
        plt.xlabel(stats_text)
       
    if xyticks:
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
   
    if title:
        plt.title(title)


# Decision Curve Analysis

def plot_dca(y_true, y_pred, fig,ax, color='crimson', label='Model', isfill=True, islast=False):
    def _binary_clf_curve(y_true, y_prob):
        y_true = np.ravel(y_true)
        y_prob = np.ravel(y_prob)
        pos_label = 1
        y_true = y_true == pos_label

        desc_prob_indices = np.argsort(y_prob, kind="mergesort")[::-1]
        y_prob = y_prob[desc_prob_indices]
        y_true = y_true[desc_prob_indices]

        distinct_value_indices = np.where(np.diff(y_prob))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size -1]

        tps = stable_cumsum(y_true)[threshold_idxs]
        fps = 1 + threshold_idxs - tps
        thresholds = y_prob[threshold_idxs]

        return fps, tps, thresholds

    def _calculate_net_benefit_model_none(y_true, y_prob, n_points=20000):
        fps, tps, thresholds = _binary_clf_curve(y_true, y_prob)
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        thresholds = np.r_[max(thresholds[0], 1-1e-10), thresholds]

        n = y_true.size
        sort_indices = np.argsort(thresholds, kind="mergesort")
        thresholds = thresholds[sort_indices]
        tps = tps[sort_indices]
        fps = fps[sort_indices]

        interp_thresholds = np.linspace(0, 1-1/n_points, n_points)
        binids = np.searchsorted(thresholds[:-1], interp_thresholds)
        net_benefits = (tps[binids] / n) - (fps[binids] / n) * (interp_thresholds /  (1 - interp_thresholds))

        return net_benefits, interp_thresholds 
    
    def _calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
        net_benefit_model = np.array([])
        for thresh in thresh_group:
            y_pred_label = y_pred_score > thresh
            tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
            n = len(y_label)
            net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
            net_benefit_model = np.append(net_benefit_model, net_benefit)
        return net_benefit_model

    def _calculate_net_benefit_all(thresh_group, y_label):
        net_benefit_all = np.array([])
        tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
        total = tp + tn
        for thresh in thresh_group:
            net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
            net_benefit_all = np.append(net_benefit_all, net_benefit)
        return net_benefit_all
    
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    
    #net_benefit_model, thresh_group = _calculate_net_benefit_model(y_pred, y_true)
    #net_benefit_all = _calculate_net_benefit_all(thresh_group, y_true)
    thresh_group = np.arange(0,1,0.001)
    net_benefit_model = _calculate_net_benefit_model(thresh_group, y_pred, y_true)
    net_benefit_all = _calculate_net_benefit_all(thresh_group, y_true)
    ax.plot(thresh_group, net_benefit_model, color=color, label=label)
    
    
    ax.plot(thresh_group, net_benefit_all, color = '#2554C7',label = 'Treat all')
    ax.plot((0, 1), (0, 0), color = 'black', linestyle = '--', label = 'Treat none')

    

    if isfill:
        #Fill，highligt the better part of the model, compared with treat all and treat none
        y2 = np.maximum(net_benefit_all, 0)
        y1 = np.maximum(net_benefit_model, y2)
        ax.fill_between(thresh_group, y1, y2, color = color, alpha = 0.1)

    if islast:
        ax.set_xlim(0,1)
        ax.set_ylim(net_benefit_model.min() - 0.15, net_benefit_model.max() + 0.15)#adjustify the y axis limitation
        ax.set_xlabel(
            xlabel = 'Threshold Probability', 
            fontdict= {'family': 'Times New Roman', 'fontsize': 20}
            )
        ax.set_ylabel(
            ylabel = 'Net Benefit', 
            fontdict= {'family': 'Times New Roman', 'fontsize': 20}
            )
        ax.tick_params(axis='x', labelsize= 18)
        ax.tick_params(axis='y', labelsize= 18)
        ax.grid('major')
        ax.spines['right'].set_color((0.8, 0.8, 0.8))
        ax.spines['top'].set_color((0.8, 0.8, 0.8))
        ax.legend(loc = 'upper right',frameon=False,prop={"family":'Times New Roman',"size":17})

    return ax


from sklearn.metrics import confusion_matrix





def plot_DCA(y_pred_score, y_label, ax=None):
    
    def _calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
        net_benefit_model = np.array([])
        for thresh in thresh_group:
            y_pred_label = y_pred_score > thresh
            tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
            n = len(y_label)
            net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
            net_benefit_model = np.append(net_benefit_model, net_benefit)
        return net_benefit_model


    def _calculate_net_benefit_all(thresh_group, y_label):
        net_benefit_all = np.array([])
        tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
        total = tp + tn
        for thresh in thresh_group:
            net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
            net_benefit_all = np.append(net_benefit_all, net_benefit)
        return net_benefit_all

    thresh_group = np.arange(0,1,0.001)
    net_benefit_model = _calculate_net_benefit_model(thresh_group, y_pred_score, y_label)
    net_benefit_all = _calculate_net_benefit_all(thresh_group, y_label)
    #Plot
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(thresh_group, net_benefit_model, color = 'crimson', label = 'Model')
    ax.plot(thresh_group, net_benefit_all, color = 'black',label = 'Treat all')
    ax.plot((0, 1), (0, 0), color = 'black', linestyle = ':', label = 'Treat none')

    #Fill，显示出模型较于treat all和treat none好的部分
    y2 = np.maximum(net_benefit_all, 0)
    y1 = np.maximum(net_benefit_model, y2)
    ax.fill_between(thresh_group, y1, y2, color = 'crimson', alpha = 0.2)

    #Figure Configuration
    ax.set_xlim(0,1)
    ax.set_ylim(net_benefit_model.min() - 0.05, net_benefit_model.max() + 0.05)#adjustify the y axis limitation
    ax.set_xlabel(
        xlabel = 'Threshold Probability', 
        fontdict= {'family': 'Times New Roman', 'fontsize': 15}
        )
    ax.set_ylabel(
        ylabel = 'Net Benefit', 
        fontdict= {'family': 'Times New Roman', 'fontsize': 15}
        )
    ax.grid('major')
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))
    ax.legend(loc = 'upper right',fontsize=15)

    return ax


def validation_hardness():
    for cnt in range(0,100):
                res = joblib.load(open('D:/projects/poct/models2023/out'+str(oidx)+'sheet'+str(dindex)+'exp'+str(eindex)+'cv'+str(cnt)+'_xgb.p', 'rb'))
                # thisTrY.append(outs[outnames[oidx]][res[4]])
                #thisTrY.append(tmp.iloc[res[4]])
                #thisTrPred.append(res[2])
                # thisTeY.append(outs[outnames[oidx]][res[5]])
                thisTeY.append(y.iloc[res[5]].reset_index(drop=True))
                thisTePred.append(res[3][:,1])
                #thisTeY.append(y.iloc[res[4]].reset_index(drop=True))
                #thisTePred.append(res[2][:,1])


# some helper functions to simplify the work
def plot_cv_files(filelist, labels, colors='red',fig=None, ax=None,islast = True):
    ytrues = {}
    ypreds = {}
    if ax is None:
        fig, ax = plt.subplots()
    for cnt in range(len(filelist)):
        # if file exists
        with open(filelist[cnt], 'rb') as f:
            dl = pickle.load(f)

            ytrues[cnt] = dl[0]
            ypreds[cnt] = dl[1][:,1]
    plot_averoc_curve(ytrues, ypreds, label=labels,color=colors,fig=fig, ax = ax, isLast = islast, withCI=True)


def plot_multiclass_roc(y_test, y_score, n_classes, figsize=(17, 17)):

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    colors = sns.color_palette("light:#5A9", n_colors=13)
    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score.iloc[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    
    color_idx = np.argsort(list(roc_auc.values()))
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_dummies.ravel(), y_score.values.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=25)
    ax.set_ylabel('True Positive Rate', fontsize=25)
    ax.set_title('Receiver operating characteristic')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], color=colors[color_idx[i]], label='ROC curve (area = %0.2f) for class=%i' % (roc_auc[i], i))
    
    #plt.plot(
    #fpr["micro"],
    #tpr["micro"],
    #label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
    #color="deeppink",
    #linestyle=":",
    #linewidth=4)

    plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
    color="deeppink",
    linestyle=":",
    linewidth=4)
    
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    ax.tick_params(axis='x', labelsize= 20)
    ax.tick_params(axis='y', labelsize= 20)
    sns.despine()
    plt.show()