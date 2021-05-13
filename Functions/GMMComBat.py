import pandas as pd
import neuroCombat as nC
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from scipy.stats import ranksums, ttest_ind, ttest_rel, ks_2samp
import os
from sklearn.mixture import GaussianMixture
import numpy as np


def GMMComBat(dat, caseno, covars, filepath, categorical_cols=None, continuous_cols=None, write_p=False,
              plotting=False):
    """
    Completes Gaussian Mixture model fitting and ComBat harmonization by the resulting sample grouping. The assumption
    here is that there is an unknown batch effect causing bimodality such that we can estimate the sample groupings for
    this hidden batch effect from the distribution. This function will take in a dataset, determine the best 2-component
    Gaussian mixture model, and use the resulting sample grouping to harmonize the data with ComBat.

    Arguments
    ---------
    dat : DataFrame of original data with shape (features, samples)
    caseno : DataFrame/Series containing sample IDs (should be aligned with dat and covars), used to return sample
        grouping assignments.
    covars : DataFrame with shape (samples, covariates) corresponding to original data. All variables should be label-
        encoded (i.e. strings converted to integer designations)
    filepath : root directory path for saving KS test p-values and kernel density plots created during harmonization
    categorical_cols : string or list of strings of categorical variables to adjust for
    continuous_cols : string or list of strings of continuous variables to adjust for
    write_p : Boolean, if True -- KS test p-values will be written as a CSV into the directory created from filepath
    plotting : Boolean, if True -- kernel density plots will be written as image files into the directory created from
        filepath

    Returns
    -------
    new_dat : DataFrame with shape (features, samples) that has been sequentially harmonized with Nested ComBat

    """
    # GENERATING GMM GROUPING
    data_keys = list(dat.T.keys())
    aic_values = []
    predictions = pd.DataFrame()
    col_list = []
    filepath2 = filepath+'1_GMM_Split/'
    if not os.path.exists(filepath2):
        os.makedirs(filepath2)

    for col in data_keys:
        # print(col)
        gmix = GaussianMixture(n_components=2)
        X = dat.T[[col, col]]
        try:
            gmix.fit(X)
            results = gmix.predict(X)
            cluster_0 = X[results == 0].iloc[:, 0]
            cluster_1 = X[results == 1].iloc[:, 0]
            # print(len(cluster_0))
            if len(cluster_0) <= .25*len(caseno) or len(cluster_1) <= .25*len(caseno):
                print('Clusters unbalanced: ' + col)
            else:
                try:
                    plt.figure()
                    cluster_0.plot.kde()
                    cluster_1.plot.kde()
                    X.iloc[:, 0].plot.kde()
                    plt.legend(['Cluster 0', 'Cluster 1', 'Original'])
                    plt.xlabel(col)
                    filename = filepath2 + 'histogram_' + col + ".png"
                    plt.savefig(filename, bbox_inches='tight')
                    plt.close()
                except:
                    plt.close()
                    print('Failed to plot: ' + col)
                predictions[col] = results
                aic_values.append(gmix.aic(X))
                col_list.append(col)
        except ValueError:
            print('Failed to fit: ' + col)
            # aic_values.append(np.nan)

    # Returning AIC values
    gaussian_df = pd.DataFrame({'Feature': predictions.keys(), 'AIC': aic_values})
    best_fit = gaussian_df[gaussian_df['AIC'] == min(gaussian_df['AIC'])]['Feature'].iloc[0].strip(' ')
    gaussian_df.to_csv(filepath2 + 'GaussianMixture_aic_values.csv')

    # Returning patient split
    predictions_df = pd.DataFrame()
    predictions_df['Patient'] = caseno
    predictions_df['Grouping'] = predictions[best_fit]
    predictions_df.to_csv(filepath2 + best_fit + '_split.csv')

    # HARMONIZATION BY PATIENT SPLIT VARIABLE
    print('Correcting for arbitrary patient split...')
    split_df = predictions_df.copy()
    split_col = 'Grouping'
    split_df['Patient'] = split_df['Patient'].str.upper()
    split_df = split_df[split_df['Patient'].isin(caseno)].reset_index(drop=True)
    covars[split_col] = split_df[split_col]
    output = nC.neuroCombat(dat, covars, split_col, continuous_cols=continuous_cols, categorical_cols=categorical_cols)['data']
    output_df = pd.DataFrame.from_records(output.T)
    output_df.columns = dat.T.columns
    filepath_x = filepath + '2_GMM_Harmonization/Grouping/'

    if not os.path.exists(filepath_x):
        os.makedirs(filepath_x)
    if plotting:
        combat_histograms(dat.T, output_df, covars, covars, split_col, filepath_x)

    if write_p:
        p_values = combat_kstest(dat.T, output_df, covars, covars, split_col, write=True, filepath=filepath2)
    else:
        p_values = combat_kstest(dat.T, output_df, covars, covars, split_col)

    return output_df


def combat_kstest(data, output, covars1, covars2, batch_col, filepath='', write=False):
    """
    Calculating KS test for differences in distribution due to batch effect before and after harmonization.

    *Note that this is differs from the version in NestedComBat only by file destination naming

    Arguments
    ---------
    data : DataFrame of original data with shape (samples, features)
    output: DataFrame of harmonized data with shape (samples, features)
    covars1 : DataFrame with shape (samples, covariates) corresponding to original data
    covars2 : DataFrame with shape (samples, covariates) corresponding to harmonized data
    batch_col : string indicating batch/imaging parameter name in covars
    filepath : write destination for ks p-value DataFrame if write is True
    write: Boolean, set to True to save ks p-value DataFrame

    Returns
    -------
    p_df : DataFrame with two colums corresponding to KS test p-value testing for significant differences in
           distribution attributable to the batch effect specified by batch_col
    """
    data_keys = data.keys()
    batch_var1 = covars1[batch_col]
    batch_var2 = covars2[batch_col]
    data_0 = data[batch_var1 == 0]
    data_1 = data[batch_var1 == 1]
    output_0 = output[batch_var2 == 0]
    output_1 = output[batch_var2 == 1]

    # KS Test (more generalized differences in distribution)
    p_before = []
    p_after = []
    for m in range(0, data.shape[1]):
        p_value1 = ks_2samp(data_0.iloc[:, m], data_1.iloc[:, m])
        p_value2 = ks_2samp(output_0.iloc[:, m], output_1.iloc[:, m])
        p_before.append(p_value1.pvalue)
        p_after.append(p_value2.pvalue)
    p_df = pd.DataFrame({'Raw': p_before, 'ComBat': p_after})

    if write:
        p_df = pd.DataFrame({'Raw': p_before, 'ComBat': p_after})
        p_df.index = data_keys
        p_df.to_csv(filepath + '_' + batch_col + '_feature_ks_values.csv')
    return p_df


def combat_histograms(data, output, covars1, covars2, batch_col, filepath):
    """
    Plots kernel density plots separated by batch effect groups and before vs. after ComBat harmonization

    Arguments
    ---------
    data : DataFrame of original data with shape (samples, features)
    output: DataFrame of harmonized data with shape (samples, features)
    covars1 : DataFrame with shape (samples, covariates) corresponding to original data
    covars2 : DataFrame with shape (samples, covariates) corresponding to harmonized data
    batch_col : string indicating batch/imaging parameter name in covars
    filepath : write destination for kernel density plots

    """

    print('Plotting histograms...')
    data_keys = data.keys()
    batch_var1 = covars1[batch_col]
    batch_var2 = covars2[batch_col]
    data_0 = data[batch_var1 == 0]
    data_1 = data[batch_var1 == 1]
    output_0 = output[batch_var2 == 0]
    output_1 = output[batch_var2 == 1]
    for k in range(0, data.shape[1]):
        plt.figure()
        data_0.iloc[:, k].plot.kde()
        data_1.iloc[:, k].plot.kde()
        output_0.iloc[:, k].plot.kde()
        output_1.iloc[:, k].plot.kde()
        plt.xlabel(data_keys[k])
        leg = ["0", "1", "0_ComBat", "1_ComBat"]
        plt.legend(leg, loc='upper right')
        plt.rcParams.update({'font.size': 12})
        filename = filepath + batch_col + '_' + 'histogram_' + data_keys[k] + ".png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()


def feature_kstest_histograms(dat, covars, batch_list, filepath):
    """
    Plots kernel density plots and computes KS test p-values separated by batch effect groups for a dataset (intended
    to assess differences in distribution to all batch effects in batch_list following harmonization with
    GMMComBat

    *Note that this is differs from the version in NestedComBat only by file destination naming

    Arguments
    ---------
    data : DataFrame of original data with shape (samples, features)
    output: DataFrame of harmonized data with shape (samples, features)
    covars : DataFrame with shape (samples, covariates) corresponding to original data. All variables should be label-
        encoded (i.e. strings converted to integer designations)
    batch_list : list of strings indicating batch effect column names within covars (i.e. ['Manufacturer', 'CE'...])
    filepath : write destination for kernel density plots

    """
    print('Plotting final feature histograms...')
    p_df = pd.DataFrame()
    for batch_col in batch_list:
        p = []
        split_col = covars[batch_col]
        filepath2 = filepath + '2_GMM_Harmonization/' + batch_col + '/'
        if not os.path.exists(filepath2):
            os.makedirs(filepath2)

        for feature in dat:
            plt.figure()
            dat[feature][split_col == 0].plot.kde()
            dat[feature][split_col == 1].plot.kde()
            plt.xlabel(feature)
            filename = filepath2 + feature + '.png'
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
            p_value = ks_2samp(dat[feature][split_col == 0], dat[feature][split_col == 1])
            p.append(p_value.pvalue)
        p_df[batch_col] = p

    p_df.index = dat.keys()
    p_df.to_csv(filepath2 + 'final_GMMSplit_ks_values.csv')


def MultiComBat(dat, covars, batch_list, categorical_cols=None, continuous_cols=None,
                 write_p=False, plotting=False, filepath=''):
    """
    Iterates through a list of batch effects, harmonizing by a batch effect with ComBat at each iteration. Used after
    harmonizing by GMMComBat to remove differences due to known batch effects. Equivalent to running a single iteration
    of NestedComBat.

    Arguments
    ---------
    data : DataFrame of original data with shape (features, samples)
    covars : DataFrame with shape (samples, covariates) corresponding to original data. All variables should be label-
        encoded (i.e. strings converted to integer designations)
    batch_list : list of strings indicating batch effect column names within covars (i.e. ['Manufacturer', 'CE'...])
    categorical_cols : string or list of strings of categorical variables to adjust for
    continuous_cols : string or list of strings of continuous variables to adjust for
    write_p : Boolean, if True -- KS test p-values will be written as a CSV into the directory created from filepath
    plotting : Boolean, if True -- kernel density plots will be written as image files into the directory created from
        filepath
    filepath : root directory path for saving KS test p-values and kernel density plots created during harmonization

    Returns
    -------
    f_dict : dictionary of DataFrames with shape (samples, features) with structure ['batch effect': feature DataFrame].

    """
    p_dict = {}
    count_dict = {}
    f_dict = {}
    print('Harmonizing by batch effects...')
    for a in range(len(batch_list)):
        batch_col = batch_list[a]
        print('Harmonizing by ' + batch_col + '...')

        filepath2 = filepath + '3_GMM_Batch/' + batch_col + '/'
        if not os.path.exists(filepath2):
            os.makedirs(filepath2)

        # RUN COMBAT
        print('ComBat with Raw Data...')
        output = nC.neuroCombat(dat, covars, batch_col, continuous_cols=continuous_cols,
                                categorical_cols=categorical_cols)['data']
        output_df = pd.DataFrame.from_records(output.T)
        output_df.columns = dat.T.columns
        f_dict[batch_col] = output_df

        if plotting:
            combat_histograms(dat.T, output_df, covars, covars, batch_col, filepath2)

        if write_p:
            p_values = combat_kstest(dat.T, output_df, covars, covars, batch_col, write=True, filepath=filepath2)
        else:
            p_values = combat_kstest(dat.T, output_df, covars, covars, batch_col)
        p_values.index = output_df.columns
        p_dict[batch_col] = p_values['ComBat']
        count_dict[batch_col] = len(p_values[p_values['ComBat'] < .05])

    return f_dict

