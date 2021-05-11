# Written by Hannah Horng (hhorng@seas.upenn.edu)
import pandas as pd
import neuroCombat as nC
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from scipy.stats import ranksums, ttest_ind, ttest_rel, ks_2samp
import os


def NestedComBat(dat, covars, batch_list, categorical_cols=None, continuous_cols=None, drop=False,
                 write_p=False, plotting=False, filepath=''):
    """
    Completes sequential nested ComBat harmonization on an input DataFrame. Order is determined by number of features
    with statistically significant differences in distribution (KS test) due to a particular batch effect.

    Arguments
    ---------
    data : DataFrame of original data with shape (features, samples)
    covars : DataFrame with shape (samples, covariates) corresponding to original data. All variables should be label-
        encoded (i.e. strings converted to integer designations)
    batch_list : list of strings indicating batch effect column names within covars (i.e. ['Manufacturer', 'CE'...])
    categorical_cols : string or list of strings of categorical variables to adjust for
    continuous_cols : string or list of strings of continuous variables to adjust for
    drop : Boolean, if True -- features with significant differences in distribution due to the batch effect being
        harmonized are dropped with each iteration (corresponds to NestedD)
    write_p : Boolean, if True -- KS test p-values will be written as a CSV into the directory created from filepath
    plotting : Boolean, if True -- kernel density plots will be written as image files into the directory created from
        filepath
    filepath : root directory path for saving KS test p-values and kernel density plots created during harmonization

    Returns
    -------
    new_dat : DataFrame with shape (features, samples) that has been sequentially harmonized with Nested ComBat

    """
    p_dict = {}
    count_dict = {}
    f_dict = {}
    print('ROUND 1:')
    for a in range(len(batch_list)):
        batch_col = batch_list[a]
        print('Harmonizing by ' + batch_col + '...')

        filepath2 = filepath + 'Round 1/' + batch_col + '/'
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

    drop_feature = [key for key, value in count_dict.items() if value == min(count_dict.values())][0]

    # Iteration
    batch_list2 = batch_list.copy()
    batch_list2.remove(drop_feature)

    new_data_df = f_dict[drop_feature]
    new_pvalues = p_dict[drop_feature]
    new_dat = new_data_df.T
    if drop:
        new_dat = new_data_df.T[new_pvalues > .05]  # Dropping every iteration

    c = 1
    while len(batch_list2) > 0:
        print('ROUND ' + str(c+1) + ':')
        p_dict = {}
        count_dict = {}
        f_dict = {}
        c = c+1
        for b in range(len(batch_list2)):
            batch_col = batch_list2[b]
            print('Harmonizing by ' + batch_col + '...')
            filepath2 = filepath+'Round '+str(c) + '/' + batch_col+'/'
            if not os.path.exists(filepath2):
                os.makedirs(filepath2)

            # RUN COMBAT
            # print('ComBat with Raw Data...')
            output = nC.neuroCombat(new_dat, covars, batch_col, continuous_cols=continuous_cols, categorical_cols=categorical_cols)['data']
            output_df = pd.DataFrame.from_records(output.T)
            output_df.columns = new_dat.T.columns
            f_dict[batch_col] = output_df

            if plotting:
                combat_histograms(new_dat.T, output_df, covars, covars, batch_col, filepath2)

            if write_p:
                p_values = combat_kstest(new_dat.T, output_df, covars, covars, batch_col, write=True, filepath=filepath2)
            else:
                p_values = combat_kstest(new_dat.T, output_df, covars, covars, batch_col)
            p_values.index = output_df.columns
            p_dict[batch_col] = p_values['ComBat']
            count_dict[batch_col] = len(p_values[p_values['ComBat'] < .05])

        drop_feature = [key for key, value in count_dict.items() if value == min(count_dict.values())][0]
        new_data_df = f_dict[drop_feature]
        new_pvalues = p_dict[drop_feature]

        if drop:
            new_dat = new_data_df.T[new_pvalues > .05]  # Iteration + Dropping
        else:
            new_dat = new_data_df.T
        batch_list2.remove(drop_feature)

    output_df = pd.DataFrame.from_records(new_dat.T)
    output_df.columns = new_dat.T.columns
    return output_df


def combat_kstest(data, output, covars1, covars2, batch_col, filepath='', write=False):
    """
    Calculating KS test for differences in distribution due to batch effect before and after harmonization
    *Note that this is differs from the version in GMMComBat only by file destination naming

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
    NestedComBat

    *Note that this is differs from the version in GMMComBat only by file destination naming

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
        filepath2 = filepath + 'feature_histograms/' + batch_col + '/'
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
    p_df.to_csv(filepath + 'final_nested_ks_values.csv')


