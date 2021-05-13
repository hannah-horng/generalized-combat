import pandas as pd
import neuroCombat as nC
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from scipy.stats import ranksums, ttest_ind, ttest_rel, ks_2samp
import os
import NestedComBat as nested
import GMMComBat as gmmc


# Loading in features
filepath = "C:/Users/horng/OneDrive/CBIG/Stanford/"
filepath2 = "C:/Users/horng/OneDrive/CBIG/Function_Testing/Stanford/"

# Loading in batch effects
batch_df = pd.read_excel('C:/Users/horng/OneDrive/CBIG/Stanford/Sandy-CT-parameters.xlsx')
batch_df = batch_df[batch_df['Manufacturer'] != 'Philips'].reset_index(drop=True)  # Dropping Phillips Case
batch_df = batch_df[batch_df['Manufacturer'] != 'TOSHIBA'].reset_index(drop=True)  # Dropping Toshiba Case
batch_df = batch_df[batch_df['Manufacturer'] != 0].reset_index(drop=True)  # Dropping 0 Case
batch_df['clc'] = batch_df['clc'].apply(lambda x: x[:-1] if len(x) > 7 else x)

# batch_df = batch_df[batch_df['ID'].isin(caseno)].reset_index(drop=True)
batch_list = ['Manufacturer', 'KernelResolution', 'CE']

# Loading in clinical covariates
covars_df = pd.read_csv('C:/Users/horng/OneDrive/CBIG/Stanford/clinical-demographics.csv')
categorical_cols = ['event', 'sex', 'smoking', 'histology']
continuous_cols = ['days']

# CAPTK
data_df = pd.read_csv(filepath+'Sandy-Captk.csv')
data_df = data_df.reset_index(drop=True)
data_df = data_df.dropna()
data_df = data_df.rename(columns={"SubjectID": "Case"})
data_df = data_df[data_df['Case'] != 'R01-159'].reset_index(drop=True)  # Missing covariates
data_df = data_df.merge(batch_df['clc'], left_on='Case', right_on='clc')
dat = data_df.iloc[:, 1:-1]
dat = dat.T.apply(pd.to_numeric)
caseno = data_df['Case'].str.upper()

# Merging batch effects, clinical covariates
batch_df = data_df[['Case']].merge(batch_df, left_on='Case', right_on='clc')
covars_df = data_df[['Case']].merge(covars_df, left_on='Case', right_on='Unnamed: 0')
covars_string = pd.DataFrame()
covars_string[categorical_cols] = covars_df[categorical_cols].copy()
covars_string[batch_list] = batch_df[batch_list].copy()
covars_quant = covars_df[continuous_cols]

covars_cat = pd.DataFrame()
for col in covars_string:
    stringcol = covars_string[col]
    le = LabelEncoder()
    le.fit(list(stringcol))
    covars_cat[col] = le.transform(stringcol)

covars = pd.concat([covars_cat, covars_quant], axis=1)

# output_data = nested.NestedComBat(dat, covars, batch_list, categorical_cols=categorical_cols,
#                                   continuous_cols=continuous_cols, drop=True, write_p=True, filepath=filepath2)
output_data = gmmc.GMMComBat(dat, caseno, covars,  filepath=filepath2, categorical_cols=categorical_cols,
                             continuous_cols=continuous_cols, write_p=True, plotting=True)

# output_df = pd.DataFrame.from_records(output_data.T)
# output_df.columns = feature_cols
write_df = pd.concat([caseno, output_data], axis=1)
write_df.to_csv(filepath2+'GMM_harmonized_features.csv')

gmmc.feature_kstest_histograms(output_data, covars, batch_list, filepath2)

f_dict = gmmc.MultiComBat(output_data.T, covars, batch_list, filepath=filepath2, categorical_cols=categorical_cols,
                          continuous_cols=continuous_cols, write_p=True, plotting=True)
for col in batch_list:
    write_df = pd.concat([caseno, f_dict[col]])
    write_df.to_csv(filepath2+'GMM_'+col+'_harmonized_features.csv')


