# import libraries
import os
import pandas as pd
import numpy as np
import pickle
# %% import data
par_dir = os.getcwd()

# get list of cmm and wp datasets
wp_datasets = os.listdir(os.path.join(par_dir, "sliced_workpieces"))
# create cmm list based on wp list --> maintain same order
cmm_datasets = ["".join(["CMM_", x]) for x in wp_datasets]

# %% read in wp datasets

wp_temp = (pd.read_csv(os.path.join(par_dir, "sliced_workpieces", pocket)) for pocket in wp_datasets)
wp_data = pd.concat(wp_temp, ignore_index=True)

cmm_temp = (pd.read_csv(os.path.join(par_dir, "cmm_data", pocket)) for pocket in cmm_datasets)
cmm_data = pd.concat(cmm_temp, ignore_index=True)

# %% split message into pos in wpnr
msgColumnName = "/Channel/ProgramInfo/msg|u1"

def msgInfoExtractor(msg_string):
    infosDict = {
        "WPNR": msg_string[5],
        "xPos": msg_string[12],
        "yPos": msg_string[19],
        "EC": msg_string[-2:]
    }
    return infosDict

wp_data['WPNR'] = [msgInfoExtractor(row)["WPNR"] for row in wp_data[msgColumnName]]
wp_data['xPos'] = [msgInfoExtractor(row)["xPos"] for row in wp_data[msgColumnName]]
wp_data['yPos'] = [msgInfoExtractor(row)["yPos"] for row in wp_data[msgColumnName]]
wp_data['EC'] = [msgInfoExtractor(row)["EC"] for row in wp_data[msgColumnName]]

cmm_data['WPNR'] = [msgInfoExtractor(row)["WPNR"] for row in cmm_data["MESSAGE"]]
cmm_data['xPos'] = [msgInfoExtractor(row)["xPos"] for row in cmm_data["MESSAGE"]]
cmm_data['yPos'] = [msgInfoExtractor(row)["yPos"] for row in cmm_data["MESSAGE"]]
cmm_data['EC'] = [msgInfoExtractor(row)["EC"] for row in cmm_data["MESSAGE"]]

# %% organize data in dictionairies
data_gF = {}
cmm_gF = {}
# get all possible geomFeatures in List
geomFeatures = list(dict.fromkeys(wp_data["geomFeature"]))
for feature in geomFeatures:
    data_gF[feature] = wp_data[wp_data["geomFeature"] == feature]#.reset_index()
    cmm_gF[feature] = cmm_data.loc[cmm_data["geomFeature"].str.contains(feature)]


# %%
data_gF_grouped = {}
# group data by "WPNR", "xPos" and "yPos"
for feature in geomFeatures:
    data_gF_grouped[feature] = data_gF[feature].groupby(["WPNR", "xPos", "yPos", "geomFeature", msgColumnName]).describe()
    # join column tuples
    data_gF_grouped[feature].columns = ["_".join(col) for col in data_gF_grouped[feature].columns]
    # remove rows with percentiles or count
    data_gF_grouped[feature] = data_gF_grouped[feature][[col for col in data_gF_grouped[feature].columns if not any(s in col for s in ["%", "count"])]]
    # reset
    data_gF_grouped[feature] = data_gF_grouped[feature].reset_index()


# %% Data preprocessing for each geomFeature
corr_values = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
drop_signals_dict = {}
for val in corr_values:
    drop_signals_dict[val] = {}
    for feature in data_gF.keys():
        # correlation matrix
        corr_matrix = data_gF[feature].corr()

        # get upper triangle of correlation matrix
        triu = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        corr_matrix_triu = corr_matrix.where(triu)

        # drop column with correlation value higher 0.95
        drop_signals_corr = [col for col in corr_matrix_triu.columns if any(corr_matrix_triu[col] > val)]

        # drop column if "DPIO" in column name
        drop_signals_dpio = [col for col in data_gF[feature].columns if "DPIO" in col]

        # drop column if signal is constant (std == 0)
        # get standard deviation of each column
        data_drop_std = data_gF[feature].groupby("geomFeature").describe()
        data_drop_std.columns = ["_".join(col) for col in data_drop_std.columns]
        data_drop_std = data_drop_std[[col for col in data_drop_std.columns if "std" in col]]
        # drop column with std == 0, get rid of "_std" in column name
        drop_signals_std = [col[:-4] for col in data_drop_std.columns if any(data_drop_std[col] == 0)]

        # list of all signals to drop
        drop_signals = [*drop_signals_corr, *drop_signals_dpio, *drop_signals_std]

        # drop signals in both dataframes
        # data_gF[feature].drop(drop_signals, axis=1, inplace=True)
        # data_gF_grouped[feature].drop([col for col in data_gF_grouped[feature].columns
        #                                if any(sig in col for sig in drop_signals)], axis=1, inplace=True)

        drop_signals_dict[val][feature] = drop_signals

# %%
msgColumnName = "/Channel/ProgramInfo/msg|u1"

# create empty dict to save converted data
all_data_gF = {}

# list of all pockets (from message column)
pocket_list = list(dict.fromkeys(data_gF['LIN_1'][msgColumnName]))

# iterate through geomFeatures
for ii, feature in enumerate(data_gF.keys()):
    all_data_gF[feature] = []
    data_temp = all_data_gF
    for jj, pocket in enumerate(pocket_list):
        # print("pocket", jj)
        cmm_pocket = cmm_gF[feature][cmm_gF[feature]["MESSAGE"] == pocket].copy()
        # data_gF_grouped oder data_gF, je nachdem was man mit cmm Daten verbinden will
        wp_pocket = data_gF_grouped[feature][data_gF_grouped[feature][msgColumnName] == pocket].copy()

        # write every measurement to a new column
        for idx, rows in cmm_pocket.iterrows():
            # print("TOL", idx)
            pd.Series(rows["ABW"], index=wp_pocket.index)
            wp_pocket.loc[:, "_".join([rows["TOLNAME"], "ABW"])] = rows["ABW"]
            # wp_pocket.loc[:, "_".join([rows["TOLNAME"], "AUSTOL"])] = rows["AUSTOL"]

        # write data for pocket (including quality data) to dict
        data_temp[feature].append(wp_pocket)
    # concatenate dataframes for each pocket
    all_data_gF[feature] = pd.concat(data_temp[feature])


# %% save data
# make new folder
if os.path.isdir(os.path.join(par_dir, "save_data")) is False:
    os.mkdir(os.path.join(par_dir, "save_data"))

# save as pickle
with open(os.path.join(par_dir, "save_data", 'data_gF.pickle'), 'wb') as handle:
    pickle.dump(data_gF, handle)
with open(os.path.join(par_dir, "save_data", 'data_gF_grouped.pickle'), 'wb') as handle:
    pickle.dump(data_gF_grouped, handle)
with open(os.path.join(par_dir, "save_data", 'cmm_gF.pickle'), 'wb') as handle:
    pickle.dump(cmm_gF, handle)
with open(os.path.join(par_dir, "save_data", 'all_data_gF.pickle'), 'wb') as handle:
    pickle.dump(all_data_gF, handle)
with open(os.path.join(par_dir, "save_data", 'drop_signals_dict.pickle'), 'wb') as handle:
    pickle.dump(drop_signals_dict, handle)

