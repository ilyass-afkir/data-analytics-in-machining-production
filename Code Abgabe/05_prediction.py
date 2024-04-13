# import libraries
import os
import pandas as pd
import pickle
from Functions import get_data, check_tolerance
# %% import data
par_dir = os.getcwd()

# get list of cmm and wp datasets
wp_datasets = os.listdir(os.path.join(par_dir, "test_workpieces"))

model_name = "kNeighbor_corrvalue0.7.pickle"

with open(os.path.join(par_dir, "save_data", 'drop_signals_dict.pickle'), 'rb') as handle:
    drop_signal_dict = pickle.load(handle)
# %% read in wp datasets

wp_temp = (pd.read_csv(os.path.join(par_dir, "test_workpieces", pocket)) for pocket in wp_datasets)
wp_data = pd.concat(wp_temp, ignore_index=True)


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

# %% organize data in dictionairies
data_gF = {}
# get all possible geomFeatures in List
geomFeatures = list(dict.fromkeys(wp_data["geomFeature"]))
for feature in geomFeatures:
    data_gF[feature] = wp_data[wp_data["geomFeature"] == feature]#.reset_index()

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

# %% load model
with open(os.path.join(par_dir, "save_model", model_name), 'rb') as handle:
    model = pickle.load(handle)

prediction = {}

# define column titles of prediction
y_cols={}
y_cols["LIN_1"] = ['Lage_POCKET_MID_X_ABW', 'Lage_POCKET_MID_Y_ABW',
       '2D-Abstand_LIN_1_MID_LIN_3_MID_M_ABW', 'Geradheit_LIN_1_MID_ABW',
       'Parallelität_LIN_1_MID_ABW', '2D-Winkel_LIN_1_MID_X_ACHSE_A_ABW']
y_cols["LIN_2"] = ['Lage_POCKET_MID_X_ABW', 'Lage_POCKET_MID_Y_ABW',
       '2D-Abstand_LIN_2_MID_LIN_4_MID_M_ABW', 'Geradheit_LIN_2_MID_ABW',
       'Parallelität_LIN_2_MID_ABW', '2D-Winkel_LIN_2_MID_Y_ACHSE_A_ABW']
y_cols["LIN_3"] = ['Lage_POCKET_MID_X_ABW', 'Lage_POCKET_MID_Y_ABW',
       '2D-Abstand_LIN_1_MID_LIN_3_MID_M_ABW', 'Geradheit_LIN_3_MID_ABW',
       'Parallelität_LIN_3_MID_ABW', '2D-Winkel_LIN_3_MID_X_ACHSE_A_ABW']
y_cols["LIN_4"] = ['Lage_POCKET_MID_X_ABW', 'Lage_POCKET_MID_Y_ABW',
       '2D-Abstand_LIN_2_MID_LIN_4_MID_M_ABW', 'Geradheit_LIN_4_MID_ABW',
       'Parallelität_LIN_4_MID_ABW', '2D-Winkel_LIN_4_MID_Y_ACHSE_A_ABW']
y_cols["CIR_1"] = ['Lage_CIR_1_MID_X_ABW', 'Lage_CIR_1_MID_Y_ABW', 'Lage_CIR_1_MID_R_ABW',
       'Rundheit_CIR_1_MID_ABW']
y_cols["CIR_2"] = ['Lage_CIR_2_MID_X_ABW', 'Lage_CIR_2_MID_Y_ABW', 'Lage_CIR_2_MID_R_ABW',
       'Rundheit_CIR_2_MID_ABW']
y_cols["CIR_3"] = ['Lage_CIR_3_MID_X_ABW', 'Lage_CIR_3_MID_Y_ABW', 'Lage_CIR_3_MID_R_ABW',
       'Rundheit_CIR_3_MID_ABW']
y_cols["CIR_4"] = ['Lage_CIR_4_MID_X_ABW', 'Lage_CIR_4_MID_Y_ABW', 'Lage_CIR_4_MID_R_ABW',
       'Rundheit_CIR_4_MID_ABW']


for feature in data_gF_grouped.keys():
    X = get_data(data_gF_grouped, feature, drop_signal_dict, 0.7)[0]

    # predict Abweichung
    prediction[feature] = pd.DataFrame(data=model[feature].predict(X), columns=y_cols[feature])

# get classification
classification = check_tolerance(prediction)[0]

# get classification by geometric feature
classification_feature = pd.DataFrame()
for key, df in check_tolerance(prediction)[1].items():
    classification_feature = classification_feature.append(df, ignore_index=True)
classification_feature = classification_feature.transpose()
classification_feature.columns = ["LIN_1","CIR_1","LIN_2","CIR_2","LIN_3","CIR_3","LIN_4","CIR_4"]

for feature in data_gF_grouped.keys():
    classification[feature] = classification[feature].replace([0, 1], ["n.i.O.", "i.O."])
    classification_feature[feature] = classification_feature[feature].replace([0, 1], ["n.i.O.", "i.O."])

print("Prediction on test dataset is saved to variable <prediction>")
print("Classification on test dataset is saved to variable <classification_feature>:")
print(classification_feature)
