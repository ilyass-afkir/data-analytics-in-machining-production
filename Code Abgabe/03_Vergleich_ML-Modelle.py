# %% Import libraries and functions
import os
import pandas as pd

from matplotlib import pyplot as plt
import pickle
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from Functions import get_data, check_tolerance, feature_selection, score_mean_squared_error

pd.set_option('precision', 15)

# %% Describtion of code:
# all data (model parameters, predictions,...) gets saved in a dictionary for each geometric feature.
# example:
# model1_pred = {"LIN_1": xxx,
#                "CIR_1: xxx,...}
# Every model gets trained for each geometric feature to get best results.
# %% import data
par_dir = os.getcwd()
# load saved data

with open(os.path.join(par_dir, "save_data", 'all_data_gF.pickle'), 'rb') as handle:
    all_data_gF = pickle.load(handle)

with open(os.path.join(par_dir, "save_data", 'drop_signals_dict.pickle'), 'rb') as handle:
    drop_signal_dict = pickle.load(handle)
# %% define variables for loop

# define model as dict --> save params for each geom feature individually
model1 = {}
model2 = {}
model3 = {}

# save predictions for each geom feature
model1_pred = {}
model2_pred = {}
model3_pred = {}

# save perfomance of all evalutated models
mse_model1 = []
mae_model1 = []
r2_model1 = []

mse_model2 = []
mae_model2 = []
r2_model2 = []

mse_model3 = []
mae_model3 = []
r2_model3 = []

# %% loop through all geomFeatures and fit each model to every geomFeature
# Warning: This part of the code may take a few hours to compile depending on your PC configurations.
for feature in all_data_gF.keys():
    # correlation value for dropping columns
    corr_value = 0

    X, y = get_data(all_data_gF, feature, drop_signal_dict, corr_value)
    # split into test and train set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # %% Machine Leanrning Models with hyperparameter optimization
    #    - Model 1: KNeighborsRegressor()
    #    - Model 2: RandomForestRegressor()
    #    - Model 3: DecisionTreeRegressor()

    param_grid_KN = {'algorithm': ['auto', "ball_tree", "kd_tree", "brute"],
                     'leaf_size': [15, 20, 25, 30, 35, 40, 45],
                     # 'metric': 'minkowski',
                     'n_neighbors': [1, 2, 3, 4],
                     'p': [1, 2],
                     'weights': ['uniform', "distance"]}

    param_grid_RF = {'n_estimators': [500, 600],
                     'max_depth': [15, 100],
                     'min_samples_split': [5, 100],
                     'min_samples_leaf': [5, 100]}

    param_grid_DT = {"splitter": ["best", "random"],
                     "max_depth": [1, 3, 5, 7, 9, 11, 12],
                     "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                     "min_weight_fraction_leaf": [0.1, 0.2, 0.3, 0.4, 0.5],
                     "max_features": ["auto", "log2", "sqrt", None],
                     "max_leaf_nodes": [None, 10, 20, 30, 40, 50, 60, 70, 80, 90]}

    # define GridSearch and perform fit
    model1[feature] = GridSearchCV(KNeighborsRegressor(), param_grid_KN, return_train_score=True)
    model2[feature] = GridSearchCV(RandomForestRegressor(), param_grid_RF, return_train_score=True)
    model3[feature] = GridSearchCV(DecisionTreeRegressor(), param_grid_DT, return_train_score=True)

    # predict best model with all features
    model1[feature].fit(X_train, y_train)
    model2[feature].fit(X_train, y_train)
    model3[feature].fit(X_train, y_train)

    mse_model1_pred, mae_model1_pred, r2_model1_pred, model1_pred[feature] = score_mean_squared_error(model1[feature],
                                                                                                      X_test, y_test)
    mse_model2_pred, mae_model2_pred, r2_model2_pred, model2_pred[feature] = score_mean_squared_error(model2[feature],
                                                                                                      X_test, y_test)
    mse_model3_pred, mae_model3_pred, r2_model3_pred, model3_pred[feature] = score_mean_squared_error(model3[feature],
                                                                                                      X_test, y_test)
    # save performance to dataframe
    mse_model1.append(pd.DataFrame(data={"mse_kneighbor_best": [mse_model1_pred]}, index=[feature]))
    mae_model1.append(pd.DataFrame(data={"mae_kneighbor_best": [mae_model1_pred]}, index=[feature]))
    r2_model1.append(pd.DataFrame(data={"r2_kneighbor_best": [r2_model1_pred]}, index=[feature]))

    mse_model2.append(pd.DataFrame(data={"mse_rfr_best": [mse_model2_pred]}, index=[feature]))
    mae_model2.append(pd.DataFrame(data={"mae_rfr_best": [mae_model2_pred]}, index=[feature]))
    r2_model2.append(pd.DataFrame(data={"r2_rfr_best": [r2_model2_pred]}, index=[feature]))

    mse_model3.append(pd.DataFrame(data={"mse_decisiontree_best": [mse_model3_pred]}, index=[feature]))
    mae_model3.append(pd.DataFrame(data={"mae_decisiontree_best": [mae_model3_pred]}, index=[feature]))
    r2_model3.append(pd.DataFrame(data={"r2_decisiontree_best": [r2_model3_pred]}, index=[feature]))

# %% Join all results together

mae1 = pd.concat(mae_model1)
mae2 = pd.concat(mae_model2)
mae3 = pd.concat(mae_model3)
mae = mae1.join(mae2).join(mae3)

mse1 = pd.concat(mse_model1)
mse2 = pd.concat(mse_model2)
mse3 = pd.concat(mse_model3)
mse = mse1.join(mse2).join(mse3)

r21 = pd.concat(r2_model1)
r22 = pd.concat(r2_model2)
r23 = pd.concat(r2_model3)
r2 = r21.join(r22).join(r23)


results = mae.join(mse).join(r2)

# %% Print out results
print(results)

# Save results to csv file
results.to_csv(os.path.join(par_dir, 'Ergebnis_ML-Modellvergleich.csv'), index = False)

# %% Evaluation of Model 1: KNeighborsRegressor()

for feature in all_data_gF.keys():
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(18.5, 10.5)
    fig.suptitle('KNN  ' + feature)
    axs[0, 0].scatter(model1_pred[feature].iloc[:, 0], y_test.iloc[:, 0])
    axs[0, 0].set_title('Lage_MID_X_ABW')
    axs[0, 0].set(xlabel='predicted', ylabel='true')
    axs[0, 0].plot([model1_pred[feature].iloc[:, 0].min(), model1_pred[feature].iloc[:, 0].max()],
                   [model1_pred[feature].iloc[:, 0].min(), model1_pred[feature].iloc[:, 0].max()], ls="--", c=".3")
    axs[0, 1].scatter(model1_pred[feature].iloc[:, 1], y_test.iloc[:, 1])
    axs[0, 1].plot([model1_pred[feature].iloc[:, 1].min(), model1_pred[feature].iloc[:, 1].max()],
                   [model1_pred[feature].iloc[:, 1].min(), model1_pred[feature].iloc[:, 1].max()], ls="--", c=".3")
    axs[0, 1].set_title('Lage_MID_Y_ABW')
    axs[0, 1].set(xlabel='predicted', ylabel='true')
    axs[1, 0].plot([model1_pred[feature].iloc[:, 2].min(), model1_pred[feature].iloc[:, 2].max()],
                   [model1_pred[feature].iloc[:, 2].min(), model1_pred[feature].iloc[:, 2].max()], ls="--", c=".3")
    axs[1, 0].scatter(model1_pred[feature].iloc[:, 2], y_test.iloc[:, 2])
    axs[1, 0].set_title('Lage_MID_R_ABW')
    axs[1, 0].set(xlabel='predicted', ylabel='true')
    axs[1, 1].scatter(model1_pred[feature].iloc[:, 3], y_test.iloc[:, 3])
    axs[1, 1].plot([model1_pred[feature].iloc[:, 3].min(), model1_pred[feature].iloc[:, 3].max()],
                   [model1_pred[feature].iloc[:, 3].min(), model1_pred[feature].iloc[:, 3].max()], ls="--", c=".3")
    axs[1, 1].set_title('Rundheit_MID_ABW')
    axs[1, 1].set(xlabel='predicted', ylabel='true')

    fig.tight_layout()

    # save figures
    plt.savefig(os.path.join(par_dir, "saved_plots", (feature +'.png')), dpi=fig.dpi)
    plt.show()

# %% Evaluation of Model 2: RandomForestRegressor()

for feature in all_data_gF.keys():
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(18.5, 10.5)
    fig.suptitle('RFR  ' + feature)
    axs[0, 0].scatter(model2_pred[feature].iloc[:, 0], y_test.iloc[:, 0])
    axs[0, 0].set_title('Lage_MID_X_ABW')
    axs[0, 0].set(xlabel='predicted', ylabel='true')
    axs[0, 0].plot([model2_pred[feature].iloc[:, 0].min(), model2_pred[feature].iloc[:, 0].max()],
                   [model2_pred[feature].iloc[:, 0].min(), model2_pred[feature].iloc[:, 0].max()], ls="--", c=".3")
    axs[0, 1].scatter(model2_pred[feature].iloc[:, 1], y_test.iloc[:, 1])
    axs[0, 1].plot([model2_pred[feature].iloc[:, 1].min(), model2_pred[feature].iloc[:, 1].max()],
                   [model2_pred[feature].iloc[:, 1].min(), model2_pred[feature].iloc[:, 1].max()], ls="--", c=".3")
    axs[0, 1].set_title('Lage_MID_Y_ABW')
    axs[0, 1].set(xlabel='predicted', ylabel='true')
    axs[1, 0].plot([model2_pred[feature].iloc[:, 2].min(), model2_pred[feature].iloc[:, 2].max()],
                   [model2_pred[feature].iloc[:, 2].min(), model2_pred[feature].iloc[:, 2].max()], ls="--", c=".3")
    axs[1, 0].scatter(model2_pred[feature].iloc[:, 2], y_test.iloc[:, 2])
    axs[1, 0].set_title('Lage_MID_R_ABW')
    axs[1, 0].set(xlabel='predicted', ylabel='true')
    axs[1, 1].scatter(model2_pred[feature].iloc[:, 3], y_test.iloc[:, 3])
    axs[1, 1].plot([model2_pred[feature].iloc[:, 3].min(), model2_pred[feature].iloc[:, 3].max()],
                   [model2_pred[feature].iloc[:, 3].min(), model2_pred[feature].iloc[:, 3].max()], ls="--", c=".3")
    axs[1, 1].set_title('Rundheit_MID_ABW')
    axs[1, 1].set(xlabel='predicted', ylabel='true')
        
    fig.tight_layout()

    # save figures
    plt.savefig(os.path.join(par_dir, "saved_plots", (feature +'.png')), dpi=fig.dpi)
    plt.show()

#%% Evaluation of Model 3: DecisionTreeRegressor()

for feature in all_data_gF.keys():
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(18.5, 10.5)
    fig.suptitle('DTR  ' + feature)
    axs[0, 0].scatter(model3_pred[feature].iloc[:, 0], y_test.iloc[:, 0])
    axs[0, 0].set_title('Lage_MID_X_ABW')
    axs[0, 0].set(xlabel='predicted', ylabel='true')
    axs[0, 0].plot([model3_pred[feature].iloc[:, 0].min(), model3_pred[feature].iloc[:, 0].max()],
                   [model3_pred[feature].iloc[:, 0].min(), model3_pred[feature].iloc[:, 0].max()], ls="--", c=".3")
    axs[0, 1].scatter(model3_pred[feature].iloc[:, 1], y_test.iloc[:, 1])
    axs[0, 1].plot([model3_pred[feature].iloc[:, 1].min(), model3_pred[feature].iloc[:, 1].max()],
                   [model3_pred[feature].iloc[:, 1].min(), model3_pred[feature].iloc[:, 1].max()], ls="--", c=".3")
    axs[0, 1].set_title('Lage_MID_Y_ABW')
    axs[0, 1].set(xlabel='predicted', ylabel='true')
    axs[1, 0].plot([model3_pred[feature].iloc[:, 2].min(), model3_pred[feature].iloc[:, 2].max()],
                   [model3_pred[feature].iloc[:, 2].min(), model3_pred[feature].iloc[:, 2].max()], ls="--", c=".3")
    axs[1, 0].scatter(model3_pred[feature].iloc[:, 2], y_test.iloc[:, 2])
    axs[1, 0].set_title('Lage_MID_R_ABW')
    axs[1, 0].set(xlabel='predicted', ylabel='true')
    axs[1, 1].scatter(model3_pred[feature].iloc[:, 3], y_test.iloc[:, 3])
    axs[1, 1].plot([model3_pred[feature].iloc[:, 3].min(), model3_pred[feature].iloc[:, 3].max()],
                   [model3_pred[feature].iloc[:, 3].min(), model3_pred[feature].iloc[:, 3].max()], ls="--", c=".3")
    axs[1, 1].set_title('Rundheit_MID_ABW')
    axs[1, 1].set(xlabel='predicted', ylabel='true')

    fig.tight_layout()

    # save figures
    plt.savefig(os.path.join(par_dir, "saved_plots", (feature +'.png')), dpi=fig.dpi)
    plt.show()

# %% get classification of test data ( valid for all models)
y_test_all = {}
for feature in all_data_gF.keys():
    print(feature)
    X, y = get_data(all_data_gF, feature, drop_signal_dict, corr_value)
    # split into test and train set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    y_test = y_test.reset_index()
    y_test = y_test.drop("index", axis=1)
    y_test_all[feature] = y_test

# %% plot confusion matrix for each model (every prediction is counted as classification)

# transofrm numeric values to classification (classification for all values get stacked)
y_test_class = check_tolerance(y_test_all)[2]
# combine/stack data of all geom features
class_test = pd.concat(y_test_class.values())
target_labels = ["iO", "niO"]

#%% Create a new folder to save the confusion matrices of all models
if os.path.isdir(os.path.join(par_dir, "saved_plots", "Konfusionsmatrizen_ML-Modellvergleich")) is False:
    os.mkdir(os.path.join(par_dir, "saved_plots", "Konfusionsmatrizen_ML-Modellvergleich"))

# %% confusion matrix for model1

# transform numeric values to classification (classification for all values get stacked)
model_class = check_tolerance(model1_pred)[2]
# combine/stack data of all geom features
class_pred = pd.concat(model_class.values())

# compute and plot confusion matrix
cm = confusion_matrix(class_test, class_pred)
ax2 = plt.subplot()
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_labels).plot(ax=ax2)
ax2.set_title('Konfusionsmatrix mit allen Klassifikationen (kNeighbor)')
plt.show()
plt.savefig(os.path.join(par_dir, 'saved_plots', "Konfusionsmatrizen_ML-Modellvergleich", 'Konfusionsmatrix mit allen Klassifikationen (kNeighbor).png'))

# %% confusion matrix for model2

# transform numeric values to classification (classification for all values get stacked)
model_class = check_tolerance(model2_pred)[2]
# combine/stack data of all geom features
class_pred = pd.concat(model_class.values())

# compute and plot confusion matrix
cm = confusion_matrix(class_test, class_pred)
ax2 = plt.subplot()
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_labels).plot(ax=ax2)
ax2.set_title('Konfusionsmatrix mit allen Klassifikationen (RandomForest)')
plt.show()
plt.savefig(os.path.join(par_dir, 'saved_plots',"Konfusionsmatrizen_ML-Modellvergleich", 'Konfusionsmatrix mit allen Klassifikationen (RandomForest).png'))

# %% confusion matrix for model3

# transform numeric values to classification (classification for all values get stacked)
model_class = check_tolerance(model3_pred)[2]
# combine/stack data of all geom features
class_pred = pd.concat(model_class.values())

# compute and plot confusion matrix
cm = confusion_matrix(class_test, class_pred)
ax2 = plt.subplot()
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_labels).plot(ax=ax2)
ax2.set_title('Konfusionsmatrix mit allen Klassifikationen (DecisionTree)')
plt.show()
plt.savefig(os.path.join(par_dir, 'saved_plots', "Konfusionsmatrizen_ML-Modellvergleich", 'Konfusionsmatrix mit allen Klassifikationen (DecisionTree).png'))
