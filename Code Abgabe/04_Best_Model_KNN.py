#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import os
import pandas as pd
from matplotlib import pyplot as plt
import pickle
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix 
from Functions import get_data, feature_selection, score_mean_squared_error, check_tolerance
pd.set_option('precision', 15)

# In[2]:


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


# In[3]:


# %% define variables for loop
# define model as dict --> save params for each geom feature individually
model1 = {}

# save predictions for each geom feature
model1_pred = {}

# save perfomance of all evalutated models
mse_model1 = []
mae_model1 = []
r2_model1 = []


# %% loop through all geomFeatures and fit each model to every geomFeature
for feature in all_data_gF.keys():
    # correlation value for dropping columns
    corr_value = 1

    X, y = get_data(all_data_gF, feature, drop_signal_dict, corr_value)
    # split into test and train set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # %% model1 : KNeighborsRegressor with hyperparameter optimization

    searchparams = {'algorithm': ['auto', "ball_tree", "kd_tree", "brute"],
                    'leaf_size': [15, 20, 25, 30, 35, 40, 45],
                    # 'metric': 'minkowski',
                    'n_neighbors': [1, 2, 3, 4],
                    'p': [1, 2],
                    'weights': ['uniform', "distance"]}
    

    # define gridsearch and perform fit
    model1[feature] = GridSearchCV(KNeighborsRegressor(), searchparams, return_train_score=True)
    
    # predict best model with all features
    model1[feature].fit(X_train, y_train)
    print(feature)
    mse_model1_pred, mae_model1_pred, r2_model1_pred, model1_pred[feature] = score_mean_squared_error(model1[feature], X_test, y_test)
    
    # save performance to dataframe
    mse_model1.append(pd.DataFrame(data={"mse_kneighbor_best": [mse_model1_pred]}, index=[feature]))
    mae_model1.append(pd.DataFrame(data={"mae_kneighbor_best": [mae_model1_pred]}, index=[feature]))
    r2_model1.append(pd.DataFrame(data={"r2_kneighbor_best": [r2_model1_pred]}, index=[feature]))


# In[4]:


#save results of the errors in dataframe
mae = pd.concat(mae_model1)   #mean absolute error
mse = pd.concat(mse_model1)   #mean squarred error
r2 = pd.concat(r2_model1)     #coefficient of determination
results = mae.join(mse).join(r2)
print(results)

# save model
# make new folder
if os.path.isdir(os.path.join(par_dir, "save_model")) is False:
    os.mkdir(os.path.join(par_dir, "save_model"))

with open(os.path.join(par_dir, "save_model", 'kNeighbor_corrvalue{}.pickle'.format(corr_value)), 'wb') as handle:
    pickle.dump(model1, handle)


# In[5]:

# get dataframe of all y test data
y_test_all = {}
for feature in all_data_gF.keys():
    print(feature)
    X, y = get_data(all_data_gF, feature, drop_signal_dict, corr_value)
    # split into test and train set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    y_test = y_test.reset_index()
    y_test = y_test.drop("index", axis=1)
    y_test_all[feature] = y_test


# In[6]:


#Prediction vs True

for feature in all_data_gF.keys():
    if 'CIR' in feature: 
        fig, axs = plt.subplots(2, 2)
        fig.set_size_inches(18.5, 10.5)
        fig.suptitle('KNN  ' +feature)
        axs[0, 0].scatter(model1_pred[feature].iloc[:,0], y_test_all[feature].iloc[:,0])
        axs[0, 0].set_title(model1_pred[feature].columns[0])
        axs[0, 0].set(xlabel='predicted', ylabel='true')
        axs[0, 0].plot([model1_pred[feature].iloc[:,0].min(), model1_pred[feature].iloc[:,0].max()], [model1_pred[feature].iloc[:,0].min(), model1_pred[feature].iloc[:,0].max()], ls="--", c=".3")
        axs[0, 1].scatter(model1_pred[feature].iloc[:,1], y_test_all[feature].iloc[:,1])
        axs[0, 1].plot([model1_pred[feature].iloc[:,1].min(), model1_pred[feature].iloc[:,1].max()], [model1_pred[feature].iloc[:,1].min(), model1_pred[feature].iloc[:,1].max()], ls="--", c=".3")
        axs[0, 1].set_title(model1_pred[feature].columns[1])
        axs[0, 1].set(xlabel='predicted', ylabel='true')
        axs[1, 0].plot([model1_pred[feature].iloc[:,2].min(), model1_pred[feature].iloc[:,2].max()], [model1_pred[feature].iloc[:,2].min(), model1_pred[feature].iloc[:,2].max()], ls="--", c=".3")
        axs[1, 0].scatter(model1_pred[feature].iloc[:,2], y_test_all[feature].iloc[:,2])
        axs[1, 0].set_title(model1_pred[feature].columns[2])
        axs[1, 0].set(xlabel='predicted', ylabel='true')
        axs[1, 1].scatter(model1_pred[feature].iloc[:,3], y_test_all[feature].iloc[:,3])
        axs[1, 1].plot([model1_pred[feature].iloc[:,3].min(), model1_pred[feature].iloc[:,3].max()], [model1_pred[feature].iloc[:,3].min(), model1_pred[feature].iloc[:,3].max()], ls="--", c=".3")
        axs[1, 1].set_title(model1_pred[feature].columns[3])
        axs[1, 1].set(xlabel='predicted', ylabel='true')
        fig.tight_layout()

        # save figures
        plt.savefig(os.path.join(par_dir, "saved_plots", (feature +'.png')), dpi=fig.dpi)
        plt.show()
    else:
        fig, axs = plt.subplots(3, 2)
        fig.set_size_inches(18.5, 10.5)
        fig.suptitle('KNN  ' +feature)
        axs[0, 0].scatter(model1_pred[feature].iloc[:,0], y_test_all[feature].iloc[:,0])
        axs[0, 0].set_title(model1_pred[feature].columns[0])
        axs[0, 0].set(xlabel='predicted', ylabel='true')
        axs[0, 0].plot([model1_pred[feature].iloc[:,0].min(), model1_pred[feature].iloc[:,0].max()], [model1_pred[feature].iloc[:,0].min(), model1_pred[feature].iloc[:,0].max()], ls="--", c=".3")
        axs[0, 1].scatter(model1_pred[feature].iloc[:,1], y_test_all[feature].iloc[:,1])
        axs[0, 1].plot([model1_pred[feature].iloc[:,1].min(), model1_pred[feature].iloc[:,1].max()], [model1_pred[feature].iloc[:,1].min(), model1_pred[feature].iloc[:,1].max()], ls="--", c=".3")
        axs[0, 1].set_title(model1_pred[feature].columns[1])
        axs[0, 1].set(xlabel='predicted', ylabel='true')
        axs[1, 0].plot([model1_pred[feature].iloc[:,2].min(), model1_pred[feature].iloc[:,2].max()], [model1_pred[feature].iloc[:,2].min(), model1_pred[feature].iloc[:,2].max()], ls="--", c=".3")
        axs[1, 0].scatter(model1_pred[feature].iloc[:,2], y_test_all[feature].iloc[:,2])
        axs[1, 0].set_title(model1_pred[feature].columns[2])
        axs[1, 0].set(xlabel='predicted', ylabel='true')
        axs[1, 1].scatter(model1_pred[feature].iloc[:,3], y_test_all[feature].iloc[:,3])
        axs[1, 1].plot([model1_pred[feature].iloc[:,3].min(), model1_pred[feature].iloc[:,3].max()], [model1_pred[feature].iloc[:,3].min(), model1_pred[feature].iloc[:,3].max()], ls="--", c=".3")
        axs[1, 1].set_title(model1_pred[feature].columns[3])
        axs[2, 0].scatter(model1_pred[feature].iloc[:,4], y_test_all[feature].iloc[:,4])
        axs[2, 0].set(xlabel='predicted', ylabel='true')
        axs[2, 0].plot([model1_pred[feature].iloc[:,4].min(), model1_pred[feature].iloc[:,4].max()], [model1_pred[feature].iloc[:,4].min(), model1_pred[feature].iloc[:,4].max()], ls="--", c=".3")
        axs[2, 0].set_title(model1_pred[feature].columns[4])
        axs[2, 1].scatter(model1_pred[feature].iloc[:,5], y_test_all[feature].iloc[:,5])
        axs[2, 1].set(xlabel='predicted', ylabel='true')
        axs[2, 1].plot([model1_pred[feature].iloc[:,5].min(), model1_pred[feature].iloc[:,5].max()], [model1_pred[feature].iloc[:,5].min(), model1_pred[feature].iloc[:,5].max()], ls="--", c=".3")
        axs[2, 1].set_title(model1_pred[feature].columns[5])        
        axs[2, 1].set(xlabel='predicted', ylabel='true')
        fig.tight_layout()

        # save figures
        plt.savefig(os.path.join(par_dir, "saved_plots", (feature +'.png')), dpi=fig.dpi)
        plt.show()

# In[7]:


# plot confusion matrix for each model (every prediction is counted as classification)
# transofrm numeric values to classification (classification for all values get stacked)
y_test_class = check_tolerance(y_test_all)[2]
# combine/stack data of all geom features
class_test = pd.concat(y_test_class.values())
target_labels = ["iO", "niO"]

model_class = check_tolerance(model1_pred)[2]
model1_all_class = check_tolerance(model1_pred)[0]
class_pred = pd.concat(model_class.values())

# compute and plot confusion matrix
cm = confusion_matrix(class_test, class_pred)
ax2 = plt.subplot()
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_labels).plot(ax=ax2)
ax2.set_title('Konfusionsmatrix für alle Klassifikationen (kNeighbor) \n corr_value = {}'.format(corr_value))
plt.savefig(os.path.join(par_dir, "saved_plots", "confusion_matrix_kNeigbor_{}.png".format(corr_value)))
plt.show()

print("Predictions for Model without Feature Importance are saved in model1_pred.")
print("Classifications for Model without Feature Importance are saved in model1_all_class.")

# In[ ]: Feature Selection

# number of selected features
fs_threshold = 68

# save predictions for each geom feature
model_fs_pred = {}
# save columns after feature selection
list_cols = {}
list_signals = {}
# model feature selection
model_fs = pickle.loads(pickle.dumps(model1))

# feature selection for each geom feature
for feature in all_data_gF.keys():

    X, y = get_data(all_data_gF, feature, drop_signal_dict, corr_value)
    # split into test and train set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


    X_train_fs, X_test_fs = feature_selection(model_fs[feature], X_train, X_test, y_test,
                                              selection_threshold=fs_threshold,
                                              plot_importance="no",
                                              feature_fct=feature)

    # predict best model with all features
    model_fs[feature].fit(X_train_fs, y_train)
    print(feature)
    mse_model_fs, mae_model_fs, r2_model_fs, model_fs_pred[feature] = score_mean_squared_error(model_fs[feature], X_test_fs, y_test)

    # get remaining statistical features
    list_cols[feature] = list(X_test_fs.columns)
    # get remaining signals
    list_signals[feature] = [col.replace("_mean", "").replace("_std", "").replace("_max", "").replace("_min", "") for col in list_cols[feature]]
    list_signals[feature] = list(set(list_signals[feature]))

# with open(os.path.join(par_dir, "save_model", 'kNeighbor_fs.pickle'), 'wb') as handle:
#     pickle.dump(model_fs, handle)
# In[ ]:
# confusion matrix for feature selection

# plot confusion matrix for each model (every prediction is counted as classification)
# transofrm numeric values to classification (classification for all values get stacked)
y_test_class = check_tolerance(y_test_all)[2]
# combine/stack data of all geom features
class_test = pd.concat(y_test_class.values())
target_labels = ["iO", "niO"]

model_class = check_tolerance(model_fs_pred)[2]
modelfs_all_class = check_tolerance(model1_pred)[0]
class_pred = pd.concat(model_class.values())

# compute and plot confusion matrix
cm = confusion_matrix(class_test, class_pred)
ax2 = plt.subplot()
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_labels).plot(ax=ax2)
ax2.set_title('Konfusionsmatrix für alle Klassifikationen (kNeighbor) \n mit Feature Selection')
plt.savefig(os.path.join(par_dir, "saved_plots", "confusion_matrix_kNeigbor_fs.png"))
plt.show()



print("Predictions for Model without Feature Importance are saved in model_fs_pred.")
print("Classifications for Model without Feature Importance are saved in model_fs_all_class.")