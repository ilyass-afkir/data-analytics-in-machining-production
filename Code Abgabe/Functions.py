
import pandas as pd
from matplotlib import pyplot as plt
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

pd.set_option('precision', 15)
# %%

def get_data(all_data, feature_fct, drop_signal_dict_fct, corr_value_fct):
    data = all_data[feature_fct].copy()
    # drop columns (based on analysis in preprocessing)
    if corr_value_fct != 0:
        try:
            drop_cols = [col for col in data.columns if
                         any(sub in col for sub in drop_signal_dict_fct[corr_value_fct][feature_fct])]
            data.drop(drop_cols, axis=1, inplace=True)
        except:
            print("ERROR: correlation value not found in drop_signal_dict, corr_value set to 0!")

    # assign data
    data_cols = [col for col in data.columns if any(sub in col for sub in ["mean", "std", "max", "min"])]
    X_fct = data[data_cols]

    # scale x data
    scaler = StandardScaler()
    X_fct = scaler.fit_transform(X_fct)
    X_fct = pd.DataFrame(data=X_fct, columns=data_cols)

    # assign label
    data_label = [col for col in data.columns if "_ABW" in col]
    y_fct = data[data_label]

    return X_fct, y_fct


def feature_selection(model_fct, x_train_fct, x_test_fct, y_test_fct, selection_threshold, plot_importance="no",
                      feature_fct="empty"):
    # some models don't output the feature importance --> feature selection algorithms can't be used.
    # Importance gets computed manually and feature selection is based on importance score
    # fit model
    # model_fct.fit(x_train_fct, y_train_fct)

    # get feature importance (for feature selection)
    scoring = ['r2', 'neg_mean_absolute_percentage_error', 'neg_mean_squared_error']
    results = permutation_importance(model_fct, x_test_fct, y_test_fct, scoring='neg_mean_absolute_percentage_error', n_repeats=10)
    # get importance
    importance = results.importances_mean
    df = pd.DataFrame(data={"columns": x_train_fct.columns, "importance": importance})
    df.sort_values(by="importance", ascending=False, inplace=True)

    # get x data with columns from feature selection
    x_train_fct_fs = x_train_fct[list(df["columns"][:selection_threshold])]
    x_test_fct_fs = x_test_fct[list(df["columns"][:selection_threshold])]

    # only plot importance if plot_importance is set to "yes"
    if plot_importance == "yes":
        # plot feature importance
        fig, ax = plt.subplots()
        ax.bar([x for x in range(len(importance))], importance)
        ax.set_title("Feature Importance for KNeighborRegressor \n geomFeature: {}".format(feature_fct))
        ax.set_xlabel("machine signal (feature)")
        plt.show()

    return x_train_fct_fs, x_test_fct_fs


def score_mean_squared_error(model_fct, x_test_fct, y_test_fct):
    # make prediction
    y_pred_fct = model_fct.predict(x_test_fct)
    y_pred_fct = pd.DataFrame(data=y_pred_fct, columns=y_test_fct.columns)
    # evaluate prediction
    mse_pred_fct = mean_squared_error(y_test_fct, y_pred_fct)
    mae_pred_fct = mean_absolute_error(y_test_fct, y_pred_fct)
    r2_pred_fct = r2_score(y_test_fct, y_pred_fct)
    print("Mean squared error on predicted data is:{}, model:{}".format(mse_pred_fct, model_fct))
    print("Mean absolute error on predicted data is:{}, model:{}".format(mae_pred_fct, model_fct))
    print("R2 score on predicted data is:{}, model:{}".format(r2_pred_fct, model_fct))
    return mse_pred_fct, mae_pred_fct, r2_pred_fct, y_pred_fct

def check_tolerance(class_data):
    # 0: niO, 1:iO
    # copy dataframe, otherwise function input dict gets changed as well
    class_data = pickle.loads(pickle.dumps(class_data))
    class_feature = {}
    class_stacked = {}
    # get classification for all features
    for feature in class_data.keys():
        if "CIR" in feature:
            class_data[feature][("Lage_" + feature + "_MID_X_ABW")] = \
                [0 if 0.04 <= val or val <= -0.04 else 1 for val in class_data[feature][("Lage_" + feature + "_MID_X_ABW")]]
            class_data[feature][("Lage_" + feature + "_MID_Y_ABW")] = \
                [0 if 0.04 <= val or val <= -0.04 else 1 for val in class_data[feature][("Lage_" + feature + "_MID_Y_ABW")]]
            class_data[feature][("Lage_" + feature + "_MID_R_ABW")] = \
                [0 if -0.015 <= val or val <= -0.04 else 1 for val in class_data[feature][("Lage_" + feature + "_MID_R_ABW")]]
            class_data[feature][("Rundheit_" + feature + "_MID_ABW")] = \
                [0 if 0.01 <= val or val <= 0 else 1 for val in class_data[feature][("Rundheit_" + feature + "_MID_ABW")]]
        if "LIN" in feature:
            class_data[feature]["Lage_POCKET_MID_X_ABW"] = \
                [0 if 0.04 <= val or val <= -0.04 else 1 for val in class_data[feature]["Lage_POCKET_MID_X_ABW"]]
            class_data[feature]["Lage_POCKET_MID_Y_ABW"] = \
                [0 if 0.04 <= val or val <= -0.04 else 1 for val in class_data[feature]["Lage_POCKET_MID_Y_ABW"]]
            class_data[feature][("Geradheit_"+ feature + "_MID_ABW")] = \
                [0 if 0.01 <= val or val <= 0 else 1 for val in class_data[feature][("Geradheit_" + feature + "_MID_ABW")]]
            class_data[feature][("Parallelität_" + feature + "_MID_ABW")] = \
                [0 if 0.01 <= val or val <= 0 else 1 for val in class_data[feature][("Parallelität_" + feature + "_MID_ABW")]]
            if "1" in feature or "3" in feature:
                class_data[feature]["2D-Abstand_LIN_1_MID_LIN_3_MID_M_ABW"] = \
                    [0 if 0.01 or val <= val <= -0.01 else 1 for val in
                     class_data[feature]["2D-Abstand_LIN_1_MID_LIN_3_MID_M_ABW"]]
                class_data[feature][("2D-Winkel_" + feature + "_MID_X_ACHSE_A_ABW")] = \
                    [0 if 0.01 <= val or val <= -0.01 else 1 for val in
                     class_data[feature][("2D-Winkel_" + feature + "_MID_X_ACHSE_A_ABW")]]
            if "2" in feature or "4" in feature:
                class_data[feature]["2D-Abstand_LIN_2_MID_LIN_4_MID_M_ABW"] = \
                    [0 if 0.01 <= val or val <= -0.01 else 1 for val in
                     class_data[feature]["2D-Abstand_LIN_2_MID_LIN_4_MID_M_ABW"]]
                class_data[feature][("2D-Winkel_" + feature + "_MID_Y_ACHSE_A_ABW")] = \
                    [0 if 0.01 <= val or val <= -0.01 else 1 for val in
                     class_data[feature][("2D-Winkel_" + feature + "_MID_Y_ACHSE_A_ABW")]]
        class_feature[feature] = class_data[feature].min(axis=1)
        class_stacked[feature] = class_data[feature].stack()
    return class_data, class_feature, class_stacked