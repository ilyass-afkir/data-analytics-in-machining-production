# import libraries
import os
import seaborn as sns
from matplotlib import pyplot as plt
import pickle
from Functions import get_data, check_tolerance
# %% import data
par_dir = os.getcwd()
# load saved data
with open(os.path.join(par_dir, "save_data", 'data_gF_grouped.pickle'), 'rb') as handle:
    data_gF_grouped = pickle.load(handle)
with open(os.path.join(par_dir, "save_data", 'data_gF.pickle'), 'rb') as handle:
    data_gF = pickle.load(handle)
with open(os.path.join(par_dir, "save_data", 'cmm_gF.pickle'), 'rb') as handle:
    cmm_gF = pickle.load(handle)
with open(os.path.join(par_dir, "save_data", 'all_data_gF.pickle'), 'rb') as handle:
    all_data_gF = pickle.load(handle)
with open(os.path.join(par_dir, "save_data", 'drop_signals_dict.pickle'), 'rb') as handle:
    drop_signal_dict = pickle.load(handle)

# %% Plot correlation matrix for eac feature
# create folder for plots
if os.path.isdir(os.path.join(par_dir, "saved_plots")) is False:
    os.mkdir(os.path.join(par_dir, "saved_plots"))

# plot and save correlation matrix for each feature
for feature in data_gF.keys():
    data_corr = data_gF[feature].select_dtypes(include=['float'])
    corr = data_corr.corr()
    # plot correlation matrix
    fig, ax = plt.subplots(figsize=(11, 10))
    sns.heatmap(corr, cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax)
    ax.set_title("Korrelationsmatrix für {}".format(feature), fontsize=20, pad=30)
    fig.savefig(os.path.join(par_dir, "saved_plots", 'correlation_matrix_{}.jpg'.format(feature)))


# %% Pairplots
par_dir = os.getcwd()
# load saved data
with open(os.path.join(par_dir, "save_data", 'all_data_gF.pickle'), 'rb') as handle:
    all_data_gF = pickle.load(handle)

X_data = {}
y_data = {}
for feature in all_data_gF.keys():
    X_data[feature], y_data[feature] = get_data(all_data_gF, feature, None, None)

    # compute classification from deviation
    class_individ, class_feature, class_pocket = check_tolerance(y_data)

# list of signals
signal_list = data_gF[feature].select_dtypes(include=['float']).columns

# plot pairplots for every signal and geom feature
for feature in all_data_gF.keys():
    if feature == "LIN_1" or feature == "CIR_1":
        # new folder to save pairplots
        if os.path.isdir(os.path.join(par_dir, "saved_plots", "Pairplot_{}".format(feature))) is False:
            os.mkdir(os.path.join(par_dir, "saved_plots", "Pairplot_{}".format(feature)))

        # plot pairplot for each signal
        for signal in signal_list:
            cols = [col for col in X_data[feature].columns if signal in col]
            data_plot = X_data[feature][cols]
            data_plot["classification"] = class_feature[feature].copy()
            data_plot["classification"].replace([0, 1], ["n.i.O.", "i.O."], inplace=True)

            # plot data
            g = sns.pairplot(data_plot, hue="classification", palette={"n.i.O.":sns.color_palette()[0], "i.O.":sns.color_palette()[1]})
            plt.savefig(os.path.join(par_dir, "saved_plots", "Pairplot_{}".format(feature), "Pairplot_{}_signal_{}".format(feature, signal).replace("|","_")))
            plt.close()

# %% Pairplot für jedes Maschinensignal zu jedem Messwert in jedem geometrischen Feature
# insgesamt 4*6*45 + 4*4*45 = 1800 Plots --> nur einzelne geometrische Features oder Signale plotten

# for feature in all_data_gF.keys():
#     if feature == "LIN_1":
#         for val in class_individ[feature].columns:
#             # new folder to save pairplots
#             if os.path.isdir(os.path.join(par_dir, "saved_plots", "Pairplot_{}".format(feature), val)) is False:
#                 os.mkdir(os.path.join(par_dir, "saved_plots", "Pairplot_{}".format(feature), val))
#
#             # plot pairplot for each signal
#             for signal in signal_list:
#                 cols = [col for col in X_data[feature].columns if signal in col]
#                 data_plot = X_data[feature][cols]
#                 data_plot["classification"] = class_individ[feature][val].copy()
#                 data_plot["classification"].replace([0, 1], ["n.i.O.", "i.O."], inplace=True)
#
#                 # plot data
#                 g = sns.pairplot(data_plot,
#                                  palette={"n.i.O.":sns.color_palette()[0], "i.O.":sns.color_palette()[1]},
#                                  hue="classification")
#                 plt.savefig(os.path.join(par_dir, "saved_plots", "Pairplot_{}".format(feature), val,
#                                          "Pairplot_{}_signal_{}".format(feature, signal).replace("|","_")))
#                 plt.close()
