from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import tensorflow as tf
import pandas as pd


def pred_metrics(y_pred, y_true):
    """Calucate evaluation metrics for precision, recall, and f1.
    Arguments
    ---------
        y_pred: ndarry, the predicted result list
        y_true: ndarray, the ground truth label list
    Returns
    -------
        precision: float, precision value
        recall: float, recall value
        f1: float, f1 measure value
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )
    return precision, recall, f1


def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


def rf_feature_selection(X_train):
    from sklearn.ensemble import RandomForestClassifier

    rfc = RandomForestClassifier()

    # fit random forest classifier on the training set
    rfc.fit(X_train, y_train)
    # extract important features
    score = np.round(rfc.feature_importances_, 3)
    importances = pd.DataFrame({"feature": X_train.columns, "importance": score})
    importances = importances.sort_values("importance", ascending=False).set_index(
        "feature"
    )
    # plot importances
    plt.rcParams["figure.figsize"] = (11, 4)
    importances.plot.bar()
    return importances


def rfe_feature_selection(X_train, y_train):
    from sklearn.feature_selection import RFE
    import itertools

    rfc = RandomForestClassifier()

    # create the RFE model and select 10 attributes
    rfe = RFE(rfc, n_features_to_select=15)
    rfe = rfe.fit(X_train, y_train)

    # summarize the selection of the attributes
    feature_map = [
        (i, v) for i, v in itertools.zip_longest(rfe.get_support(), X_train.columns)
    ]
    selected_features = [v for i, v in feature_map if i == True]

    return selected_features


def getDistanceByPoint(data, model):
    # return Series of distance between each point and his distance with the closest centroid
    distance = pd.Series()
    for i in range(0, len(data)):
        Xa = np.array(data.loc[i])
        Xb = model.cluster_centers_[model.labels_[i] - 1]
        distance.set_value(i, np.linalg.norm(Xa - Xb))
    return distance


def getTransitionMatrix(df):
    # train markov model to get transition matrix
    df = np.array(df)
    model = msm.estimate_markov_model(df, 1)
    return model.transition_matrix


def markovAnomaly(df, windows_size, threshold):
    transition_matrix = getTransitionMatrix(df)
    real_threshold = threshold ** windows_size
    df_anomaly = []
    for j in range(0, len(df)):
        if j < windows_size:
            df_anomaly.append(0)
        else:
            sequence = df[j - windows_size : j]
            sequence = sequence.reset_index(drop=True)
            df_anomaly.append(
                anomalyElement(sequence, real_threshold, transition_matrix)
            )
    return df_anomaly


def cluster_elbow_plot(X_train):
    from sklearn.cluster import KMeans

    # calculate with different number of centroids to see the loss plot (elbow method)
    n_cluster = range(1, 20)
    kmeans = [KMeans(n_clusters=i).fit(X_train) for i in n_cluster]
    scores = [kmeans[i].score(X_train) for i in range(len(kmeans))]
    fig, ax = plt.subplots()
    ax.plot(n_cluster, scores)
    plt.show()


def observe_binary_feature_distribution(df):
    import matplotlib.gridspec as gridspec

    v_features = df.iloc[:, 0:30].columns

    plt.figure(figsize=(12, 120))
    gs = gridspec.GridSpec(30, 1)
    for i, cn in enumerate(df[v_features]):
        try:
            ax = plt.subplot(gs[i])
            sns.distplot(
                df[cn][df["class"] == 0], bins=50, label="Normal", kde_kws={"bw": 1}
            )
            sns.distplot(
                df[cn][df["class"] == 1], bins=50, label="Anomalous", kde_kws={"bw": 1}
            )
            ax.set_xlabel("")
            ax.set_title("Histogram of feature: " + str(cn))
            plt.legend()
        except Exception as e:  # ignore categoricals
            pass

    plt.show()


# ------------------------------------------------------------------------- #
# ------------------------------------------------------------------------- #


def plot_val_loss(h):
    # h:tf.python.keras.callbacks.History
    # val_loss is the value of cost function for your cross-validation data and loss is the value of cost function for your training data.

    plt.figure()
    plt.plot(h.history["loss"], label="loss")
    plt.plot(h.history["val_loss"], label="val_loss")
    plt.xlabel("Epochs ")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_reconstruction_error(train_error):
    plt.figure(figsize=(12, 5))
    plt.scatter(
        train_error.index[train_error["True_class"] == 0],
        train_error[train_error["True_class"] == 0]["Reconstruction_error"],
        s=5,
        label="Normal",
    )
    plt.scatter(
        train_error.index[train_error["True_class"] == 1],
        train_error[train_error["True_class"] == 1]["Reconstruction_error"],
        s=5,
        label="Anomaly",
    )
    plt.xlabel("Index")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.legend()
    plt.show()


def find_threshold_prc(train_error):
    from sklearn.metrics import (
        precision_recall_curve,
        roc_curve,
        auc,
        average_precision_score,
    )

    # Plotting the precision recall curve.
    precision, recall, threshold = precision_recall_curve(
        train_error.True_class, train_error.Reconstruction_error
    )
    f1_score = 2 * precision * recall / (precision + recall)
    average_precision = average_precision_score(
        train_error.True_class, train_error.Reconstruction_error
    )

    # Choosing the threshold to maximize the F1 score
    max_f1 = f1_score[f1_score == max(f1_score)]
    best_threshold = threshold[f1_score[1:] == max_f1]

    return best_threshold, f1_score, average_precision


def plot_pr_recall_curve(train_error):
    from sklearn.metrics import (
        precision_recall_curve,
        roc_curve,
        auc,
        average_precision_score,
    )

    precision, recall, threshold = precision_recall_curve(
        train_error.True_class, train_error.Reconstruction_error
    )
    f1_score = 2 * precision * recall / (precision + recall)
    average_precision = average_precision_score(
        train_error.True_class, train_error.Reconstruction_error
    )

    # Choosing the threshold to maximize the F1 score
    max_f1 = f1_score[f1_score == max(f1_score)]
    best_threshold = threshold[f1_score[1:] == max_f1]
    # Precision, Recall curve
    plt.figure(figsize=(12, 6))
    plt.plot(threshold, precision[1:], label="Precision", linewidth=3)
    plt.plot(threshold, recall[1:], label="Recall", linewidth=3)
    plt.axvline(
        best_threshold,
        color="black",
        ls="--",
        label="Threshold = %0.3f" % (best_threshold),
    )
    plt.ylim(0, 1.1)
    plt.xlabel("Threshold")
    plt.ylabel("Precision/ Recall")
    plt.title("Precision and recall for different threshold values")
    plt.legend(loc="upper right")

    ## F1 score curve
    plt.figure(figsize=(12, 6))
    plt.plot(threshold, f1_score[1:], label="F1_score", linewidth=3, color="green")
    plt.scatter(
        threshold[f1_score[1:] == max_f1],
        max_f1,
        label="Max F1 score = %0.3f" % (max_f1),
        s=50,
        color="red",
    )
    plt.axvline(
        best_threshold,
        color="black",
        ls="--",
        label="Threshold = %0.3f" % (best_threshold),
    )
    plt.axhline(max_f1, color="black", ls="-")
    plt.ylim(0, 1.1)
    plt.xlabel("Threshold")
    plt.ylabel("F1 score")
    plt.title("F1 score for different threshold values")
    plt.legend(loc="upper right")

    plt.show()
    print("Best threshold = %f" % (best_threshold))
    print("Max F1 score = %f" % (max_f1))


def plot_recall_pr_curve(train_error):
    from sklearn.metrics import (
        precision_recall_curve,
        roc_curve,
        auc,
        average_precision_score,
    )

    precision, recall, threshold = precision_recall_curve(
        train_error.True_class, train_error.Reconstruction_error
    )
    f1_score = 2 * precision * recall / (precision + recall)
    average_precision = average_precision_score(
        train_error.True_class, train_error.Reconstruction_error
    )

    # Choosing the threshold to maximize the F1 score
    max_f1 = f1_score[f1_score == max(f1_score)]
    best_threshold = threshold[f1_score[1:] == max_f1]
    # Recall - Precision curve
    plt.figure(figsize=(12, 6))
    f_scores = np.linspace(0.2, 0.8, num=4)

    for f_score in f_scores:
        x = np.linspace(0.001, 1)
        y = f_score * x / (2 * x - f_score)
        plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("F1 = {0:0.2f}".format(f_score), xy=(0.95, y[45] + 0.02))

    plt.plot(
        recall[1:],
        precision[1:],
        label="Area = %0.3f" % (average_precision),
        linewidth=3,
    )
    plt.scatter(
        recall[f1_score == max_f1],
        precision[f1_score == max_f1],
        label="F1 score = %0.3f" % (max_f1),
        s=50,
        color="red",
    )
    plt.axvline(
        recall[f1_score == max_f1],
        color="black",
        ls="--",
        label="Recall = %0.3f" % (recall[f1_score == max_f1]),
    )
    plt.axhline(
        precision[f1_score == max_f1],
        color="black",
        ls="-",
        label="Precision = %0.3f" % (precision[f1_score == max_f1]),
    )
    plt.ylim(0, 1.1)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision - Recall curve")
    plt.legend(loc="upper right")

    plt.show()


def plot_reconstruction_error(train_error: pd.DataFrame, best_threshold):
    plt.figure(figsize=(12, 5))
    plt.scatter(
        train_error.index[train_error["True_class"] == 0],
        train_error[train_error["True_class"] == 0]["Reconstruction_error"],
        s=5,
        label="Normal",
    )
    plt.scatter(
        train_error.index[train_error["True_class"] == 1],
        train_error[train_error["True_class"] == 1]["Reconstruction_error"],
        s=5,
        label="Anomaly",
    )
    plt.axhline(
        best_threshold, color="red", label="Threshold = %0.3f" % (best_threshold)
    )
    plt.xlabel("Index")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("Training Set")
    plt.legend()
    plt.show()
    print("Best threshold = %f" % (best_threshold))


def plot_roc_curve(train_error):
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, thresholds = roc_curve(
        train_error.True_class, train_error.Reconstruction_error
    )
    roc_auc = auc(fpr, tpr)

    # ROC curve
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, linewidth=3, label="AUC = %0.3f" % (roc_auc))
    plt.plot([0, 1], [0, 1], linewidth=3)
    plt.xlim(left=-0.02, right=1)
    plt.ylim(bottom=0, top=1.02)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Receiver operating characteristic curve (ROC)")
    plt.legend()
    plt.show()


# Function to calculate VIF
def calculate_vif(data):
    # VIF=1, Very Less Multicollinearity
    # VIF<5, Moderate Multicollinearity
    # VIF>5, Extreme Multicollinearity (This is what we have to avoid)
    vif_df = pd.DataFrame(columns=["Var", "Vif"])
    x_var_names = data.columns
    for i in range(0, x_var_names.shape[0]):
        y = data[x_var_names[i]]
        x = data[x_var_names.drop([x_var_names[i]])]
        r_squared = sm.OLS(y, x).fit().rsquared
        vif = round(1 / (1 - r_squared), 2)
        vif_df.loc[i] = [x_var_names[i], vif]
    return vif_df.sort_values(by="Vif", axis=0, ascending=False, inplace=False)
