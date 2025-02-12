import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab
import scipy
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import auc, roc_curve


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    """
    * type-def ::(np.ndarray, List[str], bool, str, matplotlib.colors.Colormap) -> None
    * ---------------{Function}---------------
        * Prints and plots the confusion matrix.
    * ----------------{Returns}---------------
        * -> None
    * ----------------{Params}----------------
        * : cm ::np.ndarray | The confusion matrix
        * : classes ::List[str] | The list of class labels
        * : normalize ::bool | Apply normalization if set to True; default is False
        * : title ::str | The title for the confusion matrix plot; default is 'Confusion matrix'
        * : cmap ::matplotlib.colors.Colormap | The colormap for the confusion matrix plot; default is plt.cm.Blues
    * ----------------{Usage}-----------------
        * >>> import numpy as np
        * >>> cm = np.array([[10, 2], [1, 7]])
        * >>> classes = ['Normal', 'Anomaly']
        * >>> plot_confusion_matrix(cm, classes)
    * ----------------{Notes}-----------------
        * This function helps visualize the confusion matrix, which is useful for understanding the classification performance of a model on a given dataset.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()


def gini(actual, pred, weight=None):
    """
    * type-def ::(np.ndarray, np.ndarray, Optional[np.ndarray]) -> float
    * ---------------{Function}---------------
        * Calculates the Gini coefficient for the given actual and predicted values.
    * ----------------{Returns}---------------
        * -> float | The Gini coefficient
    * ----------------{Params}----------------
        * : actual ::np.ndarray | The actual values (ground truth)
        * : pred ::np.ndarray | The predicted values
        * : weight ::Optional[np.ndarray] | The weights for each observation; default is None
    * ----------------{Usage}-----------------
        * >>> import numpy as np
        * >>> actual = np.array([1, 0, 1, 0, 1])
        * >>> pred = np.array([0.9, 0.1, 0.8, 0.3, 0.7])
        * >>> gini_coefficient = gini(actual, pred)
        * >>> print(gini_coefficient)
    * ----------------{Notes}-----------------
        * The Gini coefficient measures the inequality among values of a frequency distribution, and it is commonly used to measure the performance of classification models. A Gini coefficient of 0 indicates perfect equality, while a Gini coefficient of 1 indicates maximal inequality.
    """
    pdf = pd.DataFrame(scipy.vstack([actual, pred]).T, columns=["Actual", "Predicted"])
    pdf = pdf.sort_values("Predicted")
    if weight is None:
        pdf["Weight"] = 1.0

    pdf["CummulativeWeight"] = np.cumsum(pdf["Weight"])
    pdf["CummulativeWeightedActual"] = np.cumsum(pdf["Actual"] * pdf["Weight"])
    TotalWeight = sum(pdf["Weight"])
    Numerator = sum(pdf["CummulativeWeightedActual"] * pdf["Weight"])
    Denominator = sum(pdf["Actual"] * pdf["Weight"] * TotalWeight)
    Gini = 1.0 - 2.0 * Numerator / Denominator
    return Gini


def lift_df(
    actual,
    pred,
    weight=None,
    n=10,
    xlab="Predicted Decile",
    MyTitle="Model Performance Lift Chart",
):

    pdf = pd.DataFrame(scipy.vstack([actual, pred]).T, columns=["Actual", "Predicted"])
    pdf = pdf.sort_values("Predicted")
    if weight is None:
        pdf["Weight"] = 1.0

    pdf["CummulativeWeight"] = np.cumsum(pdf["Weight"])
    pdf["CummulativeWeightedActual"] = np.cumsum(pdf["Actual"] * pdf["Weight"])
    TotalWeight = sum(pdf["Weight"])
    Numerator = sum(pdf["CummulativeWeightedActual"] * pdf["Weight"])
    Denominator = sum(pdf["Actual"] * pdf["Weight"] * TotalWeight)
    Gini = 1.0 - 2.0 * Numerator / Denominator
    NormalizedGini = Gini / gini(pdf["Actual"], pdf["Actual"])
    GiniTitle = "Normalized Gini = " + str(round(NormalizedGini, 4))

    pdf["PredictedDecile"] = np.round(
        pdf["CummulativeWeight"] * n / TotalWeight + 0.5, decimals=0
    )
    pdf["PredictedDecile"][pdf["PredictedDecile"] < 1.0] = 1.0
    pdf["PredictedDecile"][pdf["PredictedDecile"] > n] = n

    pdf["WeightedPrediction"] = pdf["Predicted"] * pdf["Weight"]
    pdf["WeightedActual"] = pdf["Actual"] * pdf["Weight"]
    lift_df = pdf.groupby("PredictedDecile").agg(
        {
            "WeightedPrediction": np.sum,
            "Weight": np.sum,
            "WeightedActual": np.sum,
            "PredictedDecile": np.size,
        }
    )
    nms = lift_df.columns.values
    nms[1] = "Count"

    lift_df.columns = nms
    lift_df["AveragePrediction"] = lift_df["WeightedPrediction"] / lift_df["Count"]
    lift_df["AverageActual"] = lift_df["WeightedActual"] / lift_df["Count"]
    lift_df["AverageError"] = lift_df["AverageActual"] / lift_df["AveragePrediction"]

    return lift_df


# TODO FIX THIS
def lift_chart(
    actual,
    pred,
    weight=None,
    n=10,
    xlab="Predicted Decile",
    MyTitle="Model Performance Lift Chart",
):
    # From https://github.com/franciscojavierarceo/Python/blob/master/My_Functions.py#L129:18
    pdf = pd.DataFrame(scipy.vstack([actual, pred]).T, columns=["Actual", "Predicted"])
    pdf = pdf.sort(columns="Predicted")
    if weight is None:
        pdf["Weight"] = 1.0

    pdf["CummulativeWeight"] = np.cumsum(pdf["Weight"])
    pdf["CummulativeWeightedActual"] = np.cumsum(pdf["Actual"] * pdf["Weight"])
    TotalWeight = sum(pdf["Weight"])
    Numerator = sum(pdf["CummulativeWeightedActual"] * pdf["Weight"])
    Denominator = sum(pdf["Actual"] * pdf["Weight"] * TotalWeight)
    Gini = 1.0 - 2.0 * Numerator / Denominator
    NormalizedGini = Gini / gini(pdf["Actual"], pdf["Actual"])
    GiniTitle = "Normalized Gini = " + str(round(NormalizedGini, 4))

    pdf["PredictedDecile"] = np.round(
        pdf["CummulativeWeight"] * n / TotalWeight + 0.5, decimals=0
    )
    pdf["PredictedDecile"][pdf["PredictedDecile"] < 1.0] = 1.0
    pdf["PredictedDecile"][pdf["PredictedDecile"] > n] = n

    pdf["WeightedPrediction"] = pdf["Predicted"] * pdf["Weight"]
    pdf["WeightedActual"] = pdf["Actual"] * pdf["Weight"]
    lift_df = pdf.groupby("PredictedDecile").agg(
        {
            "WeightedPrediction": np.sum,
            "Weight": np.sum,
            "WeightedActual": np.sum,
            "PredictedDecile": np.size,
        }
    )
    nms = lift_df.columns.values
    nms[1] = "Count"
    lift_df.columns = nms
    lift_df["AveragePrediction"] = lift_df["WeightedPrediction"] / lift_df["Weight"]
    lift_df["AverageActual"] = lift_df["WeightedActual"] / lift_df["Weight"]
    lift_df["AverageError"] = lift_df["AverageActual"] / lift_df["AveragePrediction"]

    d = pd.DataFrame(lift_df.index)
    p = lift_df["AveragePrediction"]
    a = lift_df["AverageActual"]
    pylab.plot(d, p, label="Predicted", color="blue", marker="o")
    pylab.plot(d, a, label="Actual", color="red", marker="d")
    pylab.legend(["Predicted", "Actual"])
    pylab.title(MyTitle + "\n" + GiniTitle)
    pylab.xlabel(xlab)
    pylab.ylabel("Actual vs. Predicted")
    pylab.grid()
    pylab.show()


def my_auc(actual, pred):
    fpr, tpr, thresholds = metrics.roc_curve(actual, pred)
    return metrics.auc(fpr, tpr)


def deciles(var):
    out = []
    decile = [i * 10 for i in range(0, 11)]
    for i in decile:
        out.append(np.percentile(var, i))

    outdf = pd.DataFrame()
    outdf["Decile"] = decile
    outdf["Value"] = out
    return outdf


def roc_plot(actual, pred, ttl):
    from sklearn.metrics import auc, roc_curve

    fpr, tpr, thresholds = roc_curve(actual, pred)
    roc_auc = auc(fpr, tpr)
    print("The Area Under the ROC Curve : %f" % roc_auc)
    # Plot ROC curve
    plt.clf()
    plt.plot(fpr, tpr, color="red", label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], "k")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.grid()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve" + "\n" + ttl)
    plt.legend(loc="lower right")
    plt.show()


def roc_perf(atrn, ptrn, atst, ptst):
    fprtrn, tprtrn, thresholds = roc_curve(atrn, ptrn)
    fprtst, tprtst, thresholdstst = roc_curve(atst, ptst)
    roc_auctrn = auc(fprtrn, tprtrn)
    roc_auctst = auc(fprtst, tprtst)
    print("The Training Area Under the ROC Curve : %f" % roc_auctrn)
    print("The Test Area Under the ROC Curve : %f" % roc_auctst)
    # Plot ROC curve
    plt.clf()
    plt.plot(fprtrn, tprtrn, color="red", label="Train AUC = %0.2f" % roc_auctrn)
    plt.plot(fprtst, tprtst, color="blue", label="Test AUC = %0.2f" % roc_auctst)
    plt.plot([0, 1], [0, 1], "k")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.grid()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()


def cdfplot(xvar):
    sortedvals = np.sort(xvar)
    yvals = np.arange(len(sortedvals)) / float(len(sortedvals))
    plt.plot(sortedvals, yvals)
    plt.show()


def ptable(df, var, asc=False, topn=100):
    outdf = df.groupby(var).count().reset_index().ix[:, 0:2]
    outdf.columns = [outdf.columns[0], "Count"]
    outdf = outdf.sort_values(by="Count", ascending=asc).reset_index(drop=True)
    outdf["Percent"] = outdf["Count"] / np.sum(outdf["Count"])

    if type(topn) == int:
        outdf = outdf.iloc[0:(topn), :]

    return outdf


def ptablebyv(df, var, sumvar, asc=False):
    outdf = df[[var, sumvar]].groupby(var).sum()
    outdf = outdf.reset_index().ix[:, 0:2]
    outdf.columns = [outdf.columns[0], "Count"]
    if asc == True:
        outdf = outdf.sort(columns="Count", ascending=asc)
    outdf["Percent"] = outdf["Count"] / np.sum(outdf["Count"])
    return outdf


def barplot(df, var, MyTitle="", aval=0.9, prnt=False, prcnt=False, topn=10):
    # Taken from a pandas summary file
    out = ptable(df, var, asc=False, topn=topn)

    if prcnt == True:
        out = out.sort_values("Percent").reset_index()
        out[["Percent"]].plot(kind="barh", figsize=(16, 8))
    else:
        out = out.sort_values("Count").reset_index()
        out[["Count"]].plot(kind="barh", figsize=(16, 8))

    if prnt == True:
        print(out)

    plt.yticks(out.index, out[var])
    plt.xlabel("")
    plt.title(MyTitle)
    plt.grid()
    plt.show()


def learned_frontier(data, classifier, X_train, X_test, savefig=None):
    """
    * type-def ::(pd.DataFrame, Any, np.ndarray, np.ndarray, Optional[str]) -> None
    * ---------------{Function}---------------
        * Plots the learned frontier of a given classifier for novelty detection.
    * ----------------{Returns}---------------
        * -> None
    * ----------------{Params}----------------
        * : data ::pd.DataFrame | The original dataset
        * : classifier ::Any | The classifier model used for novelty detection
        * : X_train ::np.ndarray | The training data
        * : X_test ::np.ndarray | The testing data
        * : savefig ::Optional[str] | The file name to save the plot as an image; default is None
    * ----------------{Usage}-----------------
        * >>> import numpy as np
        * >>> from sklearn.svm import OneClassSVM
        * >>> data = pd.DataFrame(np.random.randn(100, 2))
        * >>> X_train = data.iloc[:80].values
        * >>> X_test = data.iloc[80:].values
        * >>> clf = OneClassSVM()
        * >>> learned_frontier(data, clf, X_train, X_test)
    * ----------------{Notes}-----------------
        * This function helps visualize the learned frontier of a classifier model for novelty detection. The plot shows the learned frontier, the training observations, and the new regular observations. The function can also save the generated plot as an image if the 'savefig' parameter is provided with a file name.
    """
    import matplotlib.pyplot as plt

    data_min = multi_dim_min(data)
    data_min = data_min + data_min / 2
    data_max = multi_dim_max(data)
    data_max = data_max + data_max / 2

    # fit the model for novelty detection
    clf = classifier
    clf.fit(X_train)

    y_pred_test = clf.predict(X_test)

    # xx, yy = np.meshgrid(np.linspace(-30, 30, 5000), np.linspace(-30, 30, 5000))
    xx, yy = np.meshgrid(
        np.linspace(data_min, data_max, 5000), np.linspace(data_min, data_max, 5000)
    )

    print(f"Learning frontier for {type(clf)}")
    # plot the learned frontier, the points, and the nearest vectors to the plane
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.title(f"Learning frontier with  {type(clf)}")
    plt.contourf(
        xx, yy, Z, levels=np.linspace(Z.min() + Z.min() / 4, 10, 7), cmap=plt.cm.PuBu
    )

    a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors="darkred")
    plt.contourf(xx, yy, Z, levels=[0, Z.max() + Z.max() / 4], colors="palevioletred")

    s = 40
    b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c="white", s=s, edgecolors="k")
    b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c="blueviolet", s=s, edgecolors="k")

    plt.axis("tight")
    plt.xlim((data_min, data_max))
    plt.ylim((data_min, data_max))
    plt.legend(
        [a.collections[0], b1, b2],
        [
            "learned frontier",
            "training observations",
            "new regular_observations",
        ],
        loc="upper left",
        prop=matplotlib.font_manager.FontProperties(size=11),
    )

    n_error_test = y_pred_test[y_pred_test == -1].size

    plt.xlabel("errors novel regular: %d/40 ;" % (n_error_test))
    plt.show()

    if savefig != None:
        plt.savefig(savefig)


# Eval utils

#                            Lead score plotting                            #
# ------------------------------------------------------------------------- #

import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


def plotly_lead_score_histogram(
    models, clf1, log_scale=False, start=0.0, threshold=0.3
):
    print(f"\033[33m* Initializing lead score histogram plotting...\033[0m")

    tier_colors = {
        "gold": "#FFD700",
        "silver": "#C0C0C0",
        "bronze": "#CD7F32",
        "copper": "#B87333",
    }

    if not (0 <= start <= 1):
        raise ValueError("\033[31m* Start value must be between 0 and 1.\033[0m")
    if not (0 <= threshold <= 1):
        raise ValueError("\033[31m* Threshold value must be between 0 and 1.\033[0m")

    if (
        isinstance(clf1, pd.DataFrame)
        and "prediction_label" in clf1.columns
        and "prediction_score_1" in clf1.columns
    ):
        print(f"\033[33m* Pre-scored DataFrame detected. Preparing data...\033[0m")
        data = prepare_dataframe(clf1)
    else:
        if not all(
            hasattr(clf1, attr) for attr in ["X_test_transformed", "y_test_transformed"]
        ):
            raise ValueError(
                "\033[31m* clf1 must either be a DataFrame with necessary columns or an object with 'X_test_transformed' and 'y_test_transformed' attributes.\033[0m"
            )
        print(f"\033[33m* Model data detected. Generating predictions...\033[0m")
        data = generate_predictions_dataframe(models, clf1)

    performance_metrics = {}

    for model_name in models.keys():
        print(f"\033[33m* Processing model: {model_name}\033[0m")

        data_filtered = data[data["Score"] >= start]

        # Assign tiers using the new function
        data_filtered = assign_tiers(data_filtered, start)

        total_converted = sum(data_filtered["Outcome"] == "Converted")
        total_leads = len(data_filtered)
        print(
            f"\033[33m* Total leads considered: {total_leads}, Total converted: {total_converted}\033[0m"
        )

        fig = go.Figure()

        for tier, color in tier_colors.items():
            for outcome in ["Converted", "Not Converted"]:
                df_subset = data_filtered[
                    (data_filtered["Tier"] == tier)
                    & (data_filtered["Outcome"] == outcome)
                ]
                print(
                    f"\033[33m* Plotting {tier.capitalize()} - {outcome}: {len(df_subset)} leads\033[0m"
                )
                fig.add_trace(
                    go.Histogram(
                        x=df_subset["Score"],
                        name=f"{tier.capitalize()} - {outcome}",
                        marker_color=color,
                        opacity=0.75 if outcome == "Converted" else 0.5,
                        nbinsx=20,
                    )
                )

        if log_scale:
            print(f"\033[33m* Applying log scale to the y-axis\033[0m")
            fig.update_yaxes(type="log")

        add_plotly_annotations_and_overlays(fig, data_filtered, start, tier_colors)

        fig.update_layout(
            title=f"Lead Score Distribution and Performance Metrics - {model_name}",
            xaxis_title="Lead Scores",
            yaxis_title="Count",
            barmode="overlay",
            xaxis=dict(range=[start, 1]),
            legend_title_text="Tier - Outcome",
        )

        fig.show()

        metrics = calculate_performance_metrics(data_filtered, threshold, model_name)
        performance_metrics[model_name] = metrics
        print(f"\033[33m* Performance metrics for {model_name}: {metrics}\033[0m")

        plot_confusion_matrix(data_filtered, threshold, model_name)

    plot_performance_metrics(performance_metrics)


def generate_predictions_dataframe(models, clf1):
    scores_list = []
    for model_name, model in models.items():
        print(f"\033[33m* Generating predictions for model: {model_name}\033[0m")
        scores = model.predict_proba(clf1.X_test_transformed)[:, 1]
        scores = np.where(scores == 0, 1e-10, scores)
        outcomes = np.where(clf1.y_test_transformed == 1, "Converted", "Not Converted")
        df = pd.DataFrame({"Model": model_name, "Score": scores, "Outcome": outcomes})
        scores_list.append(df)
    return pd.concat(scores_list)


def prepare_dataframe(clf1):
    print(f"\033[33m* Preparing pre-scored DataFrame...\033[0m")
    data = clf1.copy()
    if (
        "prediction_label" not in data.columns
        or "prediction_score_1" not in data.columns
    ):
        raise ValueError(
            "\033[31m* DataFrame must contain 'prediction_label' and 'prediction_score_1' columns.\033[0m"
        )
    data.rename(
        columns={"opportunity_target": "Outcome", "prediction_score_1": "Score"},
        inplace=True,
    )
    data["Outcome"] = data["Outcome"].apply(
        lambda x: "Converted" if x == 1 else "Not Converted"
    )
    data["Score"] = np.where(data["Score"] == 0, 1e-10, data["Score"])
    return data


def assign_tiers(data, start):
    print(f"\033[33m* Assigning tiers based on scores...\033[0m")
    data["Tier"] = pd.cut(
        data["Score"],
        bins=[start, 0.1, 0.30, 0.75, 1],
        labels=["copper", "bronze", "silver", "gold"],
        right=False,
    )

    # Debugging: Check how leads are being distributed into tiers
    tier_counts = data["Tier"].value_counts().to_dict()
    print(f"\033[33m* Tier Distribution:\n{tier_counts}\033[0m")

    return data


def add_plotly_annotations_and_overlays(fig, data, start, tier_colors):
    print(f"\033[33m* Adding annotations and overlays to the plot...\033[0m")
    total_positives = sum(data["Outcome"] == "Converted")
    tiers = ["copper", "bronze", "silver", "gold"]

    y_coord_base = 1.05
    y_coord_step = 0.1

    for tier in tiers:
        tier_data = data[data["Tier"] == tier]
        total_in_tier = len(tier_data)
        captured = sum(tier_data["Outcome"] == "Converted")
        proportion_captured = captured / total_positives if total_positives > 0 else 0
        conversion_rate = (captured / total_in_tier) * 100 if total_in_tier > 0 else 0

        lower_bound = tier_data["Score"].min() if not tier_data.empty else start
        upper_bound = tier_data["Score"].max() if not tier_data.empty else start
        x_position = (lower_bound + upper_bound) / 2

        fig.add_vrect(
            x0=lower_bound,
            x1=upper_bound,
            fillcolor=tier_colors[tier],
            opacity=0.2,
            layer="below",
            line_width=0,
        )

        y_coord = y_coord_base

        fig.add_annotation(
            x=x_position,
            y=y_coord,
            xref="x",
            yref="paper",
            text=f"{tier.capitalize()}: {proportion_captured:.2%} captured",
            showarrow=False,
            align="center",
            bgcolor="rgba(255,255,255,0.6)",
        )

        y_coord -= y_coord_step
        fig.add_annotation(
            x=x_position,
            y=y_coord,
            xref="x",
            yref="paper",
            text=f"{conversion_rate:.2f}% conversion",
            showarrow=False,
            align="center",
            bgcolor="rgba(255,255,255,0.6)",
        )
        y_coord_base -= 2 * y_coord_step


def calculate_performance_metrics(data, threshold, model_name):
    print(f"\033[33m* Calculating performance metrics for {model_name}...\033[0m")
    y_true = data["Outcome"] == "Converted"
    y_pred = data["Score"] >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )
    print(
        f"\033[33m* Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}\033[0m"
    )
    return {"F1": f1, "Precision": precision, "Recall": recall}


def plot_performance_metrics(metrics):
    print(f"\033[33m* Plotting performance metrics...\033[0m")
    fig = go.Figure()
    for model, metric_values in metrics.items():
        for metric, value in metric_values.items():
            print(f"\033[33m* {model} - {metric}: {value:.4f}\033[0m")
            fig.add_trace(
                go.Bar(
                    name=f"{model} - {metric}",
                    x=[metric],
                    y=[value],
                    text=value,
                    textposition="auto",
                )
            )

    fig.update_layout(
        title="Performance Metrics at Specified Threshold",
        xaxis_title="Metric",
        yaxis_title="Value",
        barmode="group",
    )
    fig.show()


def plot_confusion_matrix(data, threshold, model_name):
    print(f"\033[33m* Plotting confusion matrix for {model_name}...\033[0m")
    y_true = (data["Outcome"] == "Converted").astype(int)
    y_pred = (data["Score"] >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\033[33m* Confusion Matrix:\n{cm}\033[0m")

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Predicted Not Converted", "Predicted Converted"],
            y=["Actual Not Converted", "Actual Converted"],
            hoverongaps=False,
            colorscale="Viridis",
            texttemplate="%{z}",
            textfont={"size": 20},
        )
    )

    fig.update_layout(
        title=f"Confusion Matrix - {model_name}",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
    )

    fig.show()


#                             Feature importance                            #
# ------------------------------------------------------------------------- #


#                        Imbalanced evaluation report                       #
# ------------------------------------------------------------------------- #


def evaluate_classification(pred, df, target_column, threshold=0.5):
    import pandas as pd
    import numpy as np
    from sklearn.metrics import (
        confusion_matrix,
        accuracy_score,
        precision_recall_fscore_support,
        classification_report,
        balanced_accuracy_score,
        mean_absolute_error,
    )
    from imblearn.metrics import (
        classification_report_imbalanced,
        geometric_mean_score,
        macro_averaged_mean_absolute_error,
        sensitivity_specificity_support,
    )
    import matplotlib.pyplot as plt
    import seaborn as sns

    y_true = df[target_column].astype(int)
    y_scores = pred["prediction_score_1"]
    y_pred = (y_scores >= threshold).astype(int)

    conf_matrix = confusion_matrix(y_true, y_pred)
    conf_matrix_df = pd.DataFrame(
        conf_matrix,
        index=["Actual Negative", "Actual Positive"],
        columns=["Predicted Negative", "Predicted Positive"],
    )

    print("\033[48;5;0;38;5;226m* CONFUSION MATRIX *\033[0m")
    print(conf_matrix_df)
    print("\n")

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    print("\033[48;5;0;38;5;82m* ACCURACY *\033[0m")
    print(f"\033[38;5;82mAccuracy:\033[0m {accuracy:.2f}")
    print("\033[38;5;243m- Proportion of correct predictions.\033[0m\n")

    print("\033[48;5;0;38;5;208m* PRECISION *\033[0m")
    print(f"\033[38;5;208mPrecision:\033[0m {precision:.2f}")
    print(
        "\033[38;5;243m- Proportion of true positive predictions among all positive predictions.\033[0m\n"
    )

    print("\033[48;5;0;38;5;45m* RECALL *\033[0m")
    print(f"\033[38;5;45mRecall:\033[0m {recall:.2f}")
    print(
        "\033[38;5;243m- Proportion of actual positives correctly identified.\033[0m\n"
    )

    print("\033[48;5;0;38;5;201m* F1 SCORE *\033[0m")
    print(f"\033[38;5;201mF1 Score:\033[0m {f1:.2f}")
    print("\033[38;5;243m- Harmonic mean of Precision and Recall.\033[0m\n")

    print("\033[48;5;0;38;5;160m* CLASSIFICATION REPORT *\033[0m")
    print("\033[38;5;160m- Detailed classification report:\033[0m")
    print(classification_report(y_true, y_pred))
    print(
        "\033[38;5;243m- Includes precision, recall, and F1 score for each class.\033[0m\n"
    )

    report_dict = classification_report_imbalanced(y_true, y_pred, output_dict=True)

    sup = report_dict.get("sup")
    avg_pre = report_dict.get("avg_pre")
    avg_rec = report_dict.get("avg_rec")
    avg_spe = report_dict.get("avg_spe")
    avg_f1 = report_dict.get("avg_f1")
    avg_geo = report_dict.get("avg_geo")
    avg_iba = report_dict.get("avg_iba")
    total_support = report_dict.get("total_support")

    print("\033[48;5;0;38;5;214m* IMBALANCED CLASSIFICATION REPORT *\033[0m")
    print("\033[38;5;214m- Imbalanced classification report:\033[0m")
    print("\033[38;5;214m- Extracted Metrics:\033[0m")
    print(f"\033[38;5;243m  sup: {sup}\033[0m")
    print(f"\033[38;5;243m  avg_pre: {avg_pre:.6f}\033[0m")
    print(f"\033[38;5;243m  avg_rec: {avg_rec:.6f}\033[0m")
    print(f"\033[38;5;243m  avg_spe: {avg_spe:.6f}\033[0m")
    print(f"\033[38;5;243m  avg_f1: {avg_f1:.6f}\033[0m")
    print(f"\033[38;5;243m  avg_geo: {avg_geo:.6f}\033[0m")
    print(f"\033[38;5;243m  avg_iba: {avg_iba:.6f}\033[0m")
    print(f"\033[38;5;243m  total_support: {total_support}\033[0m")
    print("\033[38;5;243m- Tailored for imbalanced datasets.\033[0m\n")

    gmean = geometric_mean_score(y_true, y_pred)
    bal_accuracy = balanced_accuracy_score(y_true, y_pred)
    mame = macro_averaged_mean_absolute_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    sensitivity, specificity, _ = sensitivity_specificity_support(
        y_true, y_pred, average="binary"
    )

    print("\033[48;5;0;38;5;219m* GEOMETRIC MEAN *\033[0m")
    print(f"\033[38;5;219mGeometric Mean:\033[0m {gmean:.3f}")
    print("\033[38;5;243m- Balance between sensitivity and specificity.\033[0m\n")

    print("\033[48;5;0;38;5;39m* BALANCED ACCURACY *\033[0m")
    print(f"\033[38;5;39mBalanced Accuracy:\033[0m {bal_accuracy:.2f}")
    print("\033[38;5;243m- Accuracy that accounts for class imbalance.\033[0m\n")

    print("\033[48;5;0;38;5;135m* MACRO-AVERAGED MAE *\033[0m")
    print(f"\033[38;5;135mMacro-Averaged MAE:\033[0m {mame:.3f}")
    print("\033[38;5;243m- Mean Absolute Error across classes.\033[0m\n")

    print("\033[48;5;0;38;5;226m* MEAN ABSOLUTE ERROR *\033[0m")
    print(f"\033[38;5;226mMean Absolute Error:\033[0m {mae:.3f}")
    print("\033[38;5;243m- Average magnitude of errors in predictions.\033[0m\n")

    print("\033[48;5;0;38;5;87m* SENSITIVITY & SPECIFICITY *\033[0m")
    print(f"\033[38;5;87mSensitivity:\033[0m {sensitivity:.3f}")
    print(f"\033[38;5;87mSpecificity:\033[0m {specificity:.3f}")
    print("\033[38;5;243m- Sensitivity: True positive rate.\033[0m")
    print("\033[38;5;243m- Specificity: True negative rate.\033[0m\n")


#                               Extra Metrics                               #
# ------------------------------------------------------------------------- #
from sklearn.metrics import fbeta_score


def f2_score(y_true, y_pred, **kwargs):
    """
    Calculate the F2 score.

    Args:
        y_true (1d array-like): The true labels.
        y_pred (1d array-like): The predicted labels.
        **kwargs: Additional arguments for fbeta_score.

    Returns:
        float: The F2 score.
    """
    return fbeta_score(y_true, y_pred, beta=2, **kwargs)


# ------------------------------------------------------------------------- #


import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


def plot_shap_values(
    shap_values,
    feature_names,
    feature_values,
    prefixes_to_merge=[
        "embeddings_",
        "UMAP",
        "channeltype",
        "cm_",
        "tp",
        "status",
        "campaign",
        "count",
    ],
    contains_to_merge=[],
    contains_to_ignore=[],
    top_n=20,
    height=800,
    output_filename="shap_plot",
):
    shap_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "SHAP Value": shap_values,
            "Feature Value": feature_values,
        }
    )
    shap_df["Abs SHAP Value"] = shap_df["SHAP Value"].abs()

    def merge_features(
        shap_df, prefixes_to_merge, contains_to_merge, contains_to_ignore
    ):
        merged_scores = {}
        hover_info = {}
        feature_count = {}
        feature_medians = {}
        top_contributors = {}
        original_shap_values = (
            {}
        )  # Store the SHAP values of the original features for plotting shadows

        def process_features(matching_features, label):
            if not matching_features.empty:
                merged_scores[label] = matching_features["SHAP Value"].sum()
                hover_info[label] = "<br>".join(
                    f"{feat} = {val}"
                    for feat, val in zip(
                        matching_features["Feature"], matching_features["Feature Value"]
                    )
                )
                feature_count[label] = len(matching_features)
                feature_medians[label] = matching_features["SHAP Value"].median()
                top_contributors[label] = matching_features.loc[
                    matching_features["Abs SHAP Value"].idxmax(), "Feature"
                ]
                original_shap_values[label] = matching_features[
                    ["SHAP Value", "Feature"]
                ].copy()  # Store original SHAP values
                return True
            return False

        for substring in contains_to_ignore:
            shap_df = shap_df[~shap_df["Feature"].str.contains(substring, case=False)]

        for item in contains_to_merge:
            if isinstance(item, tuple):
                substring, label = item
            else:
                substring, label = item, f"aggregate_contains_{item}"

            matching_features = shap_df[
                shap_df["Feature"].str.contains(substring, case=False)
            ]
            if process_features(matching_features, label):
                shap_df = shap_df[
                    ~shap_df["Feature"].str.contains(substring, case=False)
                ]

        for item in prefixes_to_merge:
            if isinstance(item, tuple):
                prefix, label = item
            else:
                prefix, label = item, f"aggregate_{item}"

            matching_features = shap_df[shap_df["Feature"].str.startswith(prefix)]
            if process_features(matching_features, label):
                shap_df = shap_df[~shap_df["Feature"].str.startswith(prefix)]

        merged_df = pd.DataFrame(
            {
                "Feature": list(merged_scores.keys()),
                "SHAP Value": list(merged_scores.values()),
                "Feature Value": ["Aggregated"] * len(merged_scores),
                "Abs SHAP Value": list(abs(value) for value in merged_scores.values()),
            }
        )

        shap_df = pd.concat([shap_df, merged_df]).reset_index(drop=True)
        return (
            shap_df,
            hover_info,
            feature_count,
            feature_medians,
            top_contributors,
            original_shap_values,
        )

    (
        shap_df,
        hover_info,
        feature_count,
        feature_medians,
        top_contributors,
        original_shap_values,
    ) = merge_features(
        shap_df, prefixes_to_merge, contains_to_merge, contains_to_ignore
    )

    shap_df = shap_df.sort_values(by="Abs SHAP Value", ascending=False).head(top_n)
    shap_df = shap_df.sort_values(by="SHAP Value", ascending=True).reset_index(
        drop=True
    )

    def truncate_label(label):
        return (label[:47] + "...") if len(label) > 50 else label

    shap_df["Feature"] = shap_df.apply(
        lambda row: (
            truncate_label(
                f"{row['Feature']}[{top_contributors.get(row['Feature'], '')}]"
            )
            + f" <span style='color:green;'>+{feature_count.get(row['Feature'], 0)}</span>"
            if row["Feature"] in hover_info
            else truncate_label(f"{row['Feature']} = {row['Feature Value']}")
        ),
        axis=1,
    )

    zebra_shapes = []
    for i in range(len(shap_df)):
        zebra_shapes.append(
            dict(
                type="rect",
                x0=min(shap_df["SHAP Value"].min(), 0) * 1.1,
                y0=i - 0.5,
                x1=max(shap_df["SHAP Value"].max(), 0) * 1.1,
                y1=i + 0.5,
                fillcolor="lightgray" if i % 2 == 0 else "white",
                opacity=0.2,
                layer="below",
                line_width=0,
            )
        )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=shap_df["SHAP Value"],
            y=shap_df.index,
            mode="markers+lines",
            name="SHAP Value",
            line=dict(color="rgba(100, 100, 100, 0.5)", width=0.5),
            marker=dict(
                color="rgba(255, 165, 0, 0.8)",
                size=24,
                line=dict(color="black", width=1.5),
            ),
            text=shap_df.apply(
                lambda row: hover_info.get(
                    row["Feature"].split("[")[0],
                    f"{row['Feature']} = {row['Feature Value']}",
                ),
                axis=1,
            ),
            hoverinfo="text",
        )
    )

    # Adding green circles as shadow values for aggregates
    for label, original_features in original_shap_values.items():
        # Check if there are any matching rows in shap_df
        matching_indices = shap_df[shap_df["Feature"].str.startswith(label)].index

        if not matching_indices.empty:
            y_index = matching_indices[0]  # Safely access the first index
            for _, row in original_features.iterrows():
                fig.add_trace(
                    go.Scatter(
                        x=[row["SHAP Value"]],
                        y=[y_index],
                        mode="markers",
                        marker=dict(color="rgba(0, 128, 0, 0.5)", size=12),
                        showlegend=False,
                        hovertext=f"Original Feature: {row['Feature']}<br>SHAP Value: {row['SHAP Value']:.2f}",
                        hoverinfo="text",
                    )
                )

    for i in range(len(shap_df)):
        fig.add_annotation(
            x=shap_df["SHAP Value"][i],
            y=i,
            text=f"{shap_df['SHAP Value'][i]:.2f}",
            showarrow=False,
            font=dict(size=10, color="black"),
        )

    for feature in hover_info:
        top_features = hover_info[feature].split("<br>")
        expected_feature_name = (
            truncate_label(f"{feature}[{top_contributors[feature]}]")
            + f" <span style='color:green;'>+{feature_count[feature]}</span>"
        )
        if expected_feature_name in shap_df["Feature"].values:
            y_index = shap_df[shap_df["Feature"] == expected_feature_name].index[0]
            for top_feature in top_features[:5]:
                feat_name = top_feature.split(" = ")[0]
                if feat_name in shap_df["Feature"].values:
                    importance = shap_df.loc[
                        shap_df["Feature"] == feat_name, "SHAP Value"
                    ].values[0]
                    fig.add_trace(
                        go.Scatter(
                            x=[importance],
                            y=[y_index],
                            mode="markers",
                            marker=dict(color="rgba(0, 128, 0, 0.6)", size=16),
                            showlegend=False,
                            hovertext=top_feature,
                            hoverinfo="text",
                        )
                    )

    for i, feature in enumerate(shap_df["Feature"]):
        original_feature = (
            feature.split("[")[0]
            .replace("<span style='color:green;'>", "")
            .replace("</span>", "")
        )
        if original_feature in feature_medians:
            fig.add_trace(
                go.Scatter(
                    x=[
                        feature_medians[original_feature],
                        feature_medians[original_feature],
                    ],
                    y=[i - 0.5, i + 0.5],
                    mode="lines",
                    line=dict(color="red", width=1.5, dash="dot"),
                    name=f"{original_feature} Median",
                    hoverinfo="text",
                    hovertext=f"{original_feature} Median: {feature_medians[original_feature]:.2f}",
                )
            )

    non_aggregate_median = shap_df.loc[
        ~shap_df["Feature"].str.startswith("aggregate_")
    ]["SHAP Value"].median()

    fig.add_trace(
        go.Scatter(
            x=[non_aggregate_median, non_aggregate_median],
            y=[-0.5, len(shap_df) - 0.5],
            mode="lines",
            line=dict(color="red", width=2, dash="dash"),
            name="Non-Aggregate Median",
            hoverinfo="text",
            hovertext=f"Non-Aggregate Median: {non_aggregate_median:.2f}",
            opacity=0.25,
        )
    )

    ticktext = []
    for i, feature in enumerate(shap_df["Feature"]):
        color = "darkblue" if i % 2 == 0 else "darkorange"
        ticktext.append(f"<span style='color:{color};'>{feature}</span>")

    fig.update_layout(
        title="SHAP Values for a Single Prediction",
        xaxis_title="SHAP Value",
        yaxis=dict(
            tickvals=list(range(len(shap_df))),
            ticktext=ticktext,
            showgrid=False,
            tickfont=dict(size=14),
            titlefont=dict(size=16),
        ),
        height=height,
        width=1400,
        template="plotly_white",
        xaxis=dict(
            showgrid=True,
            gridcolor="lightgray",
            tickfont=dict(size=14),
            titlefont=dict(size=16),
        ),
        font=dict(size=16),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=300, r=50, t=70, b=40),
        shapes=zebra_shapes,
    )

    pio.write_html(fig, file=f"{output_filename}.html", auto_open=False)
    fig.write_image(f"{output_filename}.png")

    fig.show()


def pprint_df(data, float_format="{:.4f}", header_style="bold magenta"):
    """
    Enhanced pretty print for pandas DataFrame, DataFrame columns (Pandas Index), or lists,
    using the rich library for improved formatting and flexibility.

    :param data: The data to print. Can be a pandas DataFrame, DataFrame columns, or a list.
    :param float_format: The format string for float values. Defaults to "{:.4f}".
    :param header_style: The style of the table header. Defaults to "bold magenta".
    """
    from rich.console import Console
    from rich.table import Table
    import pandas as pd

    def format_large_number(val):
        """Converts a large number to a string representation with K, M, B suffixes."""
        if isinstance(val, int) or isinstance(val, float):
            abs_val = abs(val)
            if abs_val >= 1_000_000_000:
                return f"{val / 1_000_000_000:.2f}B"
            elif abs_val >= 1_000_000:
                return f"{val / 1_000_000:.2f}M"
            elif abs_val >= 1_000:
                return f"{val / 1_000:.2f}K"
            else:
                return float_format.format(val) if isinstance(val, float) else str(val)
        return str(val)

    console = Console()
    table = Table(show_header=True, header_style=header_style)

    if isinstance(data, pd.DataFrame):
        # Add columns to the table for DataFrame
        for column in data.columns:
            table.add_column(column)
        # Format float columns and convert all values to strings for DataFrame
        for _, row in data.iterrows():
            formatted_row = [format_large_number(val) for val in row]
            table.add_row(*formatted_row)
    elif isinstance(data, pd.Index) or isinstance(data, list):
        # Handle a single row of data for Pandas Index or list
        table.add_column("Values")
        for item in data:
            table.add_row(format_large_number(item))
    else:
        # Handle unsupported types
        console.print(
            "[bold red]Unsupported data type. Please provide a DataFrame, a DataFrame columns (Index), or a list.[/bold red]"
        )
        return
    # Print the table
    console.print(table)


def prepare_shap_single_data(model, raw_df, pipeline, inference_cols, idx):
    import shap

    single_instance_raw = raw_df[inference_cols].iloc[idx : idx + 1]
    single_instance = pipeline.transform(single_instance_raw)
    columns = single_instance.columns

    explainer = shap.TreeExplainer(model)
    shap_values_single = explainer.shap_values(single_instance)

    lowest_indices = np.argsort(shap_values_single.flatten())[:5]
    top_indices = np.argsort(shap_values_single.flatten())[-5:]

    lowest_columns = list(single_instance.columns[lowest_indices])
    top_columns = list(single_instance.columns[top_indices])

    # cols_to_show = list(dict.fromkeys(cols_to_show))

    print(model.predict_proba(single_instance)[0][1])
    pprint_df(
        raw_df[
            [
                "lead_contact_title",
                "tp_geo",
                "zi_name",
                "tp_revenue",
                "opportunity_target",
            ]
        ].iloc[idx : idx + 1]
    )
    print("")
    pprint_df(
        single_instance[get_matching_columns(single_instance, "pfrev_segment")].iloc[
            idx : idx + 1
        ]
    )

    # pprint_df(single_instance_raw[ list(set(top_columns).intersection( set(single_instance_raw.columns)))])
    # print("Lowest impact/negative impact")
    # pprint_df(single_instance_raw[ list(set(lowest_columns).intersection( set(single_instance_raw.columns)))])

    return shap_values_single, single_instance


import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


def plot_lime_explanation(
    lime_explanation,
    feature_names,
    prefixes_to_merge=[
        "embeddings_",
        "UMAP",
        "channeltype",
        "cm_",
        "tp",
        "status",
        "campaign",
        "count",
    ],
    contains_to_merge=[],
    contains_to_ignore=[],
    top_n=20,
    height=800,
    output_filename="lime_plot_single_instance",
):
    exp = lime_explanation.as_list()
    exp_df = pd.DataFrame(exp, columns=["Feature", "Contribution"])
    exp_df["Abs Contribution"] = exp_df["Contribution"].abs()

    def merge_features(
        exp_df, prefixes_to_merge, contains_to_merge, contains_to_ignore
    ):
        merged_scores = {}
        hover_info = {}
        feature_count = {}
        feature_medians = {}
        top_contributors = {}

        def process_features(matching_features, label):
            if not matching_features.empty:
                merged_scores[label] = matching_features["Contribution"].sum()
                hover_info[label] = "<br>".join(
                    f"{feat} = {val:.2f}"
                    for feat, val in zip(
                        matching_features["Feature"], matching_features["Contribution"]
                    )
                )
                feature_count[label] = len(matching_features)
                feature_medians[label] = matching_features["Contribution"].median()
                top_contributors[label] = matching_features.loc[
                    matching_features["Abs Contribution"].idxmax(), "Feature"
                ]
                return True
            return False

        # Ignore features containing specified substrings
        for substring in contains_to_ignore:
            exp_df = exp_df[~exp_df["Feature"].str.contains(substring, case=False)]

        # Merge features containing specified substrings
        for item in contains_to_merge:
            if isinstance(item, tuple):
                substring, label = item
            else:
                substring, label = item, f"aggregate_contains_{item}"

            matching_features = exp_df[
                exp_df["Feature"].str.contains(substring, case=False)
            ]
            if process_features(matching_features, label):
                exp_df = exp_df[~exp_df["Feature"].str.contains(substring, case=False)]

        # Merge features starting with specified prefixes
        for item in prefixes_to_merge:
            if isinstance(item, tuple):
                prefix, label = item
            else:
                prefix, label = item, f"aggregate_{item}"

            matching_features = exp_df[exp_df["Feature"].str.startswith(prefix)]
            if process_features(matching_features, label):
                exp_df = exp_df[~exp_df["Feature"].str.startswith(prefix)]

        # Debugging print statements to verify merging
        print(f"Original features: {exp_df['Feature'].tolist()}")
        print(f"Merged features: {list(merged_scores.keys())}")

        merged_df = pd.DataFrame(
            {
                "Feature": list(merged_scores.keys()),
                "Contribution": list(merged_scores.values()),
                "Abs Contribution": list(
                    abs(value) for value in merged_scores.values()
                ),
            }
        )

        exp_df = pd.concat([exp_df, merged_df]).reset_index(drop=True)
        return exp_df, hover_info, feature_count, feature_medians, top_contributors

    exp_df, hover_info, feature_count, feature_medians, top_contributors = (
        merge_features(exp_df, prefixes_to_merge, contains_to_merge, contains_to_ignore)
    )

    exp_df = exp_df.sort_values(by="Abs Contribution", ascending=False).head(top_n)
    exp_df = exp_df.sort_values(by="Contribution", ascending=True).reset_index(
        drop=True
    )

    def truncate_label(label):
        return (label[:47] + "...") if len(label) > 50 else label

    exp_df["Feature"] = exp_df.apply(
        lambda row: (
            truncate_label(
                f"{row['Feature']}[{top_contributors.get(row['Feature'], '')}]"
            )
            + f" <span style='color:green;'>+{feature_count.get(row['Feature'], 0)}</span>"
            if row["Feature"] in hover_info
            else truncate_label(f"{row['Feature']} = {row['Contribution']:.2f}")
        ),
        axis=1,
    )

    zebra_shapes = []
    for i in range(len(exp_df)):
        zebra_shapes.append(
            dict(
                type="rect",
                x0=min(exp_df["Contribution"].min(), 0) * 1.1,
                y0=i - 0.5,
                x1=max(exp_df["Contribution"].max(), 0) * 1.1,
                y1=i + 0.5,
                fillcolor="lightgray" if i % 2 == 0 else "white",
                opacity=0.2,
                layer="below",
                line_width=0,
            )
        )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=exp_df["Contribution"],
            y=exp_df.index,
            mode="markers+lines",
            name="Contribution",
            line=dict(color="rgba(100, 100, 100, 0.5)", width=0.5),
            marker=dict(
                color="rgba(255, 165, 0, 0.8)",
                size=24,
                line=dict(color="black", width=1.5),
            ),
            text=exp_df.apply(
                lambda row: hover_info.get(
                    row["Feature"].split("[")[0],
                    f"{row['Feature']} = {row['Contribution']:.2f}",
                ),
                axis=1,
            ),
            hoverinfo="text",
        )
    )

    for i in range(len(exp_df)):
        fig.add_annotation(
            x=exp_df["Contribution"][i],
            y=i,
            text=f"{exp_df['Contribution'][i]:.2f}",
            showarrow=False,
            font=dict(size=10, color="black"),
        )

    ticktext = []
    for i, feature in enumerate(exp_df["Feature"]):
        color = "darkblue" if i % 2 == 0 else "darkorange"
        ticktext.append(f"<span style='color:{color};'>{feature}</span>")

    fig.update_layout(
        title="LIME Explanation for a Single Prediction",
        xaxis_title="Contribution",
        yaxis=dict(
            tickvals=list(range(len(exp_df))),
            ticktext=ticktext,
            showgrid=False,
            tickfont=dict(size=14),
            titlefont=dict(size=16),
        ),
        height=height,
        width=1400,
        template="plotly_white",
        xaxis=dict(
            showgrid=True,
            gridcolor="lightgray",
            tickfont=dict(size=14),
            titlefont=dict(size=16),
        ),
        font=dict(size=16),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=300, r=50, t=70, b=40),
        shapes=zebra_shapes,
    )

    pio.write_html(fig, file=f"{output_filename}.html", auto_open=False)
    fig.write_image(f"{output_filename}.png")

    fig.show()


import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


def plot_lime_explanation(
    lime_explanation,
    feature_names,
    prefixes_to_merge=[
        "embeddings_",
        "UMAP",
        "channeltype",
        "cm_",
        "tp",
        "status",
        "campaign",
        "count",
    ],
    contains_to_merge=[],
    contains_to_ignore=[],
    top_n=20,
    height=800,
    output_filename="lime_plot_single_instance",
):
    exp = lime_explanation.as_list()
    exp_df = pd.DataFrame(exp, columns=["Feature", "Contribution"])
    exp_df["Abs Contribution"] = exp_df["Contribution"].abs()

    def merge_features(
        exp_df, prefixes_to_merge, contains_to_merge, contains_to_ignore
    ):
        merged_scores = {}
        hover_info = {}
        feature_count = {}
        feature_medians = {}
        top_contributors = {}

        def process_features(matching_features, label):
            if not matching_features.empty:
                merged_scores[label] = matching_features["Contribution"].sum()
                hover_info[label] = "<br>".join(
                    f"{feat} = {val}"
                    for feat, val in zip(
                        matching_features["Feature"], matching_features["Contribution"]
                    )
                )
                feature_count[label] = len(matching_features)
                feature_medians[label] = matching_features["Contribution"].median()
                top_contributors[label] = matching_features.loc[
                    matching_features["Abs Contribution"].idxmax(), "Feature"
                ]
                return True
            return False

        for substring in contains_to_ignore:
            exp_df = exp_df[~exp_df["Feature"].str.contains(substring, case=False)]

        for item in contains_to_merge:
            if isinstance(item, tuple):
                substring, label = item
            else:
                substring, label = item, f"aggregate_contains_{item}"

            matching_features = exp_df[
                exp_df["Feature"].str.contains(substring, case=False)
            ]
            if process_features(matching_features, label):
                exp_df = exp_df[~exp_df["Feature"].str.contains(substring, case=False)]

        for item in prefixes_to_merge:
            if isinstance(item, tuple):
                prefix, label = item
            else:
                prefix, label = item, f"aggregate_{item}"

            matching_features = exp_df[exp_df["Feature"].str.startswith(prefix)]
            if process_features(matching_features, label):
                exp_df = exp_df[~exp_df["Feature"].str.startswith(prefix)]

        merged_df = pd.DataFrame(
            {
                "Feature": list(merged_scores.keys()),
                "Contribution": list(merged_scores.values()),
                "Abs Contribution": list(
                    abs(value) for value in merged_scores.values()
                ),
            }
        )

        exp_df = pd.concat([exp_df, merged_df]).reset_index(drop=True)
        return exp_df, hover_info, feature_count, feature_medians, top_contributors

    exp_df, hover_info, feature_count, feature_medians, top_contributors = (
        merge_features(exp_df, prefixes_to_merge, contains_to_merge, contains_to_ignore)
    )

    exp_df = exp_df.sort_values(by="Abs Contribution", ascending=False).head(top_n)
    exp_df = exp_df.sort_values(by="Contribution", ascending=True).reset_index(
        drop=True
    )

    def truncate_label(label):
        return (label[:47] + "...") if len(label) > 50 else label

    exp_df["Feature"] = exp_df.apply(
        lambda row: (
            truncate_label(
                f"{row['Feature']}[{top_contributors.get(row['Feature'], '')}]"
            )
            + f" <span style='color:green;'>+{feature_count.get(row['Feature'], 0)}</span>"
            if row["Feature"] in hover_info
            else truncate_label(f"{row['Feature']} = {row['Contribution']:.2f}")
        ),
        axis=1,
    )

    zebra_shapes = []
    for i in range(len(exp_df)):
        zebra_shapes.append(
            dict(
                type="rect",
                x0=min(exp_df["Contribution"].min(), 0) * 1.1,
                y0=i - 0.5,
                x1=max(exp_df["Contribution"].max(), 0) * 1.1,
                y1=i + 0.5,
                fillcolor="lightgray" if i % 2 == 0 else "white",
                opacity=0.2,
                layer="below",
                line_width=0,
            )
        )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=exp_df["Contribution"],
            y=exp_df.index,
            mode="markers+lines",
            name="Contribution",
            line=dict(color="rgba(100, 100, 100, 0.5)", width=0.5),
            marker=dict(
                color="rgba(255, 165, 0, 0.8)",
                size=24,
                line=dict(color="black", width=1.5),
            ),
            text=exp_df.apply(
                lambda row: hover_info.get(
                    row["Feature"].split("[")[0],
                    f"{row['Feature']} = {row['Contribution']:.2f}",
                ),
                axis=1,
            ),
            hoverinfo="text",
        )
    )

    for i in range(len(exp_df)):
        fig.add_annotation(
            x=exp_df["Contribution"][i],
            y=i,
            text=f"{exp_df['Contribution'][i]:.2f}",
            showarrow=False,
            font=dict(size=10, color="black"),
        )

    ticktext = []
    for i, feature in enumerate(exp_df["Feature"]):
        color = "darkblue" if i % 2 == 0 else "darkorange"
        ticktext.append(f"<span style='color:{color};'>{feature}</span>")

    fig.update_layout(
        title="LIME Explanation for a Single Prediction",
        xaxis_title="Contribution",
        yaxis=dict(
            tickvals=list(range(len(exp_df))),
            ticktext=ticktext,
            showgrid=False,
            tickfont=dict(size=14),
            titlefont=dict(size=16),
        ),
        height=height,
        width=1400,
        template="plotly_white",
        xaxis=dict(
            showgrid=True,
            gridcolor="lightgray",
            tickfont=dict(size=14),
            titlefont=dict(size=16),
        ),
        font=dict(size=16),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=300, r=50, t=70, b=40),
        shapes=zebra_shapes,
    )

    pio.write_html(fig, file=f"{output_filename}.html", auto_open=False)
    fig.write_image(f"{output_filename}.png")

    fig.show()


def set_seed(seed):
    """
    Set the seed for random number generators in Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed : int
        The seed value to set.

    Returns
    -------
    None
    """

    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For reproducibility in CUDA deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
