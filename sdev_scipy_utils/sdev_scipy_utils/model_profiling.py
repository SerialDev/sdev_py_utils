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
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
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
        ["learned frontier", "training observations", "new regular_observations",],
        loc="upper left",
        prop=matplotlib.font_manager.FontProperties(size=11),
    )

    n_error_test = y_pred_test[y_pred_test == -1].size

    plt.xlabel("errors novel regular: %d/40 ;" % (n_error_test))
    plt.show()

    if savefig != None:
        plt.savefig(savefig)
