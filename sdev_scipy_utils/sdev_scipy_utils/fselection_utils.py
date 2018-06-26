from sklearn.ensemble import RandomForestRegressor

# from sklearn.model_selection import ShuffleSplit  # TODO: New version not an iterator
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score
from collections import defaultdict
from sklearn.linear_model import LinearRegression, Lasso, Ridge, RandomizedLasso
from sklearn.feature_selection import RFE
import numpy as np


def feature_select(X, Y, names=None, method="all"):
    """
    * Function: Select features using random forest regressor
    * Usage: feature_select(X, Y)  . . .
    * -------------------------------
    * This function prints
    *  sorted features by their score based on ['linear', 'L1', 'L2',
    *  'impurity', 'acc_loss', 'stability', 'recursive_l', 'recursive_lsvc']
    """

    def pretty_print_linear(coefs, names=None, sort=False):
        # a Helper method for pretty printing linear models
        if names == None:
            names = ["X%s" % x for x in range(len(coefs))]
        lst = zip(coefs, names)
        if sort:
            lst = sorted(lst, key=lambda x: -np.abs(x[0]))
        return " + ".join("%s * %s" % (round(coef, 3), name) for coef, name in lst)

    if method == "linear":
        lr = LinearRegression()
        lr.fit(X, Y)
        print("Linear Model:", pretty_print_linear(lr.coef_, names, sort=True))
    elif method == "L1":  # Lasso
        lasso = Lasso(alpha=.3)
        lasso.fit(X, Y)
        print("Lasso model:", pretty_print_linear(lasso.coef_, names, sort=True))
    elif method == "L2":  # Ridge
        ridge = Ridge(alpha=10)
        scores = defaultdict(list)
        for train_idx, test_idx in ShuffleSplit(len(X), 100, 0.3):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            r = ridge.fit(X_train, Y_train)
            acc = r2_score(Y_test, r.predict(X_test))
            for i in range(X.shape[1]):
                X_t = X_test.copy()
                np.random.shuffle(X_t[:, i])
                shuff_acc = r2_score(Y_test, r.predict(X_t))
                scores[names[i]].append((acc - shuff_acc) / acc)
        print("Ridge model:", pretty_print_linear(ridge.coef_, names, sort=True))
        print("Features sorted by their 'L2' score:")
        print(
            sorted(
                [(round(np.mean(score), 4), feat) for feat, score in scores.items()],
                reverse=True,
            )
        )
    elif method == "impurity":
        if names == None:
            names = np.array(X.columns)
        rf = RandomForestRegressor()
        rf.fit(X, Y)
        print("Features sorted by their 'impurity' score:")
        print(
            sorted(
                zip(map(lambda x: round(x, 4), rf.feature_importances_), names),
                reverse=True,
            )
        )
    elif method == "acc_loss":
        rf = RandomForestRegressor()
        scores = defaultdict(list)
        # Crossvalidate the scores on a number of random splits of the data
        for train_idx, test_idx in ShuffleSplit(len(X), 100, 0.3):
            Y = Y.ravel()
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            r = rf.fit(X_train, Y_train)
            acc = r2_score(Y_test, rf.predict(X_test))
            for i in range(X.shape[1]):
                X_t = X_test.copy()
                np.random.shuffle(X_t[:, i])
                shuff_acc = r2_score(Y_test, rf.predict(X_t))
                scores[names[i]].append((acc - shuff_acc) / acc)
        print("Features sorted by their 'acc_loss' score:")
        print(
            sorted(
                [(round(np.mean(score), 4), feat) for feat, score in scores.items()],
                reverse=True,
            )
        )
    elif method == "stability":
        rlasso = RandomizedLasso(alpha=0.025)
        rlasso.fit(X, Y)
        print("Features sorted by their score:")
        print(
            sorted(zip(map(lambda x: round(x, 4), rlasso.scores_), names), reverse=True)
        )
    elif method == "recursive_l":
        # Recursive with linear
        lr = LinearRegression()
        # rank all features[elimination until the last one)
        rfe = RFE(lr, n_features_to_select=1)
        rfe.fit(X, Y)
        print("Features sorted by their 'recursive'[linear] rank:")
        print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))
    elif method == "recursive_lsvc":
        # Recursive with linear SVC
        lsvc = LinearSVC()
        # rank all features[elimination until the last one)
        rfe = RFE(lsvc, n_features_to_select=1)
        rfe.fit(X, Y.astype("int"))
        print("Features sorted by their 'recursive'[lSVC] rank:")
        print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))
    elif method == "all":
        try:
            feature_select(X, Y, names, method="linear")
        except Exception as e:
            print("linear failed")
        try:
            feature_select(X, Y, names, method="L1")
        except Exception as e:
            print("L1 failed")
        try:
            feature_select(X, Y, names, method="L2")
        except Exception as e:
            print("L2 failed")
        try:
            feature_select(X, Y, names, method="impurity")
        except Exception as e:
            print("impurity failed")
        try:
            feature_select(X, Y, names, method="acc_loss")
        except Exception as e:
            print("acc_loss failed")
        try:
            feature_select(X, Y, names, method="stability")
        except Exception as e:
            print("stability failed")
        try:
            feature_select(X, Y, names, method="recursive_l")
        except Exception as e:
            print("recursive_l failed")
        try:
            feature_select(X, Y, names, method="recursive_lsvc")
        except Exception as e:
            print("recursive_lsvc failed")


def root_mean_sqr(x, flag="norm"):
    if flag == "norm":
        return np.sqrt(x.dot(x) / x.size)
    if flag == "complex":
        return np.sqrt(np.vdot(x, x) / x.size)
