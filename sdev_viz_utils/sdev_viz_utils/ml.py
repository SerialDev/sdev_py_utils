def pd_scale_norm_df(df):
    import sklearn as sk

    dtype_cols = [i for i in df.columns if df[i].dtype == np.object]
    cols_to_norm = df.loc[:, ~df.columns.isin(dtype_cols)].columns
    train = df
    train[cols_to_norm] = sk.preprocessing.StandardScaler().fit_transform(
        train[cols_to_norm]
    )
    return train[cols_to_norm]


def histogram_intersection(a, b):
    v = np.minimum(a, b).sum().round(decimals=1)
    return v


def plot_collinearity(df):
    import sklearn as sk
    import seaborn as sns

    dtype_cols = [i for i in df.columns if df[i].dtype == np.object]
    cols_to_norm = df.loc[:, ~df.columns.isin(dtype_cols)].columns
    train = df.copy()
    train[cols_to_norm] = sk.preprocessing.StandardScaler().fit_transform(
        train[cols_to_norm]
    )

    # Identify collinearity between columns¶
    corr_df = train[cols_to_norm].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(corr_df)

    return None


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

    s = 10
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

    # plt.xlabel("errors novel regular: %d/40 ;" % (n_error_test))
    plt.show()

    if savefig != None:
        plt.savefig(savefig)
    return None


def plot_colinearity_variations(df):
    from pypair.association import binary_binary

    jaccard = lambda a, b: binary_binary(a, b, measure="jaccard")
    tanimoto = lambda a, b: binary_binary(a, b, measure="tanimoto_i")
    # This measure is typically used to judge the similarity between two clusters.
    ochiai = lambda a, b: binary_binary(a, b, measure="ochia_i")
    # Yule's Q is based off of the odds ratio or cross-product ratio, a measure of proportional reduction in error (PRE)
    yule = lambda a, b: binary_binary(a, b, measure="yule_q")
    #  A higher mutual information value implies strong association
    m_inf = lambda a, b: binary_binary(a, b, measure="mutual_information")
    # Tetrachoric correlation ranges from :math:`[-1, 1]`, where 0 indicates no agreement,
    # 1 indicates perfect agreement and -1 indicates perfect disagreement.
    tetrachoric = lambda a, b: binary_binary(a, b, measure="tetrachoric")

    train_df = pd_scale_norm_df(df)
    # Identify collinearity between columns¶
    fig = plt.figure(figsize=(20, 15))

    corr_df_1 = train_df.corr(method="pearson")
    ax = fig.add_subplot(3, 3, 1)
    ax.title.set_text("collinearity, Pearson similarity measure: ")
    sns.heatmap(corr_df_1, ax=ax, cmap="RdYlBu")

    corr_df_2 = train_df.corr(method="spearman")
    ax = fig.add_subplot(3, 3, 2)
    ax.title.set_text("Spearman similarity measure: ")
    sns.heatmap(corr_df_2, ax=ax, cmap="RdYlBu")

    corr_df_3 = train_df.corr(method=histogram_intersection)
    ax = fig.add_subplot(3, 3, 3)
    ax.title.set_text("collinearity, Histogram Intersection similarity measure: ")
    sns.heatmap(corr_df_3, ax=ax, cmap="RdYlBu")

    corr_df_4 = train_df.corr(method=jaccard)
    ax = fig.add_subplot(3, 3, 4)
    ax.title.set_text("Jaccard similarity measure: ")
    sns.heatmap(corr_df_4, ax=ax, cmap="RdYlBu")

    corr_df_5 = train_df.corr(method=tanimoto)
    ax = fig.add_subplot(3, 3, 5)
    ax.title.set_text("Tanimoto similarity measure (Jaccard Index): ")
    sns.heatmap(corr_df_5, ax=ax, cmap="RdYlBu")

    corr_df_6 = train_df.corr(method=ochiai)
    ax = fig.add_subplot(3, 3, 6)
    ax.title.set_text("Ochiai similarity measure (cosine similarity): ")
    sns.heatmap(corr_df_6, ax=ax, cmap="RdYlBu")

    corr_df_7 = train_df.corr(method=yule)
    ax = fig.add_subplot(3, 3, 7)
    ax.title.set_text("yule Q measure (cosine similarity): ")
    sns.heatmap(corr_df_7, ax=ax, cmap="RdYlBu")

    corr_df_8 = train_df.corr(method=m_inf)
    ax = fig.add_subplot(3, 3, 8)
    ax.title.set_text("Mutual Information: ")
    sns.heatmap(corr_df_8, ax=ax, cmap="RdYlBu")

    corr_df_9 = train_df.corr(method=tetrachoric)
    ax = fig.add_subplot(3, 3, 9)
    ax.title.set_text("tetrachoric correlation: ")
    sns.heatmap(corr_df_9, ax=ax, cmap="RdYlBu")
    fig.tight_layout()
    return fig
