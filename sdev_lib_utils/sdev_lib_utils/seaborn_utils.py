import seaborn as sns


def seaborn_setup():
    sns.set(style="whitegrid", palette="muted")


def cat_factorplot(data):
    plt.clf()
    if len(data["category"]) > 10:
        sns.factorplot(x="value", y="category", palette=["r", "c", "y"], data=data)
    else:
        sns.factorplot(x="category", y="value", palette=["r", "c", "y"], data=data)


def cat_swarmplot(data):
    # Draw a categorical scatterplot to show each observation
    plt.clf()
    if len(h_df["category"]) > 10:
        sns.swarmplot(x="value", y="category", palette=["r", "c", "y"], data=data)
    else:
        sns.swarmplot(
            x="category", y="value", palette=["r", "c", "y"], hue="category", data=data
        )


def cat_manhattan(data, facet=None):
    # Draw a categorical scatterplot to show each observation
    plt.clf()
    if len(h_df["category"]) > 10:
        g = sns.swarmplot(x="value", y="category", palette=["r", "c", "y"], data=data)
        g.set(xscale="log")

    else:
        g = sns.swarmplot(
            x="category", y="value", palette=["r", "c", "y"], hue="category", data=data,
        )
        g.set(xscale="log")


def cat_boxen(data, facet=None):
    # Draw a categorical scatterplot to show each observation
    plt.clf()
    if len(h_df["category"]) > 10:
        sns.catplot(
            x="value",
            y="category",
            kind="boxen",
            col=facet,
            palette=["r", "c", "y"],
            data=data,
        )
    else:
        sns.catplot(
            x="category",
            y="value",
            kind="boxen",
            palette=["r", "c", "y"],
            hue="category",
            data=data,
        )


def cat_box(data):
    # Draw a categorical scatterplot to show each observation
    plt.clf()
    if len(h_df["category"]) > 10:
        sns.catplot(
            x="value", y="category", kind="box", palette=["r", "c", "y"], data=data
        )
    else:
        sns.catplot(
            x="category",
            y="value",
            kind="box",
            palette=["r", "c", "y"],
            hue="category",
            data=data,
        )


def cat_strip(data, jitter=False, facet=None):
    # Draw a categorical scatterplot to show each observation
    plt.clf()
    if len(h_df["category"]) > 10:
        sns.catplot(
            x="value",
            y="category",
            col=facet,
            jitter=jitter,
            palette=["r", "c", "y"],
            data=data,
        )
    else:
        sns.catplot(
            x="category",
            y="value",
            jitter=jitter,
            palette=["r", "c", "y"],
            hue="category",
            data=data,
        )


def cat_violin(data, split=False, facet=None, inner="stick"):
    # Draw a categorical scatterplot to show each observation
    # , bw=.15, cut=0
    # inner{“box”, “quartile”, “point”, “stick”, None}, optional

    plt.clf()
    if len(h_df["category"]) > 10:
        sns.catplot(
            x="value",
            y="category",
            kind="violin",
            col=facet,
            palette=["r", "c", "y"],
            inner=inner,
            split=split,
            data=data,
        )
    else:
        sns.catplot(
            x="category",
            y="value",
            kind="violin",
            col=facet,
            palette=["r", "c", "y"],
            inner=inner,
            hue="category",
            split=split,
            data=data,
        )


def cat_dist_swarm(data, inner="violin", facet=None):
    # Draw a categorical scatterplot to show each observation
    plt.clf()
    if inner == "violin":
        if len(h_df["category"]) > 10:
            g = sns.catplot(
                x="value", y="category", col=facet, kind="violin", inner=None, data=data
            )
            sns.swarmplot(
                x="value", y="category", color="k", size=3, data=data, ax=g.ax
            )

        else:
            g = sns.catplot(
                x="category", y="value", col=facet, kind="violin", inner=None, data=data
            )
            sns.swarmplot(
                x="category", y="value", color="k", size=3, data=data, ax=g.ax
            )

    elif inner == "box":
        if len(h_df["category"]) > 10:
            g = sns.catplot(x="value", y="category", col=facet, kind="box", data=data)
            sns.swarmplot(
                x="value", y="category", color="k", size=3, data=data, ax=g.ax
            )

        else:
            g = sns.catplot(x="category", y="value", col=facet, kind="box", data=data)
            sns.swarmplot(
                x="category", y="value", color="k", size=3, data=data, ax=g.ax
            )


def cat_bar(data, facet=None):
    # Draw a categorical scatterplot to show each observation
    plt.clf()
    if len(h_df["category"]) > 10:
        sns.catplot(
            x="value",
            y="category",
            kind="bar",
            col=facet,
            palette=["r", "c", "y"],
            data=data,
        )
    else:
        sns.catplot(
            x="category",
            y="value",
            kind="bar",
            palette=["r", "c", "y"],
            hue="category",
            data=data,
        )


def cat_count(data, rev=False, facet=None):
    # Draw a categorical scatterplot to show each observation
    plt.clf()
    if rev:
        if len(h_df["category"]) > 10:
            sns.catplot(
                x="category",
                col=facet,
                kind="count",
                palette=["r", "c", "y"],
                data=data,
            )
        else:
            sns.catplot(
                x="value",
                kind="count",
                col=facet,
                palette=["r", "c", "y"],
                hue="category",
                data=data,
            )
    else:
        if len(h_df["category"]) > 10:
            sns.catplot(
                x="value", col=facet, kind="count", palette=["r", "c", "y"], data=data,
            )
        else:
            sns.catplot(
                x="category",
                col=facet,
                kind="count",
                palette=["r", "c", "y"],
                hue="category",
                data=data,
            )
