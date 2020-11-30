import seaborn as sns


def seaborn_setup():
    sns.set(style="whitegrid", palette="muted")


def cat_factorplot(data, facet=None):
    if len(data["category"]) > 10:
        result = sns.factorplot(
            x="value", y="category", palette=["r", "c", "y"], data=data, facet=facet
        )
    else:
        result = sns.factorplot(
            x="category", y="value", palette=["r", "c", "y"], data=data, facet=facet
        )
    return result


def cat_swarmplot(data):
    # Draw a categorical scatterplot to show each observation
    if len(data["category"]) > 10:
        result = sns.swarmplot(
            x="value", y="category", palette=["r", "c", "y"], data=data
        )
    else:
        result = sns.swarmplot(
            x="category",
            y="value",
            palette=["r", "c", "y"],
            hue="category",
            data=data,
            facet=facet,
        )
    return result


def cat_manhattan(data):
    # Draw a categorical scatterplot to show each observation
    if len(data["category"]) > 10:
        result = sns.swarmplot(
            x="value", y="category", palette=["r", "c", "y"], data=data
        )
        result.set(xscale="log")

    else:
        result = sns.swarmplot(
            x="category", y="value", palette=["r", "c", "y"], hue="category", data=data,
        )
        result.set(xscale="log")
    return result


def cat_manhattan_bar(data, facet=None, height=5):
    # Draw a categorical scatterplot to show each observation
    if len(data["category"]) > 10:
        result = sns.catplot(
            x="value",
            y="category",
            kind="bar",
            col=facet,
            palette=["r", "c", "y"],
            data=data,
            height=height,
            aspect=2,  # height should be two times width
        )
        result.set(xscale="log")

    else:
        result = sns.catplot(
            x="category",
            y="value",
            kind="bar",
            palette=["r", "c", "y"],
            hue="category",
            data=data,
            height=height,
            aspect=2,  # height should be two times width
        )
        result.set(yscale="log")
    return result


def cat_boxen(data):
    # Draw a categorical scatterplot to show each observation
    if len(data["category"]) > 10:
        result = sns.catplot(
            x="value",
            y="category",
            kind="boxen",
            col=facet,
            palette=["r", "c", "y"],
            data=data,
        )
    else:
        result = sns.catplot(
            x="category",
            y="value",
            kind="boxen",
            palette=["r", "c", "y"],
            hue="category",
            data=data,
        )
    return result


def cat_box(data,):
    # Draw a categorical scatterplot to show each observation
    if len(data["category"]) > 10:
        result = sns.catplot(
            x="value", y="category", kind="box", palette=["r", "c", "y"], data=data,
        )
    else:
        result = sns.catplot(
            x="category",
            y="value",
            kind="box",
            palette=["r", "c", "y"],
            hue="category",
            data=data,
        )
    return result


def cat_strip(data, jitter=False):
    # Draw a categorical scatterplot to show each observation
    if len(data["category"]) > 10:
        result = sns.catplot(
            x="value",
            y="category",
            col=facet,
            jitter=jitter,
            palette=["r", "c", "y"],
            data=data,
        )
    else:
        result = sns.catplot(
            x="category",
            y="value",
            jitter=jitter,
            palette=["r", "c", "y"],
            hue="category",
            data=data,
        )
    return result


def cat_violin(data, split=False, facet=None, inner="stick"):
    # Draw a categorical scatterplot to show each observation
    # , bw=.15, cut=0
    # inner{“box”, “quartile”, “point”, “stick”, None}, optional

    if len(data["category"]) > 10:
        result = sns.catplot(
            x="value",
            y="category",
            kind="violin",
            col=facet,
            palette=["r", "c", "y"],
            inner=inner,
            split=split,
            data=data,
            facet=facet,
        )
    else:
        result = sns.catplot(
            x="category",
            y="value",
            kind="violin",
            col=facet,
            palette=["r", "c", "y"],
            inner=inner,
            hue="category",
            split=split,
            data=data,
            facet=facet,
        )
    return result


def cat_dist_swarm(data, inner="violin", facet=None):
    # Draw a categorical scatterplot to show each observation
    if inner == "violin":
        if len(data["category"]) > 10:
            result = sns.catplot(
                x="value", y="category", col=facet, kind="violin", inner=None, data=data
            )
            sns.swarmplot(
                x="value", y="category", color="k", size=3, data=data, ax=result.ax
            )

        else:
            result = sns.catplot(
                x="category", y="value", col=facet, kind="violin", inner=None, data=data
            )
            sns.swarmplot(
                x="category", y="value", color="k", size=3, data=data, ax=result.ax
            )

    elif inner == "box":
        if len(data["category"]) > 10:
            result = sns.catplot(
                x="value", y="category", col=facet, kind="box", data=data
            )
            sns.swarmplot(
                x="value", y="category", color="k", size=3, data=data, ax=g.ax
            )

        else:
            result = sns.catplot(
                x="category", y="value", col=facet, kind="box", data=data
            )
            sns.swarmplot(
                x="category", y="value", color="k", size=3, data=data, ax=g.ax
            )
    return result


def cat_bar(data, facet=None):
    # Draw a categorical scatterplot to show each observation
    plt.clf()
    if len(data["category"]) > 10:
        result = sns.catplot(
            x="value",
            y="category",
            kind="bar",
            col=facet,
            palette=["r", "c", "y"],
            data=data,
        )
    else:
        result = sns.catplot(
            x="category",
            y="value",
            kind="bar",
            palette=["r", "c", "y"],
            hue="category",
            data=data,
        )
    return result


def cat_count(data, rev=False, facet=None):
    # Draw a categorical scatterplot to show each observation
    if rev:
        if len(data["category"]) > 10:
            result = sns.catplot(
                x="category",
                col=facet,
                kind="count",
                palette=["r", "c", "y"],
                data=data,
            )
        else:
            result = sns.catplot(
                x="value",
                kind="count",
                col=facet,
                palette=["r", "c", "y"],
                hue="category",
                data=data,
            )
    else:
        if len(data["category"]) > 10:
            result = sns.catplot(
                x="value", col=facet, kind="count", palette=["r", "c", "y"], data=data,
            )
        else:
            result = sns.catplot(
                x="category",
                col=facet,
                kind="count",
                palette=["r", "c", "y"],
                hue="category",
                data=data,
            )
    return result


def b64_div(b64_img):
    return f'<img src="data:image/png;base64,{b64_img}">'


def convert_plt_b64(viz):
    import io
    import base64

    buffer = io.BytesIO()
    viz.savefig(buffer)
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode()


def sep_div(div):
    def create_div(data):
        return f"<div> {data} </div>"

    result = f""
    for i in div:
        result += create_div(i)
    return create_div(result)


def seaborn_multi_facet(
    df, core_facet=None, secondary_facet=None, plot_type=cat_manhattan_bar
):
    result = []
    for i in list(df[core_facet].unique()):
        result.append(
            b64_div(
                convert_plt_b64(
                    plot_type(df[df[core_facet] == i][:100], facet=core_facet)
                )
            )
        )
    return sep_div(result)
