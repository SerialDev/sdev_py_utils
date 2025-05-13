def yield_batches_as_series(iterable_or_series, batch_size=20):
    """
    A generator that yields batches from an iterable or a pandas Series. When the input is a pandas Series,
    such as from value_counts(), it yields each batch as a new pandas Series, preserving the index (values) and counts.

    Parameters:
        iterable_or_series: An iterable or a pandas Series from which elements will be generated in batches.
        batch_size (int): The size of each batch to yield.
    """
    batch_indices = []  # Initialize a list to collect indices (for pandas Series)
    batch_values = []  # Initialize a list to collect values

    # Handle when input is a pandas Series
    if isinstance(iterable_or_series, pd.Series):
        for index, value in iterable_or_series.items():
            batch_indices.append(index)
            batch_values.append(value)
            if len(batch_values) == batch_size:
                yield pd.Series(
                    batch_values, index=batch_indices
                )  # Yield as a pandas Series
                batch_indices, batch_values = [], []  # Reset for next batch

    else:  # Handle generic iterables
        for element in iterable_or_series:
            batch_values.append(element)
            if len(batch_values) == batch_size:
                yield pd.Series(
                    batch_values
                )  # Yield as a pandas Series without a specific index
                batch_values = []  # Reset for next batch

    # After looping, yield any remaining items as a final batch
    if batch_values:
        if isinstance(iterable_or_series, pd.Series):
            yield pd.Series(batch_values, index=batch_indices)
        else:
            yield pd.Series(batch_values)


def pretty_print_array_colored(array):
    """
    Pretty prints an array of (name, count) tuples in color.

    Parameters:
        array: An array of tuples to be pretty printed, where each tuple is in the form (name, count).
    """
    # ANSI escape codes for colors
    RED = "\033[31m"  # Red text
    GREEN = "\033[32m"  # Green text
    RESET = "\033[0m"  # Reset to default terminal color

    # Iterate over each tuple in the array and print with colors
    for name, count in array:
        print(f"{GREEN}{name}{RESET}: {RED}{count}{RESET}")


def yield_axis(df, axis_type="column", axis_name=1):
    if axis_type == "column":
        yield df[axis_name]  # Yield an entire column
    else:
        # For row, ensure axis_name is an integer index or a label if the index is named
        yield df.loc[axis_name]  # Yield an entire row


import pandas as pd
import plotly.express as px


def create_sunburst_df(df, columns, status_filter):
    # Ensure the required columns are in the DataFrame
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} is not in the DataFrame")

    # Create the sunburst_df DataFrame with the desired columns
    sunburst_df = pd.DataFrame(df[columns].value_counts()).reset_index()
    sunburst_df.columns = columns + ["count"]

    return sunburst_df


def create_treemap(
    df, columns, path, status_filter=["Gold", "Silver"], output_file=True
):
    sunburst_df = create_sunburst_df(df, columns, status_filter)

    # Filter the DataFrame based on status_filter
    if "tiers" in columns:
        sunburst_df = sunburst_df[sunburst_df["tiers"].isin(status_filter)]

    # Create the treemap visualization
    fig = px.treemap(
        sunburst_df,
        path=path,
        values="count",
        color=path[-1],
        title="Treemap Visualization",
    )

    # Set the size of the figure
    fig.update_layout(width=1500, height=1000)

    if isinstance(output_file, str):
        # Save the figure to a file
        fig.write_image(output_file)
    else:
        # Show the figure
        fig.show()


def create_sunburst_df(df, columns):
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} is not in the DataFrame")

    # Create a DataFrame with value counts
    sunburst_df = pd.DataFrame(df[columns].value_counts()).reset_index()
    sunburst_df.columns = columns + ["count"]

    # Calculate total count
    total_count = sunburst_df["count"].sum()
    sunburst_df["percent_of_total"] = (sunburst_df["count"] / total_count) * 100
    # Initialize the percent_of_partition column
    sunburst_df["percent_of_partition"] = 0.0

    # Calculate the hierarchical partition percentages
    for i, col in enumerate(columns):
        if i == 0:
            # For the first column, percent_of_partition is the same as percent_of_total
            sunburst_df["percent_of_partition"] = (
                sunburst_df["count"] / total_count
            ) * 100
        else:
            # For subsequent columns, calculate the percentage relative to the parent segment
            parent_cols = columns[:i]
            parent_counts = sunburst_df.groupby(parent_cols)["count"].transform("sum")
            sunburst_df["percent_of_partition"] = (
                sunburst_df["count"] / parent_counts
            ) * 100

    return sunburst_df


def pretty_print_partitions(df, path):
    results = []

    def recursive_collect(df, path, level=0, parent_key=""):
        if path:
            group_col = path[0]
            for key, group in df.groupby(group_col):
                indent = "  " * level
                total_count = group["count"].sum()
                percent_total = group["percent_of_total"].iloc[0]
                parent_info = f"{indent}{key} (Count: {total_count}, % of Total: {percent_total:.2f}%)"
                results.append([parent_info, level, parent_key])
                recursive_collect(
                    group,
                    path[1:],
                    level + 1,
                    f"{parent_key} -> {key}" if parent_key else key,
                )
        else:
            for _, row in df.iterrows():
                indent = "  " * level
                child_info = f"{indent}{row['count']} (Count: {row['count']}, % of Partition: {row['percent_of_partition']:.2f}%)"
                results.append([child_info, level, parent_key])

    recursive_collect(df, path)

    pretty_df = pd.DataFrame(results, columns=["Description", "Level", "Parent"])
    return pretty_df


import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def create_sankey(
    sunburst_df, path, top_n=10, color_col=None, output_file=True, verbose=False
):
    if verbose:
        print(f"Filtered DataFrame:\n{sunburst_df}")

    labels_dict = {}
    sources = []
    targets = []
    values = []
    colors = []

    def generate_color_map(unique_values):
        color_scale = px.colors.qualitative.Plotly
        color_map = {
            value: color_scale[i % len(color_scale)]
            for i, value in enumerate(unique_values)
        }
        return color_map

    def process_level(df, path, parent_index=None, parent_count=None):
        if not path:
            return
        group_col = path[0]
        next_path = path[1:]

        if group_col not in df.columns:
            if verbose:
                print(f"Column {group_col} not in DataFrame columns")
            return

        if verbose:
            print(f"Processing level with column: {group_col}")
            print(f"DataFrame before grouping:\n{df}")

        grouped = (
            df.groupby(group_col)
            .agg({"count": "sum", "percent_of_partition": "sum"})
            .reset_index()
        )

        if verbose:
            print(f"Grouped data:\n{grouped}")

        top_grouped = grouped.nlargest(top_n, "count")
        if verbose:
            print(f"Top grouped data:\n{top_grouped}")

        total = parent_count if parent_count else top_grouped["count"].sum()

        for _, row in top_grouped.iterrows():
            percent = (row["count"] / total) * 100
            label = f"{row[group_col]} ({row['count']}, {percent:.2f}%)"
            if label not in labels_dict:
                labels_dict[label] = len(labels_dict)
            current_index = labels_dict[label]
            if parent_index is not None:
                sources.append(parent_index)
                targets.append(current_index)
                values.append(row["count"])

                if color_col and color_col in row:
                    link_color = color_map.get(
                        row[color_col], "rgba(0,0,0,0.2)"
                    )  # Default color if not found
                else:
                    link_color = "rgba(0,0,0,0.2)"  # Default transparent black

                colors.append(link_color)

            sub_df = df[df[group_col] == row[group_col]]
            if verbose:
                print(f"Processing {label}: {sub_df.shape[0]} rows")
            process_level(sub_df, next_path, current_index, row["count"])

    if color_col and color_col in sunburst_df.columns:
        unique_values = sunburst_df[color_col].unique()
        color_map = generate_color_map(unique_values)
    else:
        color_map = {}

    process_level(sunburst_df, path)

    labels = [
        label for label, _ in sorted(labels_dict.items(), key=lambda item: item[1])
    ]
    if verbose:
        print(f"Labels: {labels}")
        print(f"Sources: {sources}")
        print(f"Targets: {targets}")
        print(f"Values: {values}")
        print(f"Colors: {colors}")

    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=10, thickness=20, line=dict(color="black", width=0.5), label=labels
            ),
            link=dict(source=sources, target=targets, value=values, color=colors),
        )
    )

    fig.update_layout(title_text="Sankey Diagram", font_size=10)
    fig.update_layout(width=1500, height=1000)

    if isinstance(output_file, str):
        fig.write_image(output_file)
    else:
        fig.show()

    return sunburst_df




class RLFeatureEngineerClassifier:
    # NOTE: TODO: AGent this

    import numpy as np
    import pandas as pd
    import random
    from collections import defaultdict
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import load_digits, make_classification
    from sklearn.metrics import make_scorer, f1_score

    def __init__(self, df, target_col, estimator=None, episodes=30, alpha=0.1, gamma=0.9,
                 epsilon=0.5, epsilon_decay=0.95, random_seed=42):
        self.df = df.copy()
        self.target_col = target_col
        self.X_orig = df.drop(columns=[target_col])
        self.y = df[target_col].values
        self.episodes = episodes
        self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.epsilon_decay = epsilon, epsilon_decay
        self.random_seed = random_seed
        self.transformations = [
            lambda x: np.log1p(np.abs(x)),
            lambda x: np.sqrt(np.abs(x)),
            lambda x: np.square(x),
            lambda x: np.tanh(x),
            lambda x: 1.0/(1.0+np.abs(x)),
            lambda x: x,
            lambda x: x**3,
            lambda x: np.sin(x),
            lambda x: np.cos(x),
            lambda x: np.exp(np.clip(x,-5,5))
        ]
        self.q_table = defaultdict(lambda: np.zeros(len(self.transformations)))
        self.estimator = estimator if estimator else RandomForestClassifier(n_estimators=100, random_state=random_seed)
        self.scorer = make_scorer(f1_score, average='weighted')
        self.baseline = self.evaluate_reward(self.X_orig)
        
    def select_action(self, state):
        return np.random.choice(len(self.transformations)) if random.random() < self.epsilon else np.argmax(self.q_table[state])

    def evaluate_reward(self, X):
        model = clone(self.estimator)
        scores = cross_val_score(model, X, self.y, cv=5, scoring=self.scorer)
        return scores.mean()

    def train(self):
        best_reward = self.baseline
        best_transforms = {}
        print(f"\nBaseline Weighted-F1: {self.baseline:.5f}")
        for ep in range(1, self.episodes+1):
            X_ep, state_action_pairs = self.X_orig.copy(), []
            current_transforms = {}
            for col in X_ep.columns:
                state = (col,)
                action_idx = self.select_action(state)
                X_ep[col] = self.transformations[action_idx](X_ep[col])
                current_transforms[col]=action_idx
                state_action_pairs.append((state,action_idx))
            reward = self.evaluate_reward(X_ep)
            delta = reward-self.baseline
            sign = "↑" if delta>=0 else "↓"
            print(f"Ep{ep}/{self.episodes}: F1:{reward:.5f} Δ:{sign}{delta:.5f} ε:{self.epsilon:.3f}")
            if reward>best_reward:
                best_reward, best_transforms = reward, current_transforms.copy()
            for state,action_idx in state_action_pairs:
                q_old=self.q_table[state][action_idx]
                self.q_table[state][action_idx]=q_old+self.alpha*(reward+self.gamma*np.max(self.q_table[state])-q_old)
            self.epsilon=max(self.epsilon*self.epsilon_decay,0.01)
        self.best_transforms=best_transforms                
        print(f"Best final reward: {best_reward:.5f} Δ:{best_reward-self.baseline:.5f}")

    def transform(self):
        X_final=self.X_orig.copy()
        for col,action_idx in self.best_transforms.items():
            X_final[col]=self.transformations[action_idx](X_final[col])
        return X_final
    def run_eval(X,y,name,episodes=30):
    df=pd.DataFrame(X)
    df['target']=y
    engineer=RLFeatureEngineerClassifier(df,'target',episodes=episodes)
    engineer.train()
    X_eng=engineer.transform()
    clf=RandomForestClassifier(n_estimators=100,random_state=42)
    scorer=make_scorer(f1_score,average='weighted')
    pre=cross_val_score(clf,X,y,cv=5,scoring=scorer).mean()
    post=cross_val_score(clf,X_eng,y,cv=5,scoring=scorer).mean()
    print(f"\n{name} results:")
    print(f"Original features F1:  {pre:.5f}")
    print(f"Engineered features F1:{post:.5f}")
    print(f"Net improvement: {post-pre:+.5f}\n{'-'*60}")
