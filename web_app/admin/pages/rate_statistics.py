import ast
import collections
import copy
import pdb

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from utils import (
    BASE_URL,
    category2id,
    class2id,
    id2category,
    id2class,
    system2id,
    system_map,
)  # noqa F401


def app():
    if "data_df" not in st.session_state:
        st.session_state["data_df"] = pd.read_csv(
            "./data/wmt_2020_all_reduced_system_class_updated.csv"
        )
        st.session_state["data_df"].system = st.session_state["data_df"].system.replace(
            system_map
        )
        st.session_state["data_df"]["isUpdated"] = False
        st.session_state["data_df"].loc[:1000, "isUpdated"] = True
        st.session_state["data_df"].system = st.session_state["data_df"].system.replace(
            system_map
        )
        st.session_state["data_df"].category = st.session_state[
            "data_df"
        ].category.apply(lambda x: ast.literal_eval(x))

    if "non_reduced_data_df" not in st.session_state:
        np.random.seed(42)
        st.session_state["non_reduced_data_df"] = pd.read_csv(
            "./data/wmt_2020_all_reduced_system_class_updated.csv"
        )
        st.session_state["non_reduced_data_df"].system = st.session_state[
            "non_reduced_data_df"
        ].system.replace(system_map)
        st.session_state["non_reduced_data_df"]["isUpdated"] = False
        st.session_state["non_reduced_data_df"].loc[:1000, "isUpdated"] = True

        st.session_state["non_reduced_data_df"]["rater"] = np.nan
        # asdf["rater"] = list(
        #     np.random.choice(["Rater1", "Rater2", "Rater3"], len(asdf))
        # )
        st.session_state["non_reduced_data_df"].loc[
            st.session_state["non_reduced_data_df"]["isUpdated"] == True, "rater"
        ] = list(
            np.random.choice(
                ["Rater1", "Rater2", "Rater3"],
                len(
                    st.session_state["non_reduced_data_df"][
                        st.session_state["non_reduced_data_df"]["isUpdated"] == True
                    ]
                ),
            )
        )

    if "selected_systems" not in st.session_state:
        st.session_state["selected_systems"] = []
    if "selected_categories" not in st.session_state:
        st.session_state["selected_categories"] = []

    if "threshold" not in st.session_state:
        st.session_state["threshold"] = 0.7

    if "labelable_samples" not in st.session_state:
        st.session_state["labelable_samples"] = requests.get(
            BASE_URL + "get_pseudo_labelable_samples_count_online",
            params={"threshold": st.session_state["threshold"]},
        ).json()

    st.markdown("## Dataset Statistics")

    # slider for threshold
    st.sidebar.markdown("## Threshold for model confidence")
    st.session_state["threshold"] = st.sidebar.slider("Threshold", 0.5, 1.0, 0.7, 0.01)
    st.session_state["labelable_samples"] = requests.get(
        BASE_URL + "get_pseudo_labelable_samples_count_online",
        params={"threshold": st.session_state["threshold"]},
    ).json()

    st.sidebar.markdown(
        f"**{st.session_state['labelable_samples']['percentage']}%** of remaining samples can be labeled automatically."
    )
    st.sidebar.markdown("Would you like to label them?")
    if st.sidebar.button("Yes"):
        st.sidebar.success(
            f"You have labeled **{st.session_state['labelable_samples']['percentage']}%** of samples."
        )

    col1, col2 = st.columns(2)
    segment_lengths_labeled = [
        len(x.split(" "))
        for x in st.session_state["data_df"][
            st.session_state["data_df"]["isUpdated"] == True
        ].source.unique()
    ]
    segment_lengths_unlabeled = [
        len(x.split(" "))
        for x in st.session_state["data_df"][
            st.session_state["data_df"]["isUpdated"] == False
        ].source.unique()
    ]

    df_temp = pd.DataFrame(
        segment_lengths_labeled + segment_lengths_unlabeled,
        columns=["Segment Length"],
    )
    df_temp["Is Rated"] = ["Rated"] * len(segment_lengths_labeled) + [
        "Not Rated"
    ] * len(segment_lengths_unlabeled)

    df_describe = df_temp.describe().apply(lambda x: round(x, 0))
    df_describe["Segment Length"] = df_describe["Segment Length"].astype(int)
    # st.table(df_describe.loc[["mean", "max", "min", "std"]])

    # fig = px.box(df_temp, x="Is Rated", y="Segment Length", color="Is Rated")

    fig = px.histogram(df_temp, x="Segment Length", marginal="box")

    fig.show()
    fig.update_yaxes(title="Segment Length")
    fig.update_xaxes(title="Is Rated")

    col1.markdown(
        """
    ### Segment length distribution in number of words
    """
    )

    col1.plotly_chart(fig, use_container_width=True)

    # Two columns

    col2.markdown("### Topic Distribution of Dataset")

    filter = col2.selectbox("Filter By", ["All", "Rated", "Not Rated"])
    if filter == "All":
        df_topic = pd.DataFrame(
            st.session_state["data_df"].groupby("source_topic")["system"].count()
        )
    elif filter == "Rated":
        df_topic = pd.DataFrame(
            st.session_state["data_df"][st.session_state["data_df"].isUpdated == True]
            .groupby("source_topic")["system"]
            .count()
        )
    else:
        df_topic = pd.DataFrame(
            st.session_state["data_df"][st.session_state["data_df"].isUpdated == False]
            .groupby("source_topic")["system"]
            .count()
        )

    df_topic.reset_index(inplace=True)
    fig = px.pie(
        df_topic,
        values="system",
        names="source_topic",
        title="Topics Distribution",
    )
    col2.write(fig)

    categories = list(category2id.keys())
    reduced_categories = [cat.split("/")[0] for cat in categories]
    systems = list(system2id.keys())

    # # Two columns
    col1, col2 = st.columns(2)

    # %% [markdown]
    with col1:
        st.markdown(
            """
        ### Percentage of samples based on severity
        """
        )
        sort_by = st.selectbox("Sort by", ["No-Edit", "Edit"], index=0)
    severity_scores = {
        sys: collections.defaultdict(int)
        for sys in st.session_state["data_df"].system.unique()
    }

    for idx, row in st.session_state["data_df"].iterrows():
        severity_scores[row.system][row.severity] += 1

    for system in severity_scores.keys():
        total = sum(severity_scores[system].values())
        for k in severity_scores[system].keys():
            severity_scores[system][k] /= total

    severe_df = (
        pd.DataFrame(severity_scores)
        .T.reset_index()
        .sort_values(by=[sort_by], ascending=False)
    )
    severe_df = severe_df.rename(columns={"index": "system"})
    # Evaluation Report (3)
    fig = px.bar(
        severe_df,
        x="system",
        y=["No-Edit", "Edit"],
        # title="Model Severity over Segments",
    )
    # fig.show()
    with col1:
        st.plotly_chart(fig, use_container_width=True)

    error_category_list = [
        "Accuracy",
        "Fluency",
        "Locale convention",
        "No-error",
        "Terminology",
        "Other",
    ]

    # %% [markdown]
    with col2:
        st.markdown(
            """
        ### Highest Error Category per Model
        """
        )
        sort_by = st.selectbox("Sort by", error_category_list, index=0)
    error_codes = {
        "Accuracy",
        "Fluency",
        "Locale convention",
        "No-error",
        "Terminology",
        "Other",
    }
    error_codes = {
        "Accuracy",
        "Fluency",
        "Locale convention",
        "No-error",
        "Terminology",
        "Other",
    }
    bad_models = {
        sys: {error: 0 for error in error_codes}
        for sys in st.session_state["data_df"].system.unique()
    }

    for idx, row in st.session_state["data_df"].iterrows():
        for error in row.category:
            if "Accuracy" in error:
                bad_models[row.system]["Accuracy"] += 1
            elif "Fluency" in error:
                bad_models[row.system]["Fluency"] += 1
            elif "Locale convention" in error:
                bad_models[row.system]["Locale convention"] += 1
            elif "Terminology" in error:
                bad_models[row.system]["Terminology"] += 1
            elif "No-error" in error or "Minor" in error:
                bad_models[row.system]["No-error"] += 1
            else:
                bad_models[row.system]["Other"] += 1

    bad_models = (
        pd.DataFrame(bad_models)
        .T.reset_index()
        .sort_values(by=[sort_by], ascending=False)
    )
    bad_models = bad_models.rename(columns={"index": "system"})
    subset = bad_models.select_dtypes("number")
    bad_models[subset.columns] = subset.div(subset.sum(axis=1), axis=0)
    bad_models["system"] = st.session_state["data_df"].system.unique()
    # Evaluation Report (4)
    fig = px.bar(
        bad_models.sort_values(by=["No-error"]),
        x="system",
        y=list(bad_models.columns[1:]),
        # title="Error Types per Model",
    )
    # fig.show()
    with col2:
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        ### Estimated MT Performances on the Remaining Data
        """
    )

    effort_map = {"Edit": 0.5, "No-Edit": 0.9}
    st.session_state["data_df"]["saved_effort"] = st.session_state[
        "data_df"
    ].severity.apply(lambda x: effort_map[x])

    avg_effort = {
        sys: np.mean(
            st.session_state["data_df"][
                st.session_state["data_df"].system == sys
            ].saved_effort
        )
        for sys in st.session_state["data_df"].system.unique()
    }

    avg_severity_plot = (
        pd.DataFrame([avg_effort])
        .T.reset_index()
        .rename(columns={"index": "System", 0: "Percentage of No-edit Samples"})
    )  # .sort_values(by=['No-error'])

    # Evaluation Report (6)
    fig = px.bar(
        avg_severity_plot.sort_values(by=["Percentage of No-edit Samples"]),
        x="System",
        y="Percentage of No-edit Samples",
        # title="Avg Effort Saved per Provider",
    )
    st.plotly_chart(fig, use_container_width=True)

    def normalize_df(df):
        return (df.T / df.sum(1)).T * 100

    def f(x):
        return pd.Series(
            dict(
                system=x.system.values[0],
                seg_id=x.seg_id.values[0],
                rater=x.rater.values[0],
                source=x.source.values[0],
                target=x.target.values[0],
                severity=x.severity.values[0],
            )
        )

    asdf = st.session_state["non_reduced_data_df"][
        st.session_state["non_reduced_data_df"]["isUpdated"] == True
    ]

    if "raters_ratings" not in st.session_state:
        st.session_state["raters_ratings"] = {
            rater: np.random.choice([1, 3], 10) for rater in asdf.rater.unique()
        }
        st.session_state["raters_ratings"]["Others"] = np.random.choice([2, 3], 10)

    asd = asdf.groupby(["seg_id", "rater"]).apply(f)
    ratings = {
        rater: {rating: 0 for rating in {"No-Edit", "Edit"}}
        for rater in asd.rater.unique()
    }

    for idx, row in asd.iterrows():
        rater = row.rater
        ratings[rater][row.severity] += 1
    rating_df = pd.DataFrame(ratings).T
    rating_df = normalize_df(rating_df).round(1)

    st.markdown("# Rater Analysis")

    st.markdown(
        """
        ### Raters Statistics
        """
    )

    df_rater_count = (
        asdf.groupby("rater")
        .count()[["seg_id"]]
        .rename(columns={"seg_id": "# of Samples Rated"})
    )
    df_rater_count["Rater ID"] = df_rater_count.index
    fig = px.bar(df_rater_count, x="Rater ID", y="# of Samples Rated")
    st.plotly_chart(fig, use_container_width=True)

    # Two columns

    st.markdown(
        """
        ### Average severity per rater
        """
    )
    col1, col2, col3, col4, col5, col6 = st.columns((2, 2, 1, 1, 2, 10))
    col1.markdown("##")
    col1.markdown("#### Compare ")
    unique_list = sorted(asd.rater.unique())
    rater1 = col2.selectbox("", unique_list, key="rater1")
    col4.markdown("##")
    col4.markdown("#### with")
    unique_list2 = copy.copy(unique_list)
    unique_list2.remove(rater1)
    rater2 = col5.selectbox("", ["Others"] + unique_list2, key="rater2")

    if rater2 != "Others":
        sample = (
            rating_df.loc[[f"{rater1}", f"{rater2}"]]
            .reset_index()
            .rename(columns={"index": "rater"})
        )
    else:
        sample = (
            pd.concat(
                [
                    rating_df.loc[[f"{rater1}"]],
                    pd.DataFrame(
                        round(
                            100
                            * rating_df.loc[unique_list].sum()
                            / rating_df.loc[unique_list].sum().sum(),
                            1,
                        ),
                        columns=["Others"],
                    ).T,
                ],
                axis=0,
            )
            .reset_index()
            .rename(columns={"index": "rater"})
        )

    fig = go.Figure(
        data=[
            go.Bar(
                name=sample.iloc[0].rater,
                x=list(sample.iloc[0].keys()[1:]),
                y=list(sample.iloc[0].values[1:]),
                text=list(sample.iloc[0].values[1:]),
                textposition="outside",
            ),
            go.Bar(
                name=sample.iloc[1].rater,
                x=list(sample.iloc[1].keys()[1:]),
                y=list(sample.iloc[1].values[1:]),
                text=list(sample.iloc[1].values[1:]),
                textposition="outside",
            ),
        ]
    )
    fig.update_layout(yaxis_range=[0, 100])
    fig.update_xaxes(title="Percentage")
    fig.update_yaxes(title="Severity Classes")
    st.plotly_chart(fig, use_container_width=True)


app()
