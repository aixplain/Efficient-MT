import copy

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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

    if "data_df" not in st.session_state:
        st.session_state["data_df"] = pd.read_csv(
            "./data/wmt_2020_all_reduced_system_class_updated.csv"
        )
        st.session_state["data_df"].system = st.session_state["data_df"].system.replace(
            system_map
        )
        st.session_state["data_df"]["isUpdated"] = False
        st.session_state["data_df"].loc[:1000, "isUpdated"] = True

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

    st.markdown("### Rater Ratings")
    st.table(
        asdf.groupby("rater").count()[["seg_id"]].rename(columns={"seg_id": "Count"})
    )
    systems = list(system_map.values())

    # Two columns
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

    col1, col2 = st.columns(2)
    col1.markdown("### MT Systems Average Ratings per Rater")

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=st.session_state["raters_ratings"][rater1],
            theta=systems,
            fill="toself",
            name=f"{rater1}",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=st.session_state["raters_ratings"][rater2],
            theta=systems,
            fill="toself",
            name=f"{rater2}",
        )
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 3])),
        showlegend=True,
        title="",
    )

    with col1:
        st.write(fig)

    with col2:
        st.markdown(
            """
        ### Average severity per rater
        """
        )
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
    fig.update_xaxes(title_font_family="Percentage")
    fig.update_yaxes(title_font_family="Severity Classes")
    with col2:
        st.plotly_chart(fig, use_container_width=True)


app()
