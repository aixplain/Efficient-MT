import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils import system_map


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

    if "selected_systems" not in st.session_state:
        st.session_state["selected_systems"] = []
    if "selected_categories" not in st.session_state:
        st.session_state["selected_categories"] = []

    if "selected_dataset" not in st.session_state:
        st.session_state["selected_dataset"] = "Please Select a Dataset"

    st.header("Dataset Statistics")

    st.session_state["selected_dataset"] = st.selectbox(
        "", ["Please Select a Dataset", "Dataset 1", "Add more datasets"]
    )
    if st.session_state["selected_dataset"] != "Please Select a Dataset":
        col1, col2 = st.columns((5, 1))
        col2.button("Train a Custom MT Model")

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

        st.markdown(
            f" In total, there are {len(df_temp)} segments in the dataset and the average segment length is {round(np.mean(segment_lengths_labeled + segment_lengths_unlabeled), 2)} words. {round(len(segment_lengths_labeled)/len(df_temp), 2)*100} % of the segments are rated by 3 different raters"
        )
        df_describe = df_temp.describe().apply(lambda x: round(x, 0))
        df_describe["Segment Length"] = df_describe["Segment Length"].astype(int)
        st.table(df_describe.loc[["mean", "max", "min", "std"]])

        # fig = px.box(df_temp, x="Is Rated", y="Segment Length", color="Is Rated")

        fig = px.histogram(df_temp, x="Segment Length", marginal="box")

        fig.show()
        fig.update_yaxes(title="Segment Length")
        fig.update_xaxes(title="Is Rated")

        st.markdown(
            """
        ### Segment length distribution in number of words
        """
        )

        st.plotly_chart(fig, use_container_width=True)

        # Two columns

        st.markdown("### Topic Distribution of Dataset")

        filter = st.selectbox("Filter By", ["All", "Rated", "Not Rated"])
        if filter == "All":
            df_topic = pd.DataFrame(
                st.session_state["data_df"].groupby("source_topic")["system"].count()
            )
        elif filter == "Rated":
            df_topic = pd.DataFrame(
                st.session_state["data_df"][
                    st.session_state["data_df"].isUpdated == True
                ]
                .groupby("source_topic")["system"]
                .count()
            )
        else:
            df_topic = pd.DataFrame(
                st.session_state["data_df"][
                    st.session_state["data_df"].isUpdated == False
                ]
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
        st.write(fig)


app()
