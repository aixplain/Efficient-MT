from glob import glob
import pdb

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff  # noqa F401
import streamlit as st
from sklearn.metrics import classification_report
from src.paths import paths


from utils import (
    BASE_URL,
    category2id,
    class2id,
    feature_names,
    id2category,
    id2class,
    load_model,
)  # noqa F401


def app():
    if "models_non_reduced_data_df" not in st.session_state:
        st.session_state["models_non_reduced_data_df"] = pd.read_csv(
            paths.DATASETS_DIR / "wmt_2020_all_reduced_system_class_updated.csv"
        )
        st.session_state["models_non_reduced_data_df"]["isUpdated"] = False
        # set first 1000 samples as updated with loc
        st.session_state["models_non_reduced_data_df"].loc[:1000, "isUpdated"] = True
    if "data_df" not in st.session_state:
        st.session_state["data_df"] = pd.read_csv(
            paths.DATASETS_DIR / "wmt_2020_all_reduced_system_class_updated.csv"
        )
        st.session_state["data_df"]["isUpdated"] = False
        # set first 1000 samples as updated with loc
        st.session_state["data_df"].loc[:1000, "isUpdated"] = True

    ## get sample_features using sample_id
    def get_features(sample_id):
        sample_features = st.session_state["data_df"].loc[sample_id]
        n_rows = len(sample_id)
        return sample_features[feature_names].values.reshape(n_rows, -1), np.array(
            sample_features["severity"]
        ).reshape(n_rows, 1)

    st.markdown("## Models Analysis")

    col1, col2 = st.columns(2)
    model_list = sorted(glob("./models/model_sev_*.pkl"))
    path2name = {}
    for model in model_list:
        model_id = model.split("/")[-1].split("_")[-1].split(".")[0]
        path2name[model] = f"Model {model_id}"
    name2path = {v: k for k, v in path2name.items()}

    if "df_stats" not in st.session_state:
        stats = []
        indexes = []
        for model in model_list:
            updated_samples_df = st.session_state["models_non_reduced_data_df"][
                st.session_state["models_non_reduced_data_df"]["isUpdated"] == True
            ]
            y_true = updated_samples_df["severity"].values
            y_true = np.array([class2id[x] for x in y_true]).reshape(-1)
            sample_ids = updated_samples_df.index.values.reshape(-1)
            X = get_features(sample_ids)[0]
            model_sev = load_model(model)
            y_pred = model_sev.predict(X)

            report = classification_report(
                y_true,
                y_pred,
                output_dict=True,
                target_names=[
                    "No-Edit",
                    "Edit",
                ],
            )
            stats.append(report["weighted avg"])
            indexes.append(path2name[model])
        st.session_state["df_stats"] = pd.DataFrame.from_records(stats, index=indexes)
        st.session_state["df_stats"]["Model"] = st.session_state["df_stats"].index
    fig = px.line(
        st.session_state["df_stats"], x="Model", y=["precision", "recall", "f1-score"]
    )
    col1.markdown("#### Model Performance over Time")
    col1.write(fig)

    selected_model = col2.selectbox("Select a model", name2path.keys(), index=0)
    updated_samples_df = st.session_state["models_non_reduced_data_df"][
        st.session_state["models_non_reduced_data_df"]["isUpdated"] == True
    ]
    y_true = updated_samples_df["severity"].values
    y_true = np.array([class2id[x] for x in y_true]).reshape(-1)
    sample_ids = updated_samples_df.index.values.reshape(-1)
    X = get_features(sample_ids)[0]
    model_sev = load_model(name2path[selected_model])
    y_pred = model_sev.predict(X)
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        target_names=[
            "No-Edit",
            "Edit",
        ],
    )
    col2.markdown("#### Classification Report for Selected Model")
    col2.write(pd.DataFrame(report).transpose())


app()
