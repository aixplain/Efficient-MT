import json
import os
import pdb
import pickle
import random
import sys
from typing import List

import numpy as np
import pandas as pd
from scipy import spatial
from fastapi import FastAPI
from flaml import AutoML, AutoVW
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from vowpalwabbit import pyvw
from HyperMT.feature_extractor import FeatureExtractor
from extract_features import get_scores, clean_target
from embedders.text_embeddings import HFSentenceTransformer


from data_centric.models import ActiveLearner
from utils import (
    category2id,
    category_replace,
    class2id,
    default_feature_str,
    default_label_str,
    feature_names,
    get_test_example,
    get_training_example,
    get_vw_examples,
    get_test_sample,
    get_training_sample,
    id2category,
    id2class,
    id2system,
    load_model,
    load_model_online,
    query_next_sample_interaction,
    query_next_sample,
    save_model,
    save_model_online,
    system2id,
    uncertainty_sampling,
    get_topic_proba,
)

app = FastAPI()


app.fe = FeatureExtractor()
app.text_embedder = HFSentenceTransformer(
    model_name_or_path="sentence-transformers/LaBSE"
)
app.topic_model = pickle.load(open("./models/topic_model.pkl", "rb"))


def get_features_from_text(source, target):
    source_features = app.fe([source], "en")
    target = clean_target(target)
    target_features = app.fe([target], "de")
    try:
        comet_clsss = get_scores(source, target)
    except:
        comet_clsss = [0, 0, 0, 0, 0]

    source_topic_proba, source_topic = get_topic_proba(
        source, app.text_embedder, app.topic_model
    )
    target_topic_proba, target_topic = get_topic_proba(
        target, app.text_embedder, app.topic_model
    )
    topic_distance = round(
        spatial.distance.cosine(
            source_topic_proba.reshape(-1), target_topic_proba.reshape(-1)
        ),
        5,
    )
    source_embd = app.text_embedder.get_embedding(source)
    target_embd = app.text_embedder.get_embedding(target)
    labse_distance = round(spatial.distance.cosine(source_embd, target_embd), 5)

    f = np.hstack(
        [target_features - source_features, np.array(comet_clsss).reshape(1, -1)]
    ).reshape(-1)
    s = [source_topic]
    t = [target_topic]
    ld = [labse_distance]
    td = [topic_distance]
    data = np.concatenate([f, td, ld]).reshape(1, -1)
    return data


if not os.path.exists("./data/cat_stats.csv"):
    df_stats_cat = pd.DataFrame(
        columns=["system"]
        + list(category2id.keys())
        + [f"Severity_{_}" for _ in list(class2id.keys())]
    )
    df_stats_cat.to_csv("./data/cat_stats.csv", index=False)

app.df = pd.read_csv("./data/wmt_2020_all_reduced_system_class_updated.csv")

app.df["isUpdated"] = False
# set 1000 samples as updated
app.df.loc[app.df.sample(1000).index, "isUpdated"] = True
# Replace category column with a dictionary setting
app.df["category"] = app.df["category"].replace(category_replace)


## dataframe to keep model_path, id,
df_model_path = "./models/models.csv"
if os.path.exists(df_model_path):
    app.df_model = pd.read_csv(df_model_path)
else:
    app.df_model = pd.DataFrame(columns=["model_path", "id", "type"])
    app.df_model.to_csv(df_model_path, index=False)


app.df.severity = app.df.severity.replace(
    {
        "no-error": "Neutral",
        "No-error": "Neutral",
        "word order": "Minor",
    }
).values.reshape(-1, 1)
app.df.severity = app.df.severity.replace(class2id).values.reshape(-1, 1)


## get sample_features using sample_id
def get_features(sample_id):
    sample_features = app.df.loc[sample_id]
    n_rows = len(sample_id)
    return sample_features[feature_names].values.reshape(n_rows, -1), np.array(
        sample_features["severity"]
    ).reshape(n_rows, 1)


def get_ranking_features(sample_id):
    sample_id = np.array([sample_id]).reshape(-1)

    df_tmp = app.df[app.df.seg_id.isin(app.df.loc[sample_id]["seg_id"])].sort_values(
        by="system"
    )

    def f(x):
        return pd.Series(
            dict(
                features=x[feature_names].values.reshape(1, -1),
                best_model=system2id[x.iloc[np.argmin(x["severity"].values)]["system"]],
                seg_id=x["seg_id"].values[0],
            )
        )

    df_tmp = df_tmp.groupby("seg_id").apply(f)

    return (
        np.vstack(df_tmp["features"]),
        df_tmp["best_model"].values.reshape(-1, 1),
        df_tmp["seg_id"].values.reshape(-1, 1),
    )


init_sample_ids = app.df[app.df["isUpdated"] == True].index.values


## TODO: convert online learning models into custom models
# from data_centric.custommodels import VWModel
# from data_centric.uncertainty import prediction_sampling, classifier_prediction, entropy_sampling
# custom_model = VWModel("--quiet --save_resume --oaa 10 --probabilities", get_training_example, get_test_example)
# custom_model.learn(get_ranking_features(init_sample_ids[0])[0][0], get_ranking_features(init_sample_ids[0])[1][0][0] )
# custom_model.predict(get_ranking_features(init_sample_ids)[0])
# entropy_sampling(classifier=custom_model, X=get_ranking_features(init_sample_ids)[0])

app.model_rank_online = pyvw.vw("--quiet --save_resume --oaa 10 --probabilities")


query_idx = query_next_sample(
    app.model_rank_online, get_ranking_features(init_sample_ids[0])[0], n=1
)

model_rank_online_path = f"./models/model_online_rank.pkl"
save_model_online(app.model_rank_online, model_rank_online_path)


app.model_sev_online = pyvw.vw("--quiet --save_resume --oaa 2 --probabilities")

query_idx = query_next_sample_interaction(
    app.model_sev_online, get_features(app.df.index.values)[0], n=1
)

## Learning loop
sev_features = get_features(app.df.index.values)
for x, y in zip(sev_features[0], sev_features[1]):
    app.model_sev_online.learn(get_training_sample(x, y[0]))

model__sev_online_path = f"./models/model_online_sev.pkl"
save_model_online(app.model_sev_online, model__sev_online_path)

# get latest model id from dataframe
if app.df_model.shape[0] > 0:
    model_sev = load_model(
        app.df_model[app.df_model["type"] == "sev"].iloc[-1]["model_id"]
    )
    model_cat = load_model(
        app.df_model[app.df_model["type"] == "cat"].iloc[-1]["model_id"]
    )
    model_rank = load_model(
        app.df_model[app.df_model["type"] == "rank"].iloc[-1]["model_id"]
    )

    latest_model_id = int(app.df_model.iloc[-1]["id"])
    new_model_id = latest_model_id + 1
else:
    new_model_id = 0

    model_sev_path = f"./models/model_sev_{new_model_id}.pkl"
    model_cat_path = f"./models/model_cat_{new_model_id}.pkl"
    model_rank_path = f"./models/model_rank_{new_model_id}.pkl"

    # Specify automl goal and constraints
    automl_sev_settings = {
        "time_budget": 2,  # in seconds
        "metric": "micro_f1",
        "estimator_list": ["lgbm", "xgboost"],
        "log_file_name": "my_sev.log",
    }
    # Train with labeled input data
    X_init, y_init = get_features(init_sample_ids)

    model_sev = ActiveLearner(
        estimator=AutoML(),
        embedding_pipeline="",
        X_training=X_init,
        y_training=y_init,
        query_strategy=uncertainty_sampling,
        **automl_sev_settings,
    )

    save_model(model_sev, model_sev_path)

    # Specify automl goal and constraints
    automl_cat_settings = {}

    init_sample = app.df.sample(len(id2category))
    init_sample_ids_cat = init_sample.index.values

    # Train with labeled input data
    X_init = get_features(init_sample_ids_cat)[0]
    y_1 = np.array(list(id2category.values())).reshape(-1, 1)
    y_2 = np.array(sorted(y_1, key=lambda x: random.random())).reshape(-1, 1)
    y_init = np.hstack((y_1, y_2))

    binarizer_cat = MultiLabelBinarizer(classes=list(category2id.keys()))
    y_init = binarizer_cat.fit_transform(y_init)

    model_cat = ActiveLearner(
        estimator=OneVsRestClassifier(LinearSVC(random_state=0)),
        embedding_pipeline="",
        X_training=X_init,
        y_training=y_init,
        query_strategy=uncertainty_sampling,
        **automl_cat_settings,
    )

    save_model(model_cat, model_cat_path)

    X_init_rank = np.repeat(
        get_ranking_features(init_sample_ids[0])[0],
        repeats=len(list(id2system.keys())),
        axis=0,
    )
    y_init_rank = np.array(list(id2system.keys()))

    # Specify automl goal and constraints
    automl_rank_settings = {
        "time_budget": 2,  # in seconds
        "metric": "micro_f1",
        "estimator_list": ["lgbm", "xgboost"],
        "log_file_name": "my_sev.log",
    }
    model_rank = ActiveLearner(
        estimator=AutoML(),
        embedding_pipeline="",
        X_training=X_init_rank,
        y_training=y_init_rank,
        query_strategy=uncertainty_sampling,
        **automl_rank_settings,
    )

    save_model(model_rank, model_rank_path)

    app.df_model = app.df_model.append(
        {"model_path": model_sev_path, "id": new_model_id, "type": "sev"},
        ignore_index=True,
    )
    app.df_model = app.df_model.append(
        {"model_path": model_cat_path, "id": new_model_id, "type": "cat"},
        ignore_index=True,
    )
    app.df_model = app.df_model.append(
        {"model_path": model_rank_path, "id": new_model_id, "type": "rank"},
        ignore_index=True,
    )

app.df.to_csv("./data/temp_data.csv", index=False)
