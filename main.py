import bdb
import json
import pdb
from pickle import TRUE
from typing import List

import numpy as np
import requests
import uvicorn
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score

from helpers import *
from utils import *
from llm_utils import (
    get_best_translation_propmt,
    get_mqm_erros,
    get_postedit,
    get_translation_quality,
    select_best,
)


@app.post("/get_next_sample_automl/")
async def get_next_sample_automl(
    strategy: str = "Greedy (Regression)",
    sort: str = "desc",
    topics: List[str] = topic_label_list,
):
    app.df = pd.read_csv("./data/temp_data.csv")
    # Read from  dataframe app.df and gets next sample based on sample_id
    filtered_df = app.df[app.df.isUpdated == False]
    filtered_df = filtered_df[filtered_df.source_topic.isin(topics)]

    model_sev = load_model(
        app.df_model[app.df_model["type"] == "sev"].iloc[-1]["model_path"]
    )
    model_cat = load_model(
        app.df_model[app.df_model["type"] == "cat"].iloc[-1]["model_path"]
    )
    model_rank = load_model(
        app.df_model[app.df_model["type"] == "rank"].iloc[-1]["model_path"]
    )

    model_sev.update_query_strategy(strategy_map[strategy])

    idxs = filtered_df.index.values.reshape(-1)
    X_pool, y_pool, seg_ids = get_ranking_features(idxs)
    query_idx, query_sample = model_rank.query(X_pool, n_instances=1)

    # query_idx, query_sample = model_sev.query(
    #     get_features(filtered_df.index.values)[0], n_instances=1
    # )
    sample_seg_id = seg_ids[query_idx][0][0]
    temp_df = filtered_df[filtered_df.seg_id == sample_seg_id]

    samples = []
    severity_sort_list = []
    for index, item in temp_df.groupby("system"):
        sample = {}
        sample["sample_id"] = int(item.index.values[0])
        sample["source"] = item.source.values[0]
        sample["target"] = item.target.values[0]
        sample["severity"] = int(round(float(item.severity.mean())))
        sample["category"] = list(set(item.category.values))
        sample["system"] = index

        sample["predicted_severity"] = int(
            model_sev.predict(get_features([sample["sample_id"]])[0])[0]
        )
        sample["source_major_severity_prob"] = float(
            round(
                model_sev.predict_proba(get_features([sample["sample_id"]])[0])[0][1], 2
            )
        )

        if sample["predicted_severity"] == 1:
            sample["predicted_category"] = []
        else:
            sample["predicted_category"] = binarizer_cat.inverse_transform(
                model_cat.predict(get_features([sample["sample_id"]])[0])
            )[0]
            try:
                sample["predicted_category"].remove("No-error")
            except:
                pass

        sample["best_model"] = id2system[
            model_rank.predict(get_ranking_features(sample["sample_id"])[0])[0]
        ]
        samples.append(sample)
        severity_sort_list.append(sample["predicted_severity"])

    # Sort the best to worst
    if sort == "asc":
        sorted_idx = np.argsort(severity_sort_list)
    elif sort == "desc":
        sorted_idx = np.argsort(severity_sort_list)[::-1]
    else:
        raise ValueError("sort must be either asc or desc")

    samples = np.array(samples)[sorted_idx].tolist()
    response = {"samples": samples}
    app.df.to_csv("./data/temp_data.csv", index=False)
    return response


@app.post("/update_sample_automl/")
async def update_sample_automl(
    sample_id: int,
    post_edit: str,
    best_model: str,
    cat_label: List[str],
    sev_label: str,
    skip: bool = False,
):
    app.df = pd.read_csv("./data/temp_data.csv")
    if cat_label == []:
        cat_label = ["No-error"]

    model_sev = load_model(
        app.df_model[app.df_model["type"] == "sev"].iloc[-1]["model_path"]
    )
    model_cat = load_model(
        app.df_model[app.df_model["type"] == "cat"].iloc[-1]["model_path"]
    )
    model_rank = load_model(
        app.df_model[app.df_model["type"] == "rank"].iloc[-1]["model_path"]
    )
    latest_model_id = int(app.df_model.iloc[-1]["id"])
    new_model_id = latest_model_id + 1

    df_stats_cat = pd.read_csv("./data/cat_stats.csv", index_col=None)
    # Update the statistics in the dataframe
    system_name = app.df["system"].iloc[sample_id]
    categories = binarizer_cat.transform([cat_label])
    df_cat_dict = {}
    for k, v in category2id.items():
        if k in cat_label:
            df_cat_dict[k] = 1
        else:
            df_cat_dict[k] = 0

    for k, v in class2id.items():
        if k == sev_label:
            df_cat_dict[f"Severity_{k}"] = 1
        else:
            df_cat_dict[f"Severity_{k}"] = 0

    df_cat_dict["system"] = system_name

    df_stats_cat = df_stats_cat.append(df_cat_dict, ignore_index=True)
    df_stats_cat.to_csv("./data/cat_stats.csv", index=False)

    # TODO: Calculate the features with updated data and  teach the model
    # But we already have the features calculated for these samples
    if not skip:
        # TODO extract features for post_edited text
        X, y = get_features([sample_id])

        model_cat.teach(X, categories, **automl_cat_settings)
        model_sev.teach(
            X, np.array(class2id[sev_label]).reshape(1, 1), **automl_sev_settings
        )

        X, y, _ = get_ranking_features(sample_id)
        model_rank.teach(
            X,
            np.array(system2id[best_model]).reshape(
                1,
            ),
            **automl_rank_settings,
        )

        app.df.loc[(app.df.seg_id == _[0][0]), "isUpdated"] = True

        model_sev_path = f"./models/model_sev_{new_model_id}.pkl"
        model_cat_path = f"./models/model_cat_{new_model_id}.pkl"
        model_rank_path = f"./models/model_rank_{new_model_id}.pkl"

        save_model(model_cat, model_cat_path)
        save_model(model_sev, model_sev_path)
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
    return {"status": "success"}


@app.get("/predict_sample_automl/")
async def predict_sample_automl(source: str, post_edit: str):
    app.df = pd.read_csv("./data/temp_data.csv")
    model_sev = load_model(
        app.df_model[app.df_model["type"] == "sev"].iloc[-1]["model_path"]
    )
    model_cat = load_model(
        app.df_model[app.df_model["type"] == "cat"].iloc[-1]["model_path"]
    )
    model_rank = load_model(
        app.df_model[app.df_model["type"] == "rank"].iloc[-1]["model_path"]
    )

    # Calculate the features for the post_edit text
    X = get_features_from_text(source, post_edit)

    predicted_proba = model_sev.predict_proba(X)[0]
    app.df.to_csv("./data/temp_data.csv", index=False)
    return {"major_severity_prob": round(predicted_proba[1], 2)}


@app.post("/get_data_stats/")
async def get_data_stats(rater: str, topics: List[str] = topic_label_list):
    app.df = pd.read_csv("./data/temp_data.csv")

    filtered_df = app.df
    filtered_df = filtered_df[filtered_df.source_topic.isin(topics)]
    # read unlabeled and labeled sample count
    unlabeled = (
        filtered_df[filtered_df.isUpdated == False].groupby("seg_id").count().shape[0]
    )
    labeled = (
        filtered_df[filtered_df.isUpdated == True].groupby("seg_id").count().shape[0]
    )
    app.df.to_csv("./data/temp_data.csv", index=False)
    return {"unlabeled": unlabeled, "labeled": labeled}


@app.post("/get_next_sample_online/")
async def get_next_sample_online(
    strategy: str = "Greedy (Regression)",
    sort: str = "desc",
    topics: List[str] = topic_label_list,
    use_llm: bool = True,
):
    app.df = pd.read_csv("./data/temp_data.csv")
    # Read from  dataframe app.df and gets next sample based on sample_id
    filtered_df = app.df[app.df.isUpdated == False]
    filtered_df = filtered_df[filtered_df.source_topic.isin(topics)]

    # model_sev_path = f"./models/model_online_sev.pkl"
    model_cat_path = f"./models/model_cat_0.pkl"
    # model_rank_path = f"./models/model_online_rank.pkl"

    # model_sev = load_model_online(model_sev_path)
    # model_rank = load_model_online(model_rank_path)
    model_cat = load_model(model_cat_path)

    if strategy == "Greedy (Regression)":
        query_idx = query_next_sample_interaction(
            app.model_sev_online, get_features(filtered_df.index.values)[0], n=1
        )
        sample_seg_id = filtered_df.iloc[query_idx]["seg_id"].values[0]
    else:
        ## TODO: Implement the strategy for Entopy
        # randomly select from filtered_df index
        query_idx = np.random.choice(filtered_df.index.values)
        sample_seg_id = query_idx

    temp_df = filtered_df[filtered_df.seg_id == sample_seg_id]

    samples = []
    severity_sort_list = []
    for index, item in temp_df.groupby("system"):
        sample = {}
        sample["sample_id"] = int(item.index.values[0])
        sample["source"] = item.source.values[0]
        sample["target"] = item.target.values[0]
        sample["severity"] = int(round(float(item.severity.mean())))
        sample["category"] = list(set(item.category.values))
        sample["system"] = index

        if not use_llm:
            probs = app.model_sev_online.predict(
                get_test_sample(get_features([sample["sample_id"]])[0][0])
            )

            sample["predicted_severity"] = int(int(np.argmax(probs)) + 1)
            sample["source_major_severity_prob"] = float(round(probs[1], 2))

            if sample["predicted_severity"] == 1:
                sample["predicted_category"] = []
            else:
                sample["predicted_category"] = list(
                    binarizer_cat.inverse_transform(
                        model_cat.predict(get_features([sample["sample_id"]])[0])
                    )[0]
                )
                try:
                    sample["predicted_category"].remove("No-error")
                except:
                    pass
            best_model = np.argmax(
                app.model_rank_online.predict(
                    get_test_example(get_ranking_features(sample["sample_id"])[0][0])
                )
            )
            sample["is_best_model"] = False

        else:
            sample["predicted_category"] = get_mqm_erros(
                sample["source"], sample["target"]
            )
            sample["predicted_category"] = [
                x.capitalize() for x in sample["predicted_category"]
            ]
            # predicted_sev = if (get_translation_quality(source, post_edit)/100.0) bigger than 50 then 1 else 2
            translation_quality = get_translation_quality(
                sample["source"], sample["target"]
            )
            if translation_quality > 50:
                sample["predicted_severity"] = 1
            else:
                sample["predicted_severity"] = 2
            sample["source_major_severity_prob"] = float(
                1 - round(translation_quality / 100.0, 2)
            )
            sample["is_best_model"] = False

        samples.append(sample)
        severity_sort_list.append(sample["predicted_severity"])
    if use_llm:
        # select best
        source = temp_df.source.values[0]
        mts = temp_df.target.values.tolist()
        best_model = select_best(source, mts)

    samples[best_model]["is_best_model"] = True
    # set the other s as false

    # Sort the best to worst
    if sort == "asc":
        sorted_idx = np.argsort(severity_sort_list)
    elif sort == "desc":
        sorted_idx = np.argsort(severity_sort_list)[::-1]
    else:
        raise ValueError("sort must be either asc or desc")

    samples = np.array(samples)[sorted_idx].tolist()

    response = {"samples": samples}
    app.df.to_csv("./data/temp_data.csv", index=False)
    return response


## methods to get app stats
@app.get("/get_cat_stats/")
async def get_cat_stats():
    app.df = pd.read_csv("./data/temp_data.csv")
    df_stats_cat = pd.read_csv("./data/cat_stats.csv", index_col=None)
    app.df.to_csv("./data/temp_data.csv", index=False)
    return df_stats_cat.to_dict(orient="records")


@app.get("/get_sev_model_performance_online/")
async def get_sev_model_performanc_online():
    app.df = pd.read_csv("./data/temp_data.csv")
    # Read from  dataframe app.df and gets next sample based on sample_id
    filtered_df = app.df[app.df.isUpdated == False]

    y_pred_sev = []
    y_true_sev = []
    for i, sample in filtered_df.iterrows():
        probs = app.model_sev_online.predict(get_test_sample(get_features([i])[0][0]))

        predicted_class = int(int(np.argmax(probs)) + 1)
        y_pred_sev.append(predicted_class)
        y_true_sev.append(int(sample.severity))

    return {
        "severity_model": {
            "f1_score": float(
                round(100 * f1_score(y_true_sev, y_pred_sev, average="weighted"), 3)
            )
        }
    }


@app.post("/update_sample_online/")
async def update_sample_online(
    sample_id: int,
    post_edit: str,
    best_model: str,
    cat_label: List[str],
    sev_label: str,
    skip: bool = False,
):
    app.df = pd.read_csv("./data/temp_data.csv")
    if cat_label == []:
        cat_label = ["No-error"]

    # model_sev_path = f"./models/model_online_sev.pkl"
    # model_rank_path = f"./models/model_online_rank.pkl"
    model_cat_path = f"./models/model_cat_0.pkl"
    # model_sev = load_model_online(model_sev_path)
    # model_rank = load_model_online(model_rank_path)
    model_cat = load_model(model_cat_path)

    df_stats_cat = pd.read_csv("./data/cat_stats.csv", index_col=None)
    # Update the statistics in the dataframe
    system_name = app.df["system"].iloc[sample_id]
    categories = binarizer_cat.transform([cat_label])
    df_cat_dict = {}
    for k, v in category2id.items():
        if k in cat_label:
            df_cat_dict[k] = 1
        else:
            df_cat_dict[k] = 0

    for k, v in class2id.items():
        if k == sev_label:
            df_cat_dict[f"Severity_{k}"] = 1
        else:
            df_cat_dict[f"Severity_{k}"] = 0

    df_cat_dict["system"] = system_name

    df_stats_cat = df_stats_cat.append(df_cat_dict, ignore_index=True)
    df_stats_cat.to_csv("./data/cat_stats.csv", index=False)

    # TODO: Calculate the features with updated data and  teach the model
    # But we already have the features calculated for these samples
    if not skip:
        # TODO extract features for post_edited text
        X, y = get_features([sample_id])
        model_cat.teach(X, categories, **automl_cat_settings)

        app.model_sev_online.learn(get_training_sample(X[0], class2id[sev_label]))

        X, y, _ = get_ranking_features(sample_id)
        app.model_rank_online.learn(get_training_example(X[0], system2id[best_model]))
        app.df.loc[(app.df.seg_id == _[0][0]), "isUpdated"] = True

        save_model(model_cat, model_cat_path)
        save_model_online(app.model_sev_online, model_sev_path)
        save_model_online(app.model_rank_online, model_rank_path)
    app.df.to_csv("./data/temp_data.csv", index=False)

    return True


@app.get("/predict_sample_online/")
async def predict_sample_online(source: str, post_edit: str, use_llm=True):
    app.df = pd.read_csv("./data/temp_data.csv")
    if not use_llm:
        # model_sev_path = f"./models/model_online_sev.pkl"
        # model_rank_path = f"./models/model_online_rank.pkl"
        model_cat_path = f"./models/model_cat_0.pkl"
        # model_sev = load_model_online(model_sev_path)
        # model_rank = load_model_online(model_rank_path)
        model_cat = load_model(model_cat_path)

        # Calculate the features for the post_edit text
        X = get_features_from_text(source, post_edit)

        predicted_proba = app.model_sev_online.predict(get_test_sample(X[0]))

        predicted_sev = id2class[int(np.argmax(predicted_proba)) + 1]
        if predicted_sev == 1:
            predicted_category = []
        else:
            predicted_category = list(
                binarizer_cat.inverse_transform(model_cat.predict(X))[0]
            )
            try:
                predicted_category.remove("No-error")
            except:
                pass
        major_severity_prob = (float(round(predicted_proba[1], 2)),)
    else:
        predicted_category = get_mqm_erros(source, post_edit)
        predicted_category = [x.capitalize() for x in predicted_category]
        # predicted_sev = if (get_translation_quality(source, post_edit)/100.0) bigger than 50 then 1 else 2
        translation_quality = get_translation_quality(source, post_edit)
        if translation_quality > 50:
            predicted_sev = 1
        else:
            predicted_sev = 2
        major_severity_prob = float(1 - round(translation_quality / 100.0, 2))

    app.df.to_csv("./data/temp_data.csv", index=False)
    return {
        "major_severity_prob": major_severity_prob,
        "predicted_sev": predicted_sev,
        "predicted_category": predicted_category,
    }


@app.get("/get_pseudo_labelable_samples_count_online/")
async def get_pseudo_labelable_samples_count_online(threshold: float = None):
    from sklearn.metrics import roc_curve

    # Read from  dataframe app.df and gets next sample based on sample_id
    filtered_df = app.df[app.df.isUpdated == False]

    yhat = []
    y_true = []
    for index, item in filtered_df.iterrows():
        sample_id = int(index)

        proba = app.model_sev_online.predict(
            get_test_sample(get_features([sample_id])[0][0])
        )

        yhat.append(proba)

        y_true.append(int(item.severity) - 1)
    yhat = np.array(yhat)
    y_true = np.array(y_true)

    if threshold is None:
        fpr, tpr, thresholds = roc_curve(y_true, yhat[:, 1])

        # calculate the g-mean for each threshold
        gmeans = np.sqrt(tpr * (1 - fpr))
        ix = np.argmax(gmeans)
        threshold = float(thresholds[ix])

    # get number of samples more than threshold in yhat[:,1]
    yhat_threshold = int(sum(yhat[:, 1] > threshold))
    return {
        "count": yhat_threshold,
        "percentage": round(100 * float(yhat_threshold / len(yhat)), 2),
        "threshold": threshold,
    }


if __name__ == "__main__":
    uvicorn.run(
        "helpers:app", host="0.0.0.0", port=8088, reload=False, debug=False, workers=1
    )
