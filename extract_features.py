import json
import os
import pdb
import pickle
import time
from time import sleep

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from HyperMT.feature_extractor import FeatureExtractor

BASE_URL = "https://benchmarking-scoring.aixplain.io/"

headers = {"Content-Type": "application/json"}

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def clean_target(text):
    return text.replace("<v>", "").replace("</v>", "")


def get_scores(source, target):
    input_payload = {
        "input": f'["{source}"]',
        "output": f'["{target}"]',
        "refs": "",
        "mode": "string",
        "data_uri": "string",
        "metric_name": "string",
    }
    tmp_response = requests.request(
        "POST",
        BASE_URL + "startasynctranslationscores",
        headers=headers,
        data=json.dumps(input_payload),
    )
    tmp_response = tmp_response.json()
    if "request_id" in tmp_response:
        req_id = tmp_response["request_id"]
        get_score_payload = {"request_id": f"{req_id}"}
    else:
        raise ValueError("Error in the Scoring")
    start = time.time()
    while True:
        sleep(0.05)
        scores_response = requests.request(
            "POST",
            BASE_URL + "getscores",
            headers=headers,
            data=json.dumps(get_score_payload),
        )
        scores_response = scores_response.json()
        if scores_response["completed"]:
            break
        end = time.time()
        if (end - start) >= 10:
            raise Exception("Timeout for get scores")
    scores = scores_response["data"]
    clsss = scores["CLSSS"]
    comet = [
        scores["COMET_QE, cased, punctuated"],
        scores["COMET_QE, cased, not punctuated"],
        scores["COMET_QE, uncased, punctuated"],
        scores["COMET_QE, uncased, not punctuated"],
    ]
    comet = [float(_) for _ in comet]
    return comet + [clsss]


if __name__ == "__main__":

    df_2020 = pd.read_csv(
        "./wmt-mqm-human-evaluation/newstest2020/ende/mqm_newstest2020_ende.tsv",
        sep="\t",
    )
    # df_2021 = pd.read_csv(
    #     "./wmt-mqm-human-evaluation/newstest2020/ende/mqm-newstest2021_ende.tsv",
    #     sep="\t",
    #     on_bad_lines="skip",
    # )

    fe = FeatureExtractor()
    data = []
    group = df_2020.groupby("source")
    p_bar = tqdm(range(df_2020.shape[0]))
    for source_text, subset in group:
        source_features = fe([source_text], "en")
        for i, row in subset.iterrows():
            sample_id = format(i, "08d")
            sample_path = f"./data/wmt_2020_with_bench/{sample_id}.pkl"
            if not os.path.exists(sample_path):
                target_text = clean_target(row.target)
                target_features = fe([target_text], "de")
                try:
                    comet_clsss = get_scores(source_text, target_text)
                except:
                    comet_clsss = [0, 0, 0, 0, 0]

                sample = {
                    "id": i,
                    "source_features": source_features,
                    "target_features": target_features,
                    "target-source": (target_features - source_features),
                    "category": row.category,
                    "severity": row.severity,
                    "commet_qe_clsss": comet_clsss,
                }
                with open(sample_path, "wb") as tf:
                    pickle.dump(sample, tf)
            p_bar.update(1)
            p_bar.refresh()
