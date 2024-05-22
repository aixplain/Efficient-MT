import sys

import numpy as np

sys.path.append("../")

from llm_utils import (
    get_best_translation_propmt,
    get_mqm_erros,
    get_postedit,
    get_translation_quality,
    select_best,
)


from gemba.CREDENTIALS import credentials
from gemba.prompt import prompts, language_codes
from gemba.gpt_api import GptApi
from gemba.cache import Cache
from gemba.gemba_mqm_utils import (
    TEMPLATE_GEMBA_MQM,
    apply_template,
    parse_mqm_answer,
)
from collections import defaultdict
from gemba.prompt import get_best_translation_propmt


data_path = "../data/wmt_2020_all_reduced_system_class_updated_predicted_categories.csv"


import pandas as pd

df = pd.read_csv(data_path)
df.head()
source, hypothesis, categories = (
    df["source"].tolist(),
    df["target"].tolist(),
    df["category"].tolist(),
)
from tqdm import tqdm

tqdm.pandas()

# predicted_categories_list = []
# for i in tqdm(range(len(source))):
#     try:
#         predicted_categories = get_mqm_erros(
#             hypothesis[i], source[i], src_lng="en", trg_lng="de", ref=None
#         )
#         predicted_categories = list(set(predicted_categories))
#         predicted_categories_list.append(predicted_categories)
#         df.loc[i, "predicted_categories"] = str(predicted_categories)
#     except Exception as e:
#         print(e)
#         predicted_categories_list.append([])
#         df.loc[i, "predicted_categories"] = None
#     try:
#         translation_quality = get_translation_quality(
#             source[i], hypothesis[i], src_lng="en", trg_lng="de", ref=None
#         )
#         df.loc[i, "translation_quality"] = translation_quality
#     except Exception as e:
#         print(e)
#         df.loc[i, "translation_quality"] = None

#     try:
#         postedit = get_postedit(
#             source[i], hypothesis[i], src_lng="en", trg_lng="de", ref=None
#         )
#         df.loc[i, "postedit"] = postedit
#     except Exception as e:
#         print(e)
#         df.loc[i, "postedit"] = None

#     try:
#         translation_quality_post_edit = get_translation_quality(
#             source[i], postedit, src_lng="en", trg_lng="de", ref=None
#         )
#         df.loc[i, "post_edit_translation_quality"] = translation_quality_post_edit
#     except Exception as e:
#         print(e)
#         df.loc[i, "post_edit_translation_quality"] = None
#     df.to_csv(data_path.replace(".csv", "_predicted_categories.csv"), index=False)

# df.to_csv(data_path.replace(".csv", "_predicted_categories.csv"), index=False)
# get the best mt for each source unique id
df["best_mt"] = None
for sample_id in tqdm(df.sample_id.unique()):
    mt_names = df[df.sample_id == sample_id].system.to_list()
    mts = df[df.sample_id == sample_id].target.to_list()
    src = df[df.sample_id == sample_id].source.to_list()[0]
    try:
        best_mt = select_best(src=src, mts=mts, src_lng="en", trg_lng="de")
        df.loc[df.sample_id == sample_id, "best_mt"] = np.array(mt_names)[best_mt]
    except Exception as e:
        print(e)
        df.loc[df.sample_id == sample_id, "best_mt"] = None


df.to_csv(
    "../data/wmt_2020_all_reduced_system_class_updated_predicted_categories_best_mt.csv",
    index=False,
)
