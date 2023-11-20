import json
import pdb
import time

import numpy as np
import requests
import streamlit as st

from utils import (
    BASE_URL,
    category2id,
    class2id,
    id2category,
    id2class,
    topic_label_list,
)  # noqa F401
from web_app.load_css import local_css
from llm_utils import get_postedit


def clean_target(text):
    return text.replace("<v>", "").replace("</v>", "")


def app():
    st.set_page_config(layout="wide")

    columns_sizes = (40, 2, 2, 2, 2, 2, 3, 5)

    if "samples" not in st.session_state:
        st.session_state["samples"] = [
            {
                "sample_id": "",
                "system": "INIT",
                "doc": "",
                "doc_id": 0,
                "seg_id": 0,
                "rater": "",
                "source": "",
                "target": "",
                "category": "Fluency/Punctuation",
                "severity": 2,
                "predicted_category": "Fluency/Punctuation",
                "predicted_severity": 1,
                "source_major_severity_prob": 0.5,
                "best_model": 0,
            }
        ]
    if "user_input" not in st.session_state:
        st.session_state["user_input"] = {}

    if "query_button" not in st.session_state:
        st.session_state["query_button"] = True

    if "latest_f1_score" not in st.session_state:
        scores = requests.get(BASE_URL + "get_sev_model_performance_online").json()
        st.session_state["latest_f1_score"] = scores["severity_model"]["f1_score"]

    if "bestSelected" not in st.session_state:
        st.session_state["bestSelected"] = False

    if "isFirstRun" not in st.session_state:
        st.session_state["isFirstRun"] = True

    if "espertise_areas" not in st.session_state:
        st.session_state["espertise_areas"] = None
    if "query_strategy" not in st.session_state:
        st.session_state["query_strategy"] = None
    if "rater_id" not in st.session_state:
        st.session_state["rater_id"] = None

    if "prev_expertise_areas" not in st.session_state:
        st.session_state["prev_expertise_areas"] = None
    if "prev_query_strategy" not in st.session_state:
        st.session_state["prev_query_strategy"] = None
    if "prev_rater_id" not in st.session_state:
        st.session_state["prev_rater_id"] = None

    efforts = list(class2id.keys())
    categories = list(category2id.keys())
    categories.remove("No-error")
    expertise_areas = topic_label_list

    def map_expertise_areas(areas):
        if "All" in areas:
            return expertise_areas

        return areas

    def update_sample(sample, post_edit, best_model, label, category, skip=False):
        ## Update sample with new label
        sample = requests.post(
            BASE_URL + "update_sample_online",
            params={
                "sample_id": sample["sample_id"],
                "post_edit": post_edit,
                "best_model": best_model,
                "sev_label": label,
                "skip": skip,
            },
            data=json.dumps(category),
        )

    def query():
        if len(st.session_state["user_input"]) > 0:
            for i, sample in enumerate(st.session_state["samples"]):
                if st.session_state["user_input"][sample["sample_id"]]["isBest"]:
                    ## TODO: after integrating automl learning uncomment here
                    print("SENDING UPDATE")
                    update_sample(
                        sample,
                        st.session_state["user_input"][sample["sample_id"]][
                            "post_edit"
                        ],
                        sample["system"],
                        st.session_state["user_input"][sample["sample_id"]][
                            "sev_label"
                        ],
                        st.session_state["user_input"][sample["sample_id"]]["category"],
                        skip=False,
                    )
                    scores = requests.get(
                        BASE_URL + "get_sev_model_performance_online"
                    ).json()
                    st.session_state["latest_f1_score"] = scores["severity_model"][
                        "f1_score"
                    ]

        # send a request to get next sample
        samples = requests.post(
            BASE_URL + "get_next_sample_online",
            params={
                "strategy": st.session_state["query_strategy"],
                "sort": "desc",
            },
            data=json.dumps(map_expertise_areas(st.session_state["espertise_areas"])),
        ).json()
        samples = samples["samples"]
        st.session_state["samples"] = samples
        st.session_state["bestSelected"] = False
        st.session_state["query_button"] = False

    def sample_container(sample):
        container = st.container()
        # Divide screen into 3 columns
        col1, col2, col3, col4, col5, col6, col7, col8 = container.columns(
            columns_sizes
        )
        col2.markdown(f"#")
        col2.markdown(f"#")
        col3.markdown(f"#")
        col3.markdown(f"#")
        col4.markdown(f"#")
        col4.markdown(f"#")
        col5.markdown(f"#")
        col5.markdown(f"#")
        col6.markdown(f"#")
        col6.markdown(f"#")
        col7.markdown(f"#")
        col7.markdown(f"#")
        col8.markdown(f"#")
        col8.markdown(f"#")

        # Write system name
        sample["target"] = f"{clean_target(sample['target'])}"
        # For all systems show all output sentences under each other and system names
        target = col1.text_area(
            "",
            sample["target"],
            key=f"sys_{sample['system']}",
        )

        major_accuracy = col2.checkbox(
            "A",
            key=f"Major_accuracy_{sample['system']}_{sample['sample_id']}",
            value=True if "Accuracy" in sample["predicted_category"] else False,
        )
        major_fluency = col3.checkbox(
            "F",
            key=f"Major_fluency_{sample['system']}_{sample['sample_id']}",
            value=True if "Fluency" in sample["predicted_category"] else False,
        )
        major_locale = col4.checkbox(
            "L",
            key=f"Major_locale_{sample['system']}_{sample['sample_id']}",
            value=True if "Locale" in sample["predicted_category"] else False,
        )
        major_terminology = col5.checkbox(
            "T",
            key=f"Major_terminology_{sample['system']}_{sample['sample_id']}",
            value=True if "Terminology" in sample["predicted_category"] else False,
        )
        major_other = col6.checkbox(
            "O",
            key=f"Major_other_{sample['system']}_{sample['sample_id']}",
            value=True if "Other" in sample["predicted_category"] else False,
        )

        majors = [
            major_accuracy,
            major_fluency,
            major_locale,
            major_terminology,
            major_other,
        ]
        categories_array = np.array(
            ["Accuracy", "Fluency", "Locale convention", "Terminology", "Other"]
        )
        if sum(majors) > 0:
            sev_label = "Edit"
            category = categories_array[majors]
        else:
            sev_label = "No-Edit"
            category = ["No-error"]

        if not isinstance(category, list):
            category = list(category)

        st.session_state["user_input"][sample["sample_id"]] = {
            "category": category,
            "sev_label": sev_label,
            "target": target,
            "isBest": False,
            "post_edit": None,
        }

        ph = st.empty()
        if not st.session_state["bestSelected"]:
            isbest = col8.checkbox(
                "",
                key=f"edit_{sample['system']}_{sample['sample_id']}",
                value=True if sample["is_best_model"] else False,
            )
            if isbest and not st.session_state["bestSelected"]:
                if (
                    st.session_state["user_input"][sample["sample_id"]]["post_edit"]
                    is None
                ):
                    postedit = get_postedit(
                        sample["source"],
                        sample["target"],
                        src_lng="en",
                        trg_lng="de",
                    )
                    st.session_state["user_input"][sample["sample_id"]][
                        "post_edit"
                    ] = postedit[1:-1]

                st.session_state["user_input"][sample["sample_id"]][
                    "post_edit"
                ] = ph.text_area(
                    "Post edit here",
                    st.session_state["user_input"][sample["sample_id"]]["post_edit"],
                    key=f"best_sys_{sample['system']}",
                )
                st.session_state["user_input"][sample["sample_id"]]["isBest"] = True

                # Request a predict api call to get post edit and previous comparision here
                probs = requests.get(
                    BASE_URL + "predict_sample_online",
                    params={
                        "source": sample["source"],
                        "post_edit": st.session_state["user_input"][
                            sample["sample_id"]
                        ]["post_edit"],
                    },
                ).json()
                source_goodness = 1.01 - sample["source_major_severity_prob"]
                target_goodness = 1.01 - probs["major_severity_prob"]

                col1, col2 = st.columns((15, 2))

                improvement = round(
                    min(
                        max(
                            (
                                (target_goodness - source_goodness)
                                / source_goodness
                                * 100
                            ),
                            0,
                        ),
                        100,
                    )
                )

                local_css("web_app/style.css")
                if improvement > 0:
                    col1.markdown(
                        f"<div>Great work, we guesstimate that you improved the machine translation by <span class='highlight green'>{improvement}% </span></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    col1.markdown(
                        f"<div>Great work, we guesstimate that there is not much improvement in the machine translation yet.</div>",
                        unsafe_allow_html=True,
                    )

                delta_list = []
                for cat in list(probs["predicted_category"]):
                    if (
                        cat
                        not in st.session_state["user_input"][sample["sample_id"]][
                            "category"
                        ]
                    ):
                        if cat != "No-error":
                            delta_list.append(cat)

                if len(delta_list) == 0:
                    t = f"<div>We estimate that you have already fixed all the existing problems and the translation does not require any further edits. You can simply click submit to receive next-item.</div>"
                else:
                    t = "<div>But, we still guess that you can improve its"
                    for pred_cat in delta_list:
                        if pred_cat != "No-error":
                            if pred_cat == "Locale convention":
                                pred_cat = "Locale"
                            t += f"<span class='highlight red'><span class='bold'>{pred_cat}</span></span>   "
                    t += " errors.</div>"
                col1.markdown(t, unsafe_allow_html=True)
                col1.markdown(
                    f"If you agree, please do that and click to submit. If not, just click to submit to receive the next item."
                )
                col2.markdown(f"#")
                submit = col2.button("Submit", key=f"submit_{sample['system']}")
                if submit:
                    st.session_state["query_button"] = True

        return "not-updated"

    """
    # Rate Systems
    """

    progress_status = st.sidebar.empty()
    progress_bar = st.sidebar.empty()

    if st.session_state["isFirstRun"]:
        if st.session_state["query_strategy"] is None:
            with st.spinner("Select Query Strategy"):
                st.session_state["query_strategy"] = st.sidebar.selectbox(
                    "Please Select the Query Strategy",
                    [
                        "Greedy (Regression)",
                        "Curiosity (Uncertainity)",
                        "Mixed",
                    ],
                )
                time.sleep(3)
        else:
            st.session_state["query_strategy"] = st.sidebar.selectbox(
                "Please Select the Query Strategy",
                [
                    "Greedy (Regression)",
                    "Curiosity (Uncertainity)",
                    "Mixed",
                ],
            )
        if st.session_state["rater_id"] is None:
            with st.spinner("Select Your Rater ID"):
                st.session_state["rater_id"] = st.sidebar.selectbox(
                    "Rater ID",
                    ["Rater1", "Rater2", "Rater3", "Rater4", "Rater5", "Rater6"],
                )
                time.sleep(3)
        else:
            st.session_state["rater_id"] = st.sidebar.selectbox(
                "Rater ID", ["Rater1", "Rater2", "Rater3", "Rater4", "Rater5", "Rater6"]
            )

        if st.session_state["espertise_areas"] is None:
            with st.spinner("Select your Expertise Areas"):
                st.session_state["espertise_areas"] = st.sidebar.multiselect(
                    "Expertise Areas", expertise_areas
                )
                time.sleep(3)
        else:
            # Select Raters Expertise Areas in Topics
            st.session_state["espertise_areas"] = st.sidebar.multiselect(
                "Expertise Areas", expertise_areas
            )
    else:
        # Select strategy to bring new samples [Entropy, Uncertainity, Translation Quality]
        st.session_state["query_strategy"] = st.sidebar.selectbox(
            "Please Select the Query Strategy",
            [
                "Greedy (Regression)",
                "Curiosity (Uncertainity)",
                "Mixed",
            ],
        )

        st.session_state["rater_id"] = st.sidebar.selectbox(
            "Rater ID", ["Rater1", "Rater2", "Rater3", "Rater4", "Rater5", "Rater6"]
        )

        # Select Raters Expertise Areas in Topics
        st.session_state["espertise_areas"] = st.sidebar.multiselect(
            "Expertise Areas", expertise_areas
        )

    if st.session_state["espertise_areas"] == []:
        st.session_state["espertise_areas"] = "All"
    # Infomation on remaining and rated samples count
    stats = requests.post(
        BASE_URL + "get_data_stats",
        params={
            "rater": st.session_state["rater_id"],
        },
        data=json.dumps(map_expertise_areas(st.session_state["espertise_areas"])),
    ).json()

    progress_status.write(f"{stats['labeled']} Rated, {stats['unlabeled']} Remaining")
    my_bar = progress_bar.progress(float(stats["labeled"] / stats["unlabeled"]))

    # Accuracy:"If translation includes any of the following: Untranslated text, Omission, Mistranslation, Addition",
    # Fluency: "Spelling, grammar, or punctuation errors are categorized as Fluency errors.",
    # Locale: "Formatting errors in Currency, Date, Time, or Adress are categorized as Locale convention errors.",
    # Terminology: "Inconsistent and inappropriate terminology usage is categorized as Terminology errors.",
    # Other: "Any other errors not covered by the above categories are categorized as Other errors.",
    text = """
    ## Error Categories"

    ##### (A)ccuracy:
    If translation includes any of the following: Untranslated text, Omission, Mistranslation, Addition,

    ##### (F)luency:
    Spelling, grammar, or punctuation errors are categorized as Fluency errors.

    ##### (L)ocale:
    Formatting errors in Currency, Date, Time, or Adress are categorized as Locale convention errors.

    ##### (T)erminology:
    Inconsistent and inappropriate terminology usage is categorized as Terminology errors.

    ##### (O)ther:
    Any other errors not covered by the above categories are categorized as Other errors.

    """
    st.sidebar.markdown(text)

    col1, col2 = st.sidebar.columns((3, 1))

    # if (
    #     (st.session_state["query_strategy"] != st.session_state["prev_query_strategy"])
    #     or (
    #         st.session_state["espertise_areas"]
    #         != st.session_state["prev_espertise_areas"]
    #     )
    #     or (st.session_state["rater_id"] != st.session_state["prev_rater_id"])
    # ):
    #     st.session_state["query_button"] = True
    if st.session_state["query_button"] and st.session_state["isFirstRun"]:
        st.balloons()
        st.session_state["isFirstRun"] = False
        query()
    elif st.session_state["query_button"]:
        query()

    col1, col2 = st.columns((15, 3))
    local_css("web_app/style.css")
    col2.markdown(
        f"<div>Model's Performance <span class='highlight green'>{st.session_state['latest_f1_score']}%</span></div>",
        unsafe_allow_html=True,
    )
    if not st.session_state["isFirstRun"]:
        if len(st.session_state["samples"]) > 0:
            # Input sentence
            st.markdown(f"**Input Sentence**")
            text = st.text_area(
                "", f"{st.session_state['samples'][0]['source']}", key="text"
            )

            col1, col2, col3 = st.columns(
                (columns_sizes[0], sum(columns_sizes[1:-2]), columns_sizes[-1])
            )
            col1.markdown(f"**Output Sentences**")
            col2.markdown("**Error Categories**")
            col3.markdown(f"**Is Best Translation**")

            st.session_state["user_input"] = {}
            for sample in st.session_state["samples"]:
                sample_container(sample)

        else:
            st.warning("No samples to rate")


app()
