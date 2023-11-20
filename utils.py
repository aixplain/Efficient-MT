import pdb
import pickle
import string

import numpy as np
from vowpalwabbit import pyvw
import nltk


from data_centric.uncertainty import (
    entropy_sampling,
    margin_sampling,
    prediction_sampling,
    regression_entropy_combined,
    regression_uncertainity_combined,
    uncertainty_sampling,
)

NS_LIST = list(string.ascii_lowercase) + list(string.ascii_uppercase)


def default_label_str(y):
    return f"{y}"


def default_feature_str(x: list):
    return " ".join([f"{j}:{x[j]}" for j in range(len(x))])


def get_training_example(
    x, y, label_str_fn=default_label_str, feature_str_fn=default_feature_str
):
    return f"{label_str_fn(y)} | {feature_str_fn(x)}"


def get_test_example(
    x, label_str_fn=default_label_str, feature_str_fn=default_feature_str
):
    return f"| {feature_str_fn(x)}"


def get_vw_examples(
    X,
    y=None,
    label_str_fn=default_label_str,
    feature_str_fn=default_feature_str,
    isTrain=True,
):
    vw_examples = []
    for i in range(X.shape[0]):
        if isTrain:
            assert y is not None, "y must be provided"
            vw_examples.append(
                get_training_example(
                    X[i], y[i], label_str_fn=label_str_fn, feature_str_fn=feature_str_fn
                )
            )
        else:
            vw_examples.append(
                get_test_example(
                    X[i], label_str_fn=label_str_fn, feature_str_fn=feature_str_fn
                )
            )
    return vw_examples


with open("./data/group_indexes.pkl", "rb") as f:
    group_indexes = pickle.load(f)


def get_training_sample(x, y):
    ns_content = []
    for zz in range(len(group_indexes)):
        ns_features = " ".join(
            "{}:{:.6f}".format(ind, x[ind]) for ind in group_indexes[zz]
        )
        ns_content.append(ns_features)
    ns_line = "{} |{}".format(
        str(y),
        "|".join(
            "{} {}".format(NS_LIST[j], ns_content[j]) for j in range(len(group_indexes))
        ),
    )
    return ns_line


def get_test_sample(x):
    ns_content = []
    for zz in range(len(group_indexes)):
        ns_features = " ".join(
            "{}:{:.6f}".format(ind, x[ind]) for ind in group_indexes[zz]
        )
        ns_content.append(ns_features)
    ns_line = "|{}".format(
        "|".join(
            "{} {}".format(NS_LIST[j], ns_content[j]) for j in range(len(group_indexes))
        )
    )
    return ns_line


def query_next_sample(vw_alg, X_pool, n: int = 1):
    pool_examples = get_vw_examples(X_pool, isTrain=False)

    preds = [float(np.argmax(vw_alg.predict(ex)) + 1) for ex in pool_examples]
    idxs = np.argsort(preds)[::-1]
    idxs = idxs[:n]
    return idxs


def query_next_sample_interaction(vw_alg, X_pool, n: int = 1):
    pool_examples = [get_test_sample(_) for _ in X_pool]

    preds = [float(np.argmax(vw_alg.predict(ex)) + 1) for ex in pool_examples]
    idxs = np.argsort(preds)[::-1]
    idxs = idxs[:n]
    return idxs


feature_names = [
    "Num token",
    "Num char",
    "Avg word length",
    "Num ADJ",
    "Num ADP",
    "Num ADV",
    "Num AUX",
    "Num CCONJ",
    "Num DET",
    "Num INTJ",
    "Num NOUN",
    "Num NUM",
    "Num PART",
    "Num PRON",
    "Num PROPN",
    "Num PUNCT",
    "Num SCONJ",
    "Num SYM",
    "Num VERB",
    "Num X",
    "Num LOC",
    "Num MISC",
    "Num ORG",
    "Num PER",
    "Num Abbr=Yes",
    "Num Case=Acc",
    "Num Case=Nom",
    "Num Definite=Def",
    "Num Definite=Ind",
    "Num Degree=Cmp",
    "Num Degree=Pos",
    "Num Degree=Sup",
    "Num Foreign=Yes",
    "Num Gender=Fem",
    "Num Gender=Masc",
    "Num Gender=Neut",
    "Num Mood=Imp",
    "Num Mood=Ind",
    "Num NumForm=Digit",
    "Num NumForm=Word",
    "Num NumType=Card",
    "Num NumType=Mult",
    "Num NumType=Ord",
    "Num Number=Plur",
    "Num Number=Sing",
    "Num Person=1",
    "Num Person=2",
    "Num Person=3",
    "Num Polarity=Neg",
    "Num Poss=Yes",
    "Num PronType=Art",
    "Num PronType=Dem",
    "Num PronType=Int",
    "Num PronType=Prs",
    "Num PronType=Rel",
    "Num Reflex=Yes",
    "Num Tense=Past",
    "Num Tense=Pres",
    "Num VerbForm=Fin",
    "Num VerbForm=Ger",
    "Num VerbForm=Inf",
    "Num VerbForm=Part",
    "Num Voice=Pass",
    "Num Style=Expr",
    "Num NumForm=Roman",
    "Num Mood=Cnd",
    "Num Mood=Sub",
    "Num Number[psor]=Plur",
    "Num Number[psor]=Sing",
    "Num Person[psor]=1",
    "Num Person[psor]=2",
    "Num Person[psor]=3",
    "Num PronType=Exc",
    "Num PronType=Ind",
    "Num PronType=Neg",
    "Num Tense=Fut",
    "Num Tense=Imp",
    "Num Typo=Yes",
    "Num Case=Dat",
    "Num Case=Gen",
    "Num Gender[psor]=Masc,Neut",
    "Num Animacy=Anim",
    "Num Animacy=Inan",
    "Num Aspect=Imp",
    "Num Aspect=Perf",
    "Num Case=Ins",
    "Num Case=Loc",
    "Num Variant=Short",
    "Num VerbForm=Conv",
    "Num Voice=Act",
    "Num Voice=Mid",
    "Num AdpType=Comprep",
    "Num AdpType=Prep",
    "Num AdpType=Voc",
    "Num Case=Voc",
    "Num ConjType=Oper",
    "Num Gender=Fem,Masc",
    "Num Gender=Fem,Neut",
    "Num Gender=Masc,Neut",
    "Num Gender[psor]=Fem",
    "Num Gender[psor]=Masc",
    "Num Hyph=Yes",
    "Num NameType=Com",
    "Num NameType=Geo",
    "Num NameType=Giv",
    "Num NameType=Nat",
    "Num NameType=Sur",
    "Num NumType=Frac",
    "Num NumType=Sets",
    "Num NumValue=1",
    "Num NumValue=1,2,3",
    "Num Number=Dual",
    "Num Number=Plur,Sing",
    "Num Polarity=Pos",
    "Num PrepCase=Npr",
    "Num PrepCase=Pre",
    "Num PronType=Emp",
    "Num PronType=Int,Rel",
    "Num PronType=Tot",
    "Num Style=Arch",
    "Num Style=Coll",
    "COMET_QE, cased, punctuated",
    "COMET_QE, cased, not punctuated",
    "COMET_QE, uncased, punctuated",
    "COMET_QE, uncased, not punctuated",
    "CLSSS",
    "topic_distance",
    "labse_distance",
]


class2id = {
    "No-Edit": 1,
    "Edit": 2,
}
id2class = {v: k for k, v in class2id.items()}


id2category = {
    0: "Accuracy",
    1: "Fluency",
    2: "Locale convention",
    3: "No-error",
    4: "Other",
    5: "Terminology",
}

category2id = {v: k for k, v in id2category.items()}


id2system = {
    1: "MT A",
    2: "MT B",
    3: "MT C",
    4: "MT D",
    5: "MT E",
    6: "MT F",
    7: "MT G",
    8: "MT H",
    9: "MT I",
    10: "MT J",
}
system2id = {v: k for k, v in id2system.items()}


def save_model(model, filename):
    print("Saved model in {}".format(filename))
    pickle.dump(model, open(filename, "wb"))


def load_model(filename):
    print("Loaded model from {}".format(filename))
    return pickle.load(open(filename, "rb"))


def save_model_online(model, filename):
    print("Saved model in {}".format(filename))
    model.save(filename)


def load_model_online(filename):
    print("Loaded model from {}".format(filename))
    model = pyvw.vw(initial_regressor=filename)
    return model


BASE_URL = "http://0.0.0.0:8088/"

strategy_map = {
    "Greedy (Regression)": prediction_sampling,
    "Curiosity (Uncertainity)": uncertainty_sampling,
    "Mixed": regression_uncertainity_combined,
}


def get_topic_features(embds):
    mean = np.mean(embds, axis=0).reshape(1, -1)
    median = np.median(embds, axis=0).reshape(1, -1)
    std = np.std(embds, axis=0).reshape(1, -1)

    feature = np.concatenate([mean, median, std], axis=1)
    return feature


topic_label_list = [
    "Society & Culture",
    "Science & Mathematics",
    "Health",
    "Education & Reference",
    "Computers & Internet",
    "Sports",
    "Business & Finance",
    "Entertainment & Music",
    "Family & Relationships",
    "Politics & Government",
    "Other",
]

system_map = {
    "Human-A.0": "MT A",
    "Human-B.0": "MT B",
    "Human-P.0": "MT C",
    "Huoshan_Translate.832": "MT D",
    "OPPO.1535": "MT E",
    "Online-A.1574": "MT F",
    "Online-B.1590": "MT G",
    "Tencent_Translation.1520": "MT H",
    "Tohoku-AIP-NTT.890": "MT I",
    "eTranslation.737": "MT J",
}


def get_topic_features(embds):
    mean = np.mean(embds, axis=0).reshape(1, -1)
    median = np.median(embds, axis=0).reshape(1, -1)
    std = np.std(embds, axis=0).reshape(1, -1)

    feature = np.concatenate([mean, median, std], axis=1)
    return feature


def get_topic_proba(intext, text_embedder, topic_model):
    sentences = nltk.sent_tokenize(intext)
    emb = text_embedder.get_embedding(list(sentences)).reshape(-1, 768)
    proba = np.asarray(topic_model.predict_proba(get_topic_features(emb))).reshape(
        1, -1
    )
    first_class_id = np.argmax(proba, 1)[0]
    first_proba = float(proba[0, first_class_id])
    if first_proba < (
        2 * 0.11691211
    ):  ## Threshold for unknown class, this was found by optimizing true positive rate and true negative rate
        first_class_id = 10  # Unknown class
    pred_class = topic_label_list[first_class_id]
    return proba, pred_class


category_replace = {
    "no-error": "No-error",
    "Other/-": "Other",
    "Terminology/Inappropriate for context": "Terminology/Inappropriate",
    "Terminology/Inconsistent use of terminology": "Terminology/Inconsistent",
}

categories_map = {
    " Style/Awkward": "Other",
    "Accuracy/Addition": "Accuracy",
    " Terminology/Inconsistent": "Terminology",
    " Fluency/Register": "Fluency",
    " Locale convention/Currency format": "Locale convention",
    "No-error": "No-error",
    "Fluency/Inconsistency": "Fluency",
    " Locale convention/Time format": "Locale convention",
    " Fluency/Spelling": "Fluency",
    "Other": "Other",
    " Fluency/Grammar": "Fluency",
    "Accuracy/Untranslated text": "Accuracy",
    "Terminology/Inappropriate": "Terminology",
    "Fluency/Register": "Fluency",
    " No-error": "No-error",
    "Locale convention/Date format": "Locale convention",
    " Terminology/Inappropriate": "Terminology",
    " Other": "Other",
    "Accuracy/Omission": "Accuracy",
    "Fluency/Character encoding": "Fluency",
    "Fluency/Grammar": "Fluency",
    " Accuracy/Mistranslation": "Accuracy",
    " Fluency/Punctuation": "Fluency",
    "Fluency/Punctuation": "Fluency",
    " Non-translation!": "Other",
    "Locale convention/Time format": "Locale convention",
    " Fluency/Inconsistency": "Fluency",
    " Fluency/Character encoding": "Fluency",
    " Accuracy/Addition": "Accuracy",
    "Terminology/Inconsistent": "Terminology",
    " Accuracy/Omission": "Accuracy",
    "Accuracy/Mistranslation": "Accuracy",
    " Accuracy/Untranslated text": "Accuracy",
    " Locale convention/Address format": "Locale convention",
    "Style/Awkward": "Other",
    "Fluency/Spelling": "Fluency",
    "Locale convention/Currency format": "Locale convention",
    "Locale convention/Address format": "Locale convention",
    "Non-translation!": "Other",
}
