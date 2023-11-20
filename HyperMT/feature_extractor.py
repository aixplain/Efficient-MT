__author__ = "lucaspavanelli"

"""
Author: Lucas Pavanelli
Date: August 18th, 2021
Description:
    Extract features from batch of texts.
"""

import json

import numpy as np
import stanza
from tqdm import tqdm


class FeatureExtractor:
    def __init__(self):
        """
        Initialize the feature extractor
        """

        self.all_features = [
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
        ]

        self.stanza_map = {}
        with open("HyperMT/stanza_supported_languages.txt", "r") as f:
            for line in f:
                lang_code = line.split()[1]
                if lang_code == "en":
                    stanza.download(lang="en", processors={"ner": "conll03"})
                    self.stanza_map[lang_code] = stanza.Pipeline(
                        lang_code, processors={"ner": "conll03"}, use_gpu=True
                    )
                else:
                    stanza.download(lang=lang_code)
                    self.stanza_map[lang_code] = stanza.Pipeline(
                        lang_code, use_gpu=True
                    )

    def __call__(self, batch_text: list, language: str):
        """
        Extract features from a batch of texts to be translated
        """
        nlp = self.stanza_map[language]
        batch_result = []
        for input_sentence in batch_text:
            cur_features = np.zeros(len(self.all_features))
            num_char, num_token = 0, 0

            try:
                doc = nlp(input_sentence)

                for snt in doc.sentences:
                    num_token += len(snt.words)

                    # token features
                    for token in snt.words:
                        num_char += len(token.text)

                        # universal POS
                        upos = token.upos
                        upos_column = f"Num {upos}"
                        if upos_column in self.all_features:
                            cur_features[self.all_features.index(upos_column)] += 1

                        # morphological features
                        morph_str = token.feats if token.feats else ""
                        morph_list = [
                            morph_tag
                            for morph_tag in morph_str.split("|")
                            if morph_tag != ""
                        ]
                        for morph_tag in morph_list:
                            morph_column = f"Num {morph_tag}"
                            if morph_column in self.all_features:
                                cur_features[self.all_features.index(morph_column)] += 1

                    # named entities
                    for entity in snt.ents:
                        entity_column = f"Num {entity.type}"
                        if entity_column in self.all_features:
                            cur_features[self.all_features.index(entity_column)] += 1

            except Exception as e:
                print(f"ERROR: {input_sentence}")
                print("Message:")
                print(str(e))

            cur_features[self.all_features.index("Num token")] = num_token
            cur_features[self.all_features.index("Num char")] = num_char
            if num_token == 0:
                cur_features[self.all_features.index("Avg word length")] = 0
            else:
                cur_features[self.all_features.index("Avg word length")] = (
                    num_char / num_token
                )
            batch_result.append(cur_features)
        return np.array(batch_result)
