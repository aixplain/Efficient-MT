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
from gemba.prompt import get_best_translation_propmt, parse_numerical_answer


gptapi = GptApi(credentials, verbose=False)


def get_translation_quality(src, hyp, src_lng="en", trg_lng="de", ref=None):
    use_model = "GPT-4o"
    annotation = "GEMBA-DA"
    cache = Cache(f"{use_model}_{annotation}.jsonl")
    lng = "en-de"
    if prompts[annotation]["use_ref"] and ref is None:
        raise ValueError("Reference is required for this metric")

    data = {
        "source_seg": src,
        "target_seg": hyp,
        "reference_seg": ref,
        "source_lang": src_lng,
        "target_lang": trg_lng,
    }
    prompt = prompts[annotation]["prompt"].format(**data)
    parsed_answers = gptapi.request(
        prompt, use_model, prompts[annotation]["validate_answer"], cache=cache
    )
    return parsed_answers[0]["answer"]


def get_mqm_erros(src, hyp, src_lng="en", trg_lng="de", ref=None):
    use_model = "GPT-4o"
    cache = Cache(f"{use_model}_GEMBA-MQM.jsonl")
    data = {
        "source_seg": src,
        "target_seg": hyp,
        "source_lang": src_lng,
        "target_lang": trg_lng,
    }
    prompt = apply_template(TEMPLATE_GEMBA_MQM, data)
    parsed_answers = gptapi.request(
        prompt,
        use_model,
        lambda x: parse_mqm_answer(x, list_mqm_errors=True, full_desc=False),
        cache=cache,
    )

    errors = defaultdict(list)
    errors.update(parsed_answers[0]["answer"])
    error_list = errors["minor"] + errors["major"] + errors["critical"]

    return error_list


def get_postedit(src, hyp, src_lng="en", trg_lng="de", ref=None):
    use_model = "GPT-4o"
    annotation = "POSTEDIT"
    cache = Cache(f"{use_model}_{annotation}.jsonl")
    data = {
        "source_seg": src,
        "target_seg": hyp,
        "source_lang": src_lng,
        "target_lang": trg_lng,
    }
    prompt = prompts[annotation]["prompt"].format(**data)
    parsed_answers = gptapi.request(
        prompt, use_model, prompts[annotation]["validate_answer"], cache=cache
    )
    return parsed_answers[0]["answer"]


def select_best(src, mts, src_lng="en", trg_lng="de"):
    use_model = "GPT-4o"
    annotation = "SELECT-BEST"
    cache = Cache(f"{use_model}_{annotation}.jsonl")
    prompt = get_best_translation_propmt(src, mts, src_lng, trg_lng)
    parsed_answers = gptapi.request(
        prompt, use_model, parse_numerical_answer, cache=cache
    )
    return parsed_answers[0]["answer"]
