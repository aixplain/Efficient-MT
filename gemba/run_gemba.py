from gemba.CREDENTIALS import credentials
from gemba.prompt import prompts, language_codes
from gemba.gpt_api import GptApi
from gemba.cache import Cache


def main():
    scenarios = [
        [
            "ChatGPT",
            "GEMBA-DA",
        ],
    ]

    gptapi = GptApi(credentials)
    for scenario in scenarios:
        use_model = scenario[0]
        annotation = scenario[1]
        cache = Cache(f"{use_model}_{annotation}.jsonl")

        scoring_name = f"{annotation}_{use_model}"

        if use_model not in credentials["deployments"].keys():
            print(f"Model {use_model} not supported by credentials")
            continue

        src = "This is a test sentence in english"
        hyp = "Das ist ein Testsatz in Deutsch"
        ref = "Dies ist ein Testsatz in Deutsch"
        lng = "en-de"
        if prompts[annotation]["use_ref"]:
            ref = None
        data = {
            "source_seg": src,
            "target_seg": hyp,
            "reference_seg": ref,
            "source_lang": language_codes[lng.split("-")[0]],
            "target_lang": language_codes[lng.split("-")[1]],
        }
        prompt = prompts[annotation]["prompt"].format(**data)
        parsed_answers = gptapi.request(prompt, use_model, prompts[annotation]["validate_answer"], cache=cache)


if __name__ == "__main__":
    main()
