import pdb
import time
from termcolor import colored
from datetime import datetime
import os

from dotenv import load_dotenv, find_dotenv
from src.paths import paths


load_dotenv(find_dotenv(paths.PROJECT_ROOT_DIR / "secrets.env"), override=True)
load_dotenv(find_dotenv(paths.PROJECT_ROOT_DIR / "vars.env"), override=True)


from aixplain.factories import ModelFactory


# class for calling OpenAI API and handling cache
class GptApi:
    def __init__(self, credentials, verbose=True):
        assert "deployments" in credentials, "deployments not found in credentials"

        self.deployments = credentials["deployments"]
        self.verbose = verbose

        # Azure API access
        self.api_type = credentials["api_type"]
        self.deployments = credentials["deployments"]

        # limit the number of requests per second
        if "requests_per_second_limit" in credentials:
            self.rps_limit = 1 / credentials["requests_per_second_limit"]
        else:
            self.rps_limit = 0
        self.last_call_timestamp = 0

    # answer_id is used for determining if it was the top answer or how deep in the list it was
    def request(
        self, prompt, model, parse_response, temperature=0, answer_id=-1, cache=None
    ):
        max_tokens = 2000
        answers = None
        if cache is not None:
            prompt_key = prompt
            if isinstance(prompt, list):
                prompt_key = prompt[-1]["content"]
            answers = cache.get(
                {
                    "model": model,
                    "temperature": temperature,
                    "prompt": prompt_key,
                }
            )

        if answers is None:
            answers = self.request_api(prompt, model, temperature, max_tokens)

            if cache is not None:
                cache.add(
                    {
                        "model": model,
                        "temperature": temperature,
                        "prompt": prompt_key,
                        "answers": answers,
                    }
                )
        else:
            if self.verbose:
                print(
                    f"Answer (t={temperature}) (from cache): "
                    + colored(answers, "yellow")
                )

        # there is no valid answer
        if len(answers) == 0:
            return [
                {
                    "temperature": temperature,
                    "answer_id": answer_id,
                    "answer": None,
                    "prompt": prompt,
                    "finish_reason": None,
                    "model": model,
                }
            ]

        parsed_answers = []
        for full_answer in answers:
            full_answer = full_answer["answer"]
            answer_id += 1

            answer = parse_response(full_answer)
            if self.verbose or temperature > 0:
                print(
                    f"Answer (t={temperature}): "
                    + colored(answer, "yellow")
                    + " ("
                    + colored(full_answer, "blue")
                    + ")"
                )
            if answer is None:
                continue
            parsed_answers.append(
                {
                    "temperature": temperature,
                    "answer_id": answer_id,
                    "answer": answer,
                    "prompt": prompt,
                    "model": model,
                }
            )

        # there was no valid answer, increase temperature and try again
        if len(parsed_answers) == 0 and temperature < 3:
            return self.request(
                prompt,
                model,
                parse_response,
                temperature=temperature + 1,
                answer_id=answer_id,
                cache=cache,
            )

        return parsed_answers

    def request_api(self, prompt, model, temperature=0, max_tokens=20):
        # if temperature is 0, then request only 1 response
        n = 1

        if max_tokens > 2000 or temperature > 10:
            return []

        dt = datetime.now()
        ts = datetime.timestamp(dt)
        if ts - self.last_call_timestamp < self.rps_limit:
            time.sleep(self.rps_limit - (ts - self.last_call_timestamp))

        self.last_call_timestamp = ts

        if self.verbose:
            print(prompt)
        while True:
            try:
                response = self.call_api(prompt, model, n, temperature, max_tokens)
                break
            except Exception as e:
                # response was filtered
                if hasattr(e, "code"):
                    if e.code == "content_filter":
                        return []
                    print(e.code)
                # frequent error is reaching the API limit
                print(colored("Error, retrying...", "red"))
                print(e)
                time.sleep(1)

        answers = []

        answer = response["data"].strip()
        # one of the responses didn't finish, we need to request more tokens

        if response["status"] != "SUCCESS":
            if self.verbose:
                print(
                    colored(f"Increasing max tokens to fit answers.", "red")
                    + colored(answer, "blue")
                )
            return self.request_api(
                prompt, model, temperature=temperature, max_tokens=max_tokens + 200
            )

        answers.append({"answer": answer})

        if len(answers) > 1:
            # remove duplicate answers
            answers = [dict(t) for t in {tuple(d.items()) for d in answers}]

        return answers

    def call_api(self, prompt, model, n, temperature, max_tokens):
        if self.api_type == "aixplain":
            model_obj = ModelFactory.get(self.deployments[model])
            if isinstance(prompt, str):
                prompt = [{"role": "assistant", "content": prompt}]
            elif isinstance(prompt, list):
                pass

            response = model_obj.run(
                data=prompt,
                parameters={"max_tokens": max_tokens, "temperature": temperature / 10},
            )
            return response

    def bulk_request(self, df, model, parse_mqm_answer, cache, max_tokens=20):
        answers = []
        for i, row in df.iterrows():
            prompt = row["prompt"]
            parsed_answers = self.request(
                prompt, model, parse_mqm_answer, cache=cache, max_tokens=max_tokens
            )
            answers += parsed_answers
        return answers
