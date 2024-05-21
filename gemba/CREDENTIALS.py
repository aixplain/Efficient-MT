# deployment is used to list available models
# for Azure API, specify model name as a key and deployment name as a value
# for OpenAI API, specify model name as a key and a value
credentials = {
    "api_type": "aixplain",
    "deployments": {
        "GPT-4": "6414bd3cd09663e9225130e8",
        "ChatGPT-3.5": "646796796eb56367b25d0751",
        "ChatGPT": "640b517694bf816d35a59125",
        "GPT-4o": "6646261c6eb563165658bbb1",
    },
    "requests_per_second_limit": 1,
}
