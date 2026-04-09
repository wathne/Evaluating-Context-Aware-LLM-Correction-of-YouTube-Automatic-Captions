# OpenAI GPT LLM module.

# Module requirements (included in requirements.txt):
# openai

# For convenience, let "DHH" be a shortened reference to the following study:
# "Empowering the Deaf and Hard of Hearing Community: Enhancing Video Captions
# Using Large Language Models".
# https://arxiv.org/abs/2412.00342
# https://github.com/monikabhole001/Improving-the-Quality-of-Video-Captions-for-the-DHH-Community-Using-LLM


from captions_with_evaluation_results_wrapper import Record
from captions_with_evaluation_results_wrapper import RecordList
from helpers import randomized_wait
from helpers import str_or_none_if_empty
from json import JSONDecoder
from openai import OpenAI
from openai.types.responses import Response
from os import getenv


GPT_3_5_MODEL: str = "gpt-3.5-turbo-0125"
GPT_5_4_MINI_MODEL: str = "gpt-5.4-mini-2026-03-17"
GPT_5_4_MODEL: str = "gpt-5.4-2026-03-05"
DEFAULT_MODEL: str = GPT_5_4_MODEL
DHH_MODEL: str = GPT_3_5_MODEL

# This prompt is included only for reference. It is an exact stringified copy of
# the GPT 2 prompt of the DHH study, as is, from their GitHub repository. Note
# that the DHH study used GPT 3.5 Turbo, not GPT 2, to generate their final GPT
# captions.
_DHH_PROMPT_GPT_2: str = (
    "Correct the following caption as per english standard."
    " Do not give additional information.\n"
    "Caption:\n"
)

# This prompt is included only for reference. It is an exact stringified copy of
# the GPT 3.5 prompt of the DHH study, as is, from their github repository. Note
# that the prompt instructs the LLM to make the caption relevant to education.
# It is unclear if this exact prompt, with "education", was used to generate all
# GPT captions. They may or may not have swapped "education" for the other
# categories: "cooking", "entertainment", "news", and "travel". The DHH study is
# not explicit about using category as context. See also the get_dhh_prompt
# function below, it provides the same DHH GPT 3.5 prompt, but adapted to
# category. Another possibility is that none of the prompts available in the DHH
# GitHub repository were actually used to generate their final LLM captions.
_DHH_PROMPT_GPT_3_5: str = (
    "Correct the following caption as per english standard."
    " Make the caption relevant to education.\n"
    "Caption:\n"
)

# TODO(wathne): Should scoring metrics not be mentioned?
# Default GPT pre-prompt. To be used by the dynamic get_prompt function.
# The pre-prompt contains the following statement as copied from Google support:
# "... automatic captions might misrepresent the spoken content due to
# mispronunciations, accents, dialects, or background noise."
# https://support.google.com/youtube/answer/6373554
DEFAULT_PROMPT_PRE: str = (
    "YouTube uses automatic speech recognition (ASR) to add automatic captions"
    " to YouTube videos. You will receive an exact stringified copy of one such"
    " automatic caption. Automatic captions might misrepresent the spoken"
    " content due to mispronunciations, accents, dialects, or background noise."
    " The YouTube ASR lacks context and can make mistakes that would be obvious"
    " to context-aware humans. Your task is to correct the entire automatic"
    " caption and return the entire corrected caption. Know that your response"
    " is part of an automated evaluation pipeline and will be stringified and"
    " forwarded as it is. The corrected caption is later scored against a"
    " ground truth caption by metrics WER, BLEU, and ROUGE.\n"
)

# Default GPT metadata-intro-prompt. To be used by the dynamic get_prompt
# function.
DEFAULT_PROMPT_METADATA_INTRO: str = (
    "An automatic process has fetched supplementary metadata for the YouTube"
    " video. You are provided with this supplementary metadata as follows:\n"
)

# Default GPT post-prompt. To be used by the dynamic get_prompt function.
DEFAULT_PROMPT_POST: str = "\nAutomatic caption:\n"


def _get_api_key() -> str | None:
    api_key: str | None
    try:
        from private_api_keys import OPENAI_GPT_API_KEY
        api_key = str_or_none_if_empty(OPENAI_GPT_API_KEY)
    except (ImportError, NameError) as instance:
        print(instance)
        api_key = getenv(
            key="OPENAI_GPT_API_KEY",
            default=None,
        )

    return api_key


# Returns the DHH GPT 3.5 prompt, but adapted to category.
def get_dhh_prompt(
    category: str | None = None,
) -> str:
    category = str_or_none_if_empty(category)
    if category is None:
        return (
            "Correct the following caption as per english standard.\n"
            "Caption:\n"
        )

    category = category.lower()

    return (
        "Correct the following caption as per english standard."
        f" Make the caption relevant to {category}.\n"
        "Caption:\n"
    )


def get_prompt(
    record: Record,
    provide_category: bool = False,
    provide_title: bool = False,
    provide_description: bool = False,
    provide_top_comments: bool = False,
) -> str:
    prompt: str = ""

    prompt += DEFAULT_PROMPT_PRE

    if (
        provide_category or
        provide_title or
        provide_description or
        provide_top_comments
    ):
        prompt += DEFAULT_PROMPT_METADATA_INTRO

    if provide_category:
        category: str | None = str_or_none_if_empty(record.category)
        if category is not None:
            prompt += f"metadata:video_category: {category}\n"

    if (
        provide_title or
        provide_description or
        provide_top_comments
    ):
        title: str | None
        description: str | None
        comment: str | None
        top_comments: list[str] = []

        # Initializes an instance of JSONDecoder.
        json_decoder: JSONDecoder = JSONDecoder(
            object_hook=None,
            parse_float=None,
            parse_int=None,
            parse_constant=None,
            strict=True,
            object_pairs_hook=None,
        )

        comment_dict: dict[str, str]
        comment_dict_list: str | list[dict[str, str]] # list[dict[str, str]]
        metadata_dict: dict[str, str | list[dict[str, str]]]
        metadata_json: str | None

        metadata_json = record.metadata
        if metadata_json is not None:
            metadata_dict = json_decoder.decode(metadata_json)

            if provide_title:
                title = str_or_none_if_empty(metadata_dict["title"])
                if title is not None:
                    prompt += f"metadata:video_title: {title}\n"

            if provide_description:
                description = str_or_none_if_empty(metadata_dict["description"])
                if description is not None:
                    prompt += f"metadata:video_description: {description}\n"

            if provide_top_comments:
                comment_dict_list = metadata_dict["top_comments"]
                if isinstance(comment_dict_list, list):
                    for comment_dict in comment_dict_list:
                        comment = str_or_none_if_empty(comment_dict["comment"])
                        if comment is not None:
                            top_comments.append(comment)

                for comment in top_comments:
                    prompt += f"metadata:top_comments:comment: {comment}\n"

    prompt += DEFAULT_PROMPT_POST

    # TODO(wathne): Test. Remove this.
    #print(prompt)

    return prompt


def generate_openai_gpt_caption_for_records(
    records: RecordList,
    model: str | None = DEFAULT_MODEL,
    dhh_model: bool = False, # True will force the use of DHH_MODEL (GPT 3.5).
    dhh_prompt: bool = False, # True will use the DHH prompt.
    dhh_parameters: bool = False, # Only applicable to GPT 3.5.
    provide_category: bool = False,
    provide_title: bool = False, # Not applicable to DHH prompt.
    provide_description: bool = False, # Not applicable to DHH prompt.
    provide_top_comments: bool = False, # Not applicable to DHH prompt.
    wait_milliseconds_min: int | None = 12000, # 12 seconds
    wait_milliseconds_max: int | None = 24000, # 24 seconds
) -> None:
    try:
        dhh_model = bool(dhh_model)
    except (TypeError, ValueError):
        dhh_model = False

    try:
        dhh_prompt = bool(dhh_prompt)
    except (TypeError, ValueError):
        dhh_prompt = False

    api_key: str | None = _get_api_key()
    if api_key is None:
        print("Error: OPENAI_GPT_API_KEY does not exist.")
        return None

    client: OpenAI = OpenAI(
        api_key=api_key,
    )

    if dhh_model:
        model = str_or_none_if_empty(DHH_MODEL)

        if model is None:
            print("Error: DHH_MODEL does not exist.")
            return None

    if model is None:
        model = str_or_none_if_empty(DEFAULT_MODEL)

    if model is None:
        print("Error: DEFAULT_MODEL does not exist.")
        return None

    prompt: str # instructions
    youtube_caption: str | None
    response: Response
    chatgpt_caption: str | None

    youtube_video_id: str | None

    length: int = len(records)
    i: int
    record: Record # Reference/pointer to a mutable Record.
    for i, record in enumerate(records):
        print(f"({i+1}/{length}) ", end="", flush=True)

        if dhh_prompt:
            if provide_category:
                prompt = get_dhh_prompt(category=record.category)
            else:
                prompt = get_dhh_prompt(category=None)
        else:
            prompt = get_prompt(
                record=record,
                provide_category=provide_category,
                provide_title=provide_title,
                provide_description=provide_description,
                provide_top_comments=provide_top_comments,
            )

        youtube_caption = str_or_none_if_empty(record.youtube_caption)
        if youtube_caption is None:
            print("Fetching OpenAI GPT caption ...", end="", flush=True)
            print("skipped (YouTube caption must be a non-empty string).")

            # Mutates the Record.
            #record.chatgpt_caption = None

            continue

        youtube_video_id = str_or_none_if_empty(record.video_id)
        if youtube_video_id is None:
            print("Fetching OpenAI GPT caption .", end="", flush=True)
        else:
            print(
                f"{youtube_video_id} Fetching OpenAI GPT caption .",
                end="",
                flush=True,
            )

        if dhh_parameters and model == GPT_3_5_MODEL:
            # DHH GPT 3.5 parameters for OpenAI Completions API:
            # temperature = 0.1 or 0.01 (default is 1.0 ?)
            # top_p = 0.95 or 0.9 (default is 1.0 ?)
            # frequency_penalty = 0 (default is 0 ?)
            # presence_penalty = 0 (default is 0 ?)
            #
            # These GPT 3.5 parameters are from the DHH github repository. It is
            # unclear exactly what combination of temperature and top_p the DHH
            # study used when generating their final GPT captions. Let us
            # conservatively assume that the DHH study used the pair of
            # parameter values that were closest to the API defaults. That means
            # temperature = 0.1 and top_p = 0.95.
            #
            # Beware that OpenAI "generally recommend altering temperature or
            # top_p but not both". See the following links:
            # https://developers.openai.com/api/reference/resources/responses/methods/create
            # https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
            #
            # GPT 3.5 Turbo supports maximum 4096 output tokens.
            # TODO(wathne): Use try/except?
            response = client.responses.create(
                input=youtube_caption,
                instructions=prompt,
                max_output_tokens=4096,
                model=model,
                temperature=0.1,
                top_p=0.95,
            )
        elif model == GPT_3_5_MODEL:
            # GPT 3.5 Turbo supports maximum 4096 output tokens.
            # TODO(wathne): Use try/except?
            response = client.responses.create(
                input=youtube_caption,
                instructions=prompt,
                max_output_tokens=4096,
                model=model,
            )
        else:
            # GPT 5.4 Mini supports maximum 128000 output tokens.
            # GPT 5.4 supports maximum 128000 output tokens.
            # TODO(wathne): Use try/except?
            response = client.responses.create(
                input=youtube_caption,
                instructions=prompt,
                max_output_tokens=128000,
                model=model,
            )

        print(".", end="", flush=True)

        chatgpt_caption = str_or_none_if_empty(response.output_text)

        print(".", end="", flush=True)

        # Mutates the Record.
        record.chatgpt_caption = chatgpt_caption

        print("done.")

        # Waits and hopefully avoids getting banned by OpenAI.
        print(f"    ", end="", flush=True)
        randomized_wait(
            wait_milliseconds_min=wait_milliseconds_min,
            wait_milliseconds_max=wait_milliseconds_max,
            verbose=True,
        )


# TODO(wathne): Test. Remove this.
def test() -> None:
    records: RecordList = RecordList(tsv_file_path="./test.tsv")

    generate_openai_gpt_caption_for_records(
        records=records,
        model=DEFAULT_MODEL,
        dhh_model=False,
        dhh_prompt=False,
        dhh_parameters=True,
        provide_category=False,
        provide_title=False,
        provide_description=False,
        provide_top_comments=False,
        wait_milliseconds_min=3000,
        wait_milliseconds_max=6000,
    )

    records.save(tsv_file_path="./test.tsv")


# TODO(wathne): Test. Remove this.
if __name__ == "__main__":
    test()
