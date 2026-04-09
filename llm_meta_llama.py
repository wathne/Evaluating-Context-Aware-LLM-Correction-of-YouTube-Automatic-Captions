# Meta Llama-2-13b-chat LLM module (not used).

# For convenience, let "DHH" be a shortened reference to the following study:
# "Empowering the Deaf and Hard of Hearing Community: Enhancing Video Captions
# Using Large Language Models".
# https://arxiv.org/abs/2412.00342
# https://github.com/monikabhole001/Improving-the-Quality-of-Video-Captions-for-the-DHH-Community-Using-LLM


from helpers import str_or_none_if_empty
from os import getenv


# This prompt is included only for reference. It is an exact stringified copy of
# the Llama-2-13b-chat prompt of the DHH study, as is, from their GitHub
# repository.
_DHH_PROMPT: str = (
    "Gramatically correct the text and spellings, don't change word sequence"
    " Correct the text and spellings and give corrected text don't change word"
    " sequence. Text is: "
)


def _get_api_key() -> str | None:
    api_key: str | None
    try:
        from private_api_keys import META_LLAMA_API_KEY
        api_key = str_or_none_if_empty(META_LLAMA_API_KEY)
    except (ImportError, NameError) as instance:
        print(instance)
        api_key = getenv(
            key="META_LLAMA_API_KEY",
            default=None,
        )

    return api_key
