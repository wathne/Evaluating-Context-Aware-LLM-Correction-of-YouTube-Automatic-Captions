# Google Gemini LLM module (testing, not used).

# Module requirements (included in requirements.txt):
# google-genai


from google.genai import Client
from google.genai.types import GenerateContentResponse
from helpers import str_or_none_if_empty
from os import getenv


GEMINI_3_FLASH_MODEL: str = "gemini-3-flash-preview"
GEMINI_3_1_PRO_MODEL: str = "gemini-3.1-pro-preview"
DEFAULT_MODEL: str = GEMINI_3_FLASH_MODEL


def _get_api_key() -> str | None:
    api_key: str | None
    try:
        from private_api_keys import GOOGLE_GEMINI_API_KEY
        api_key = str_or_none_if_empty(GOOGLE_GEMINI_API_KEY)
    except (ImportError, NameError) as instance:
        print(instance)
        api_key = getenv(
            key="GOOGLE_GEMINI_API_KEY",
            default=None,
        )

    return api_key


def test_google_gemini(
    model: str | None = DEFAULT_MODEL,
) -> None:
    api_key: str | None = _get_api_key()
    if api_key is None:
        print("Error: GOOGLE_GEMINI_API_KEY does not exist.")
        return None

    client: Client = Client(
        api_key=api_key,
    )

    if model is None:
        model = str_or_none_if_empty(DEFAULT_MODEL)

    if model is None:
        print("Error: DEFAULT_MODEL does not exist.")
        return None

    # Tests Google Gemini.

    contents: str = "Explain how AI works in a few words"

    response: GenerateContentResponse = client.models.generate_content(
        model=model,
        contents=contents,
        config=None,
    )

    response_text: str | None = response.text
    print(response_text)

    return None
