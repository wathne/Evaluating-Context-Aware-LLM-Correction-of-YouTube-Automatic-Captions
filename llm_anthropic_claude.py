# Anthropic Claude LLM module (testing, not used).

# Module requirements (included in requirements.txt):
# anthropic


from anthropic import Anthropic
from anthropic.types import ContentBlock
from anthropic.types import Message
from helpers import str_or_none_if_empty
from os import getenv


CLAUDE_SONNET_4_6_MODEL: str = "claude-sonnet-4-6"
CLAUDE_OPUS_4_6_MODEL: str = "claude-opus-4-6"
DEFAULT_MODEL: str = CLAUDE_OPUS_4_6_MODEL


def _get_api_key() -> str | None:
    api_key: str | None
    try:
        from private_api_keys import ANTHROPIC_CLAUDE_API_KEY
        api_key = str_or_none_if_empty(ANTHROPIC_CLAUDE_API_KEY)
    except (ImportError, NameError) as instance:
        print(instance)
        api_key = getenv(
            key="ANTHROPIC_CLAUDE_API_KEY",
            default=None,
        )

    return api_key


def test_anthropic_claude(
    model: str | None = DEFAULT_MODEL,
) -> None:
    api_key: str | None = _get_api_key()
    if api_key is None:
        print("Error: ANTHROPIC_CLAUDE_API_KEY does not exist.")
        return None

    client: Anthropic = Anthropic(
        api_key=api_key,
    )

    if model is None:
        model = str_or_none_if_empty(DEFAULT_MODEL)

    if model is None:
        print("Error: DEFAULT_MODEL does not exist.")
        return None

    # Tests Anthropic Claude.

    content: str = "Hello, Claude"

    message: Message = client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        model=model,
    )

    message_content: list[ContentBlock] = message.content
    print(message_content)

    return None
