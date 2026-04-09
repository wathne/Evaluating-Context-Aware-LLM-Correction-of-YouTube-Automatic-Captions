# Google T5-Base LLM module (not used).

# For convenience, let "DHH" be a shortened reference to the following study:
# "Empowering the Deaf and Hard of Hearing Community: Enhancing Video Captions
# Using Large Language Models".
# https://arxiv.org/abs/2412.00342
# https://github.com/monikabhole001/Improving-the-Quality-of-Video-Captions-for-the-DHH-Community-Using-LLM


# This prompt is included only for reference. It is an exact stringified copy of
# the T5-Base prompt of the DHH study, as is, from their GitHub repository. Note
# that the DHH study did not use T5-Base to generate their final LLM captions.
_DHH_PROMPT: str = (
    "Correct the following caption as per english standard."
    " Do not give additional information.\n"
    "Caption:\n"
)
