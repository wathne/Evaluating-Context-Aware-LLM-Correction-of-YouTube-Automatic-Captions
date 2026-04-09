# Evaluation module.
#
# See the main function at the end. The main function is meant be modified.
# Running the full main function will cost you about $1 in total credit with the
# OpenAI API for GPT 3.5 and GPT 5.4.
#
# Set your environment variables or populate "./private_api_keys.py" with your
# own private API keys if you want to use LLM functionality (GPT, Llama, Claude,
# and Gemini). YouTube transcripts (closed captions) are fetched without an API
# key, but other video metadata requires a private API key. Modules will try to
# read a wanted API key from your environment variables if not found in
# "./private_api_keys.py".
#
# Complete list of API key environment variables:
# ANTHROPIC_CLAUDE_API_KEY (testing, not used)
# GOOGLE_GEMINI_API_KEY (testing, not used)
# GOOGLE_YOUTUBE_API_KEY
# META_LLAMA_API_KEY (not used)
# OPENAI_GPT_API_KEY

# Module requirements (included in requirements.txt):
# youtube-transcript-api (YouTube transcripts module dependency)
# google-api-python-client (YouTube video metadata module dependency)
# google-auth-oauthlib (YouTube video metadata module dependency)
# google-auth-httplib2 (YouTube video metadata module dependency)
# python-youtube (YouTube video metadata module dependency)
# openai (OpenAI GPT LLM module dependency)
# jiwer (WER)
# absl-py (ROUGE)
# nltk (ROUGE)
# numpy (ROUGE)
# six (ROUGE)
# rouge_score (ROUGE)
# evaluate (BLEU, ROUGE)

# For convenience, let "DHH" be a shortened reference to the following study:
# "Empowering the Deaf and Hard of Hearing Community: Enhancing Video Captions
# Using Large Language Models".
# https://arxiv.org/abs/2412.00342
# https://github.com/monikabhole001/Improving-the-Quality-of-Video-Captions-for-the-DHH-Community-Using-LLM


from captions_with_evaluation_results_wrapper import clear_llm_captions_from_records
from captions_with_evaluation_results_wrapper import clear_results_from_records
from captions_with_evaluation_results_wrapper import initialize_records_from_dhh_records_and_sources
from captions_with_evaluation_results_wrapper import Record
from captions_with_evaluation_results_wrapper import RecordList
from evaluate import EvaluationModule
from evaluate import load as load_evaluation_module
from helpers import float_or_none
from helpers import float_or_zero_if_none
from helpers import int_or_none
from helpers import int_or_zero_if_none
from helpers import str_or_none_if_empty
from jiwer import wer
from math import log10
from yt_metadata import fetch_metadata_for_records
from yt_transcripts import fetch_transcripts_for_records
from llm_openai_gpt import DEFAULT_MODEL
from llm_openai_gpt import GPT_5_4_MODEL
from llm_openai_gpt import generate_openai_gpt_caption_for_records


# Local BLEU metric.
# Pulled from Hugging Face Evaluate repo, 2026 April 9, at commit a7dd338:
# https://github.com/huggingface/evaluate/tree/a7dd338/metrics/bleu
METRIC_BLEU: str = "./metrics/bleu/bleu.py"

# Local ROUGE metric.
# Pulled from Hugging Face Evaluate repo, 2026 April 9, at commit a7dd338:
# https://github.com/huggingface/evaluate/tree/a7dd338/metrics/rouge
METRIC_ROUGE: str = "./metrics/rouge/rouge.py"

DIR: str = "./captions_with_evaluation_results/"
BASE: str = "captions_with_evaluation_results"
EXT: str = ".tsv"

# "./captions_with_evaluation_results/captions_with_evaluation_results.tsv"
DEFAULT_PATH: str = DIR + BASE + EXT

# Path: Old records (DHH), duplicated.
# Combination of DHH records and DHH sources (corrected).
# Fetching: NO, keep old youtube captions (DHH).
# Metadata: NO.
# Prompting: NO, keep old GPT and Llama captions (DHH).
# Evaluation: NO, keep old evaluation results (DHH).
OLD_DUPLICATED_PATH: str = DIR + BASE + "_old_duplicated" + EXT

# Path: Old records (DHH), reevaluated.
# Combination of DHH records and DHH sources (corrected).
# Fetching: NO, keep old youtube captions (DHH).
# Metadata: NO.
# Prompting: NO, keep old GPT and Llama captions (DHH).
# Evaluation: YES, evaluate by metrics WER, BLEU, and ROUGE.
OLD_REEVALUATED_PATH: str = DIR + BASE + "_old_reevaluated" + EXT

# Path: Old records (DHH), reprompted GPT 3.5 (DHH prompt, DHH parameters).
# Combination of DHH records and DHH sources (corrected).
# Fetching: NO, keep old youtube captions (DHH).
# Metadata: NO, no metadata and no category in prompt.
# Prompting: YES, prompt GPT 3.5 with DHH prompt and DHH parameters.
# Evaluation: YES, evaluate by metrics WER, BLEU, and ROUGE.
OLD_REPROMPT_GPT_3_5_DHH_PATH: str = (
    DIR + BASE + "_old_reprompt_gpt_3_5_dhh" + EXT
)

# Path: Old records (DHH), reprompted GPT 3.5 (new prompt, DHH parameters).
# Combination of DHH records and DHH sources (corrected).
# Fetching: NO, keep old youtube captions (DHH).
# Metadata: NO, no metadata and no category in prompt.
# Prompting: YES, prompt GPT 3.5 with new prompt and DHH parameters.
# Evaluation: YES, evaluate by metrics WER, BLEU, and ROUGE.
OLD_REPROMPT_GPT_3_5_NEW_PATH: str = (
    DIR + BASE + "_old_reprompt_gpt_3_5_new" + EXT
)

# Path: New records, with metadata, without LLM captions, without results.
# Based on combination of DHH records and DHH sources (corrected).
# Fetching: YES, fetch new available transcripts and blacklist unavailable.
# Metadata: YES.
# Prompting: NO, all LLM captions are cleared.
# Evaluation: NO, all evaluation results are cleared.
NEW_NO_LLM_PATH: str = DIR + "captions_new_no_llm" + EXT

# Path: New records, GPT 3.5 (new prompt, DHH parameters), without metadata.
# Based on combination of DHH records and DHH sources (corrected).
# Fetching: YES, fetch new available transcripts and blacklist unavailable.
# Metadata: NO, no metadata and no category in prompt.
# Prompting: YES, prompt GPT 3.5 with new prompt and DHH parameters.
# Evaluation: YES, evaluate by metrics WER, BLEU, and ROUGE.
NEW_GPT_3_5_NO_META_PATH: str = DIR + BASE + "_new_gpt_3_5_no_metadata" + EXT

# Path: New records, GPT 3.5 (new prompt, DHH parameters), with metadata.
# Based on combination of DHH records and DHH sources (corrected).
# Fetching: YES, fetch new available transcripts and blacklist unavailable.
# Metadata: YES, all metadata in prompt, no category in prompt.
# Prompting: YES, prompt GPT 3.5 with new prompt and DHH parameters.
# Evaluation: YES, evaluate by metrics WER, BLEU, and ROUGE.
NEW_GPT_3_5_META_PATH: str = DIR + BASE + "_new_gpt_3_5_metadata" + EXT

# Path: New records, GPT 5.4 (new prompt), without metadata.
# Based on combination of DHH records and DHH sources (corrected).
# Fetching: YES, fetch new available transcripts and blacklist unavailable.
# Metadata: NO, no metadata and no category in prompt.
# Prompting: YES, prompt GPT 5.4 with new prompt and default parameters.
# Evaluation: YES, evaluate by metrics WER, BLEU, and ROUGE.
NEW_GPT_5_4_NO_META_PATH: str = DIR + BASE + "_new_gpt_5_4_no_metadata" + EXT

# Path: New records, GPT 5.4 (new prompt), with metadata.
# Based on combination of DHH records and DHH sources (corrected).
# Fetching: YES, fetch new available transcripts and blacklist unavailable.
# Metadata: YES, all metadata in prompt, no category in prompt.
# Prompting: YES, prompt GPT 5.4 with new prompt and default parameters.
# Evaluation: YES, evaluate by metrics WER, BLEU, and ROUGE.
NEW_GPT_5_4_META_PATH: str = DIR + BASE + "_new_gpt_5_4_metadata" + EXT

# Path: New records, GPT 5.4 (new prompt), with metadata without top comments.
# Based on combination of DHH records and DHH sources (corrected).
# Fetching: YES, fetch new available transcripts and blacklist unavailable.
# Metadata: YES, metadata without top comments in prompt, no category in prompt.
# Prompting: YES, prompt GPT 5.4 with new prompt and default parameters.
# Evaluation: YES, evaluate by metrics WER, BLEU, and ROUGE.
NEW_GPT_5_4_META_NO_COMMENTS_PATH: str = (
    DIR + BASE + "_new_gpt_5_4_metadata_no_comments" + EXT
)

# Path: Difference of
# "Old records (DHH), reevaluated" versus
# "Old records (DHH), duplicated".
# See also the DIFF_TOLERANCE constant.
DIFF_OLD_REEVALUATED_VS_OLD_DUPLICATED_PATH: str = (
    DIR + "diff_old_reevaluated_vs_old_duplicated" + EXT
)

# Path: Difference of
# "Old records (DHH), reprompted GPT 3.5 (DHH prompt, DHH parameters)" versus
# "Old records (DHH), reevaluated".
# See also the DIFF_TOLERANCE constant.
DIFF_OLD_REPROMPT_GPT_3_5_DHH_VS_OLD_REEVALUATED_PATH: str = (
    DIR + "diff_old_reprompt_gpt_3_5_dhh_vs_old_reevaluated" + EXT
)

# Path: Difference of
# "Old records (DHH), reprompted GPT 3.5 (new prompt, DHH parameters)" versus
# "Old records (DHH), reevaluated".
# See also the DIFF_TOLERANCE constant.
DIFF_OLD_REPROMPT_GPT_3_5_NEW_VS_OLD_REEVALUATED_PATH: str = (
    DIR + "diff_old_reprompt_gpt_3_5_new_vs_old_reevaluated" + EXT
)

# Path: Difference of
# "Old records (DHH), reprompted GPT 3.5 (new prompt, DHH parameters)" versus
# "Old records (DHH), reprompted GPT 3.5 (DHH prompt, DHH parameters)".
# See also the DIFF_TOLERANCE constant.
DIFF_OLD_REPROMPT_GPT_3_5_NEW_VS_OLD_REPROMPT_GPT_3_5_DHH_PATH: str = (
    DIR + "diff_old_reprompt_gpt_3_5_new_vs_old_reprompt_gpt_3_5_dhh" + EXT
)

# Path: Difference of
# "New records, GPT 3.5 (new prompt, DHH parameters), without metadata" versus
# "Old records (DHH), reprompted GPT 3.5 (new prompt, DHH parameters)".
# See also the DIFF_TOLERANCE constant.
DIFF_NEW_GPT_3_5_NO_META_VS_OLD_REPROMPT_GPT_3_5_NEW_PATH: str = (
    DIR + "diff_new_gpt_3_5_no_metadata_vs_old_reprompt_gpt_3_5_new" + EXT
)

# Path: Difference of
# "New records, GPT 3.5 (new prompt, DHH parameters), with metadata" versus
# "New records, GPT 3.5 (new prompt, DHH parameters), without metadata".
# See also the DIFF_TOLERANCE constant.
DIFF_NEW_GPT_3_5_META_VS_NEW_GPT_3_5_NO_META_PATH: str = (
    DIR + "diff_new_gpt_3_5_metadata_vs_new_gpt_3_5_no_metadata" + EXT
)

# Path: Difference of
# "New records, GPT 5.4 (new prompt), without metadata" versus
# "New records, GPT 3.5 (new prompt, DHH parameters), without metadata".
# See also the DIFF_TOLERANCE constant.
DIFF_NEW_GPT_5_4_NO_META_VS_NEW_GPT_3_5_NO_META_PATH: str = (
    DIR + "diff_new_gpt_5_4_no_metadata_vs_new_gpt_3_5_no_metadata" + EXT
)

# Path: Difference of
# "New records, GPT 5.4 (new prompt), with metadata" versus
# "New records, GPT 3.5 (new prompt, DHH parameters), with metadata".
# See also the DIFF_TOLERANCE constant.
DIFF_NEW_GPT_5_4_META_VS_NEW_GPT_3_5_META_PATH: str = (
    DIR + "diff_new_gpt_5_4_metadata_vs_new_gpt_3_5_metadata" + EXT
)

# Path: Difference of
# "New records, GPT 5.4 (new prompt), with metadata" versus
# "New records, GPT 5.4 (new prompt), without metadata".
# See also the DIFF_TOLERANCE constant.
DIFF_NEW_GPT_5_4_META_VS_NEW_GPT_5_4_NO_META_PATH: str = (
    DIR + "diff_new_gpt_5_4_metadata_vs_new_gpt_5_4_no_metadata" + EXT
)

# Path: Difference of
# "New records, GPT 5.4 (new prompt), with metadata without top comments" versus
# "New records, GPT 5.4 (new prompt), with metadata".
# See also the DIFF_TOLERANCE constant.
DIFF_NEW_GPT_5_4_META_NO_COMMENTS_VS_NEW_GPT_5_4_META_PATH: str = (
    DIR + "diff_new_gpt_5_4_metadata_no_comments_vs_new_gpt_5_4_metadata" + EXT
)

# Path: Difference of
# "New records, GPT 5.4 (new prompt), with metadata without top comments" versus
# "New records, GPT 5.4 (new prompt), without metadata".
# See also the DIFF_TOLERANCE constant.
DIFF_NEW_GPT_5_4_META_NO_COMMENTS_VS_NEW_GPT_5_4_NO_META_PATH: str = (
    DIR +
    "diff_new_gpt_5_4_metadata_no_comments_vs_new_gpt_5_4_no_metadata" +
    EXT
)

# Difference tolerance.
# A difference is taken as zero if the absolute difference is less than the
# specified tolerance. This is to remove noise and floating point artifacts.
DIFF_TOLERANCE: float = 0.001


# Functionally the same as the calculate_wer function of the DHH study.
def dhh_calculate_wer(
    caption: str | list[str],
    ground_truth: str | list[str],
) -> float:
    print("Calculating WER .", end="", flush=True)

    # https://github.com/jitsi/jiwer
    word_error_rate: float = wer(
        reference=ground_truth,
        hypothesis=caption,
        #reference_transform=wer_default,
        #hypothesis_transform=wer_default,
    )

    print(".", end="", flush=True)

    word_error_rate_percentage: float = round(
        number=word_error_rate*100,
        ndigits=2,
    )

    print(".done.")

    return word_error_rate_percentage


# Functionally the same as the calculate_bleu function of the DHH study.
# Refactored, typed, and now much faster with a local BLEU metric.
def dhh_calculate_bleu(
    caption: str | list[str],
    ground_truth: str | list[str],
) -> float | None:
    print("Calculating BLEU .", end="", flush=True)

    # https://github.com/huggingface/evaluate/blob/main/metrics/bleu/bleu.py
    # https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py
    bleu: EvaluationModule = load_evaluation_module(
        path=METRIC_BLEU,
        #config_name=None,
        #module_type="metric",
        #process_id=0,
        #num_process=1,
        #cache_dir="~/.cache/huggingface/evaluate/",
        #experiment_id=None,
        #keep_in_memory=False,
        #download_config=None,
        #download_mode=None,
        #revision=None,
    )

    print(".", end="", flush=True)

    results: dict | None = bleu.compute(
        predictions=[caption],
        references=[ground_truth],
        #tokenizer=Tokenizer13a(), # default
        #max_order=4, # default
        #smooth=False, # default
    )

    print(".", end="", flush=True)

    if results is None:
        return None

    bleu_score: float = results["bleu"]
    #bleu_precisions: list[float] = bleu_results["precisions"]
    #bleu_brevity_penalty: float = bleu_results["brevity_penalty"]
    #bleu_length_ratio: float = bleu_results["length_ratio"]
    #bleu_translation_length: int = bleu_results["translation_length"]
    #bleu_reference_length: int = bleu_results["reference_length"]

    bleu_score_rounded: float = round(
        number=bleu_score,
        ndigits=10,
    )

    print("done.")

    return bleu_score_rounded


# Functionally the same as the calculate_rouge function of the DHH study.
# Refactored, typed, and now much faster with a local ROUGE metric.
def dhh_calculate_rouge(
    caption: str | list[str],
    ground_truth: str | list[str],
) -> tuple[float | None, float | None, float | None, float | None]:
    print("Calculating ROUGE .", end="", flush=True)

    # https://github.com/huggingface/evaluate/blob/main/metrics/rouge/rouge.py
    # https://github.com/google-research/google-research/tree/master/rouge
    rouge: EvaluationModule = load_evaluation_module(
        path=METRIC_ROUGE,
        #config_name=None,
        #module_type="metric",
        #process_id=0,
        #num_process=1,
        #cache_dir="~/.cache/huggingface/evaluate/",
        #experiment_id=None,
        #keep_in_memory=False,
        #download_config=None,
        #download_mode=None,
        #revision=None,
    )

    print(".", end="", flush=True)

    results: dict | None = rouge.compute(
        predictions=[caption],
        references=[ground_truth],
        #rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"], # default
        #use_aggregator=True, # default
        #use_stemmer=False, # default
        #tokenizer=None, # default
    )

    print(".", end="", flush=True)

    if results is None:
        return (None, None, None, None)

    rouge1_score: float = results["rouge1"]
    rouge2_score: float = results["rouge2"]
    rougeL_score: float = results["rougeL"]
    rougeLsum_score: float = results["rougeLsum"]

    rouge1_score_rounded: float = round(
        number=rouge1_score,
        ndigits=10,
    )
    rouge2_score_rounded: float = round(
        number=rouge2_score,
        ndigits=10,
    )
    rougeL_score_rounded: float = round(
        number=rougeL_score,
        ndigits=10,
    )
    rougeLsum_score_rounded: float = round(
        number=rougeLsum_score,
        ndigits=10,
    )

    print("done.")

    return (
        rouge1_score_rounded,
        rouge2_score_rounded,
        rougeL_score_rounded,
        rougeLsum_score_rounded,
    )


def evaluate_records(
    records: RecordList,
) -> None:
    # Clears the results from all mutable Record in the RecordList.
    clear_results_from_records(records=records)

    length: int = len(records)
    i: int
    record: Record # Reference/pointer to a mutable Record.
    for i, record in enumerate(records):
        # Mutates the Record.

        # TODO(wathne): Make this emptiness check more robust. Trim whitespace?
        # An empty ground_truth causes ZeroDivisionError later when scoring the
        # results.
        if record.ground_truth is None or record.ground_truth == "":
            print(f"({i+1}/{length}) {record.video_id} ", end="", flush=True)

            record.wer_gpt = None
            record.bleu_gpt = None
            record.rouge1_gpt = None
            record.rouge2_gpt = None
            record.rougeL_gpt = None
            record.rougeLsum_gpt = None
            record.wer_llama2 = None
            record.bleu_llama2 = None
            record.rouge1_llama2 = None
            record.rouge2_llama2 = None
            record.rougeL_llama2 = None
            record.rougeLsum_llama2 = None
            record.wer_asr = None
            record.bleu_asr = None
            record.rouge1_asr = None
            record.rouge2_asr = None
            record.rougeL_asr = None
            record.rougeLsum_asr = None

            print("Skipped (Ground truth must be a non-empty string).")
            continue

        if record.chatgpt_caption is not None:
            print(
                f"({i+1}/{length}) {record.video_id} GPT ",
                end="",
                flush=True,
            )

            record.wer_gpt = dhh_calculate_wer(
                caption=record.chatgpt_caption,
                ground_truth=record.ground_truth,
            )

            print(
                f"({i+1}/{length}) {record.video_id} GPT ",
                end="",
                flush=True,
            )

            record.bleu_gpt = dhh_calculate_bleu(
                caption=record.chatgpt_caption,
                ground_truth=record.ground_truth,
            )

            print(
                f"({i+1}/{length}) {record.video_id} GPT ",
                end="",
                flush=True,
            )

            (
                record.rouge1_gpt,
                record.rouge2_gpt,
                record.rougeL_gpt,
                record.rougeLsum_gpt,
            ) = dhh_calculate_rouge(
                caption=record.chatgpt_caption,
                ground_truth=record.ground_truth,
            )

        if record.llama2_caption is not None:
            print(
                f"({i+1}/{length}) {record.video_id} Llama2 ",
                end="",
                flush=True,
            )

            record.wer_llama2 = dhh_calculate_wer(
                caption=record.llama2_caption,
                ground_truth=record.ground_truth,
            )

            print(
                f"({i+1}/{length}) {record.video_id} Llama2 ",
                end="",
                flush=True,
            )

            record.bleu_llama2 = dhh_calculate_bleu(
                caption=record.llama2_caption,
                ground_truth=record.ground_truth,
            )

            print(
                f"({i+1}/{length}) {record.video_id} Llama2 ",
                end="",
                flush=True,
            )

            (
                record.rouge1_llama2,
                record.rouge2_llama2,
                record.rougeL_llama2,
                record.rougeLsum_llama2,
            ) = dhh_calculate_rouge(
                caption=record.llama2_caption,
                ground_truth=record.ground_truth,
            )

        if record.youtube_caption is not None:
            print(
                f"({i+1}/{length}) {record.video_id} YouTube ASR ",
                end="",
                flush=True,
            )

            record.wer_asr = dhh_calculate_wer(
                caption=record.youtube_caption,
                ground_truth=record.ground_truth,
            )

            print(
                f"({i+1}/{length}) {record.video_id} YouTube ASR ",
                end="",
                flush=True,
            )

            record.bleu_asr = dhh_calculate_bleu(
                caption=record.youtube_caption,
                ground_truth=record.ground_truth,
            )

            print(
                f"({i+1}/{length}) {record.video_id} YouTube ASR ",
                end="",
                flush=True,
            )

            (
                record.rouge1_asr,
                record.rouge2_asr,
                record.rougeL_asr,
                record.rougeLsum_asr,
            ) = dhh_calculate_rouge(
                caption=record.youtube_caption,
                ground_truth=record.ground_truth,
            )

    return None


def float_diff_or_zero_within_tolerance(
    float_1: float | None, # None = 0 if return_none_if_input_is_none is False.
    float_2: float | None, # None = 0 if return_none_if_input_is_none is False.
    tolerance: float | None = None, # 1e-12 <= t <= 1.0, None = 1e-12.
    ndigits_past_tolerance: int | None = None, # Integer n >= 0, None = 0.
    return_none_if_input_is_none = True,
) -> float | None:
    try:
        return_none_if_input_is_none = bool(return_none_if_input_is_none)
    except (TypeError, ValueError):
        return_none_if_input_is_none = True

    if return_none_if_input_is_none:
        if float_or_none(float_1) is None:
            return None
        if float_or_none(float_2) is None:
            return None

    tolerance = float_or_zero_if_none(tolerance)
    if tolerance < 1e-12:
        tolerance = 1e-12
    if tolerance > 1.0:
        tolerance = 1.0

    difference: float = (
        float_or_zero_if_none(float_1) - 
        float_or_zero_if_none(float_2)
    )

    ndigits_past_tolerance = int_or_zero_if_none(ndigits_past_tolerance)
    if ndigits_past_tolerance < 0:
        ndigits_past_tolerance = 0

    ndigits: int = int(log10(1/tolerance) + ndigits_past_tolerance)
    if ndigits > 10:
        ndigits = 10

    if abs(difference) >= tolerance:
        return round(
            number=difference,
            ndigits=ndigits,
        )

    return 0.0


def diff_results_of_records(
    records_1: RecordList,
    records_2: RecordList,
    records_diff: RecordList, # Reference/pointer to a mutable RecordList.
    tolerance: float | None = None, # 1e-12 <= t <= 1.0, None = 1e-12.
    ndigits_past_tolerance: int | None = None, # Integer n >= 0, None = 0.
    return_none_if_input_is_none = True,
) -> None:
    # Clears the mutable RecordList (records_diff).
    records_diff.clear()

    length: int = len(records_1)
    i: int
    record: Record
    for i in range(0, length, 1):
        record = Record(
            video_id=str_or_none_if_empty(records_1[i].video_id),
            cc_status=int_or_none(records_1[i].cc_status),
            note=str_or_none_if_empty(records_1[i].note),
            category=str_or_none_if_empty(records_1[i].category),
            label=int_or_none(records_1[i].label),
            metadata=None,
            youtube_caption=None,
            ground_truth=None,
            chatgpt_caption=None,
            llama2_caption=None,
            wer_gpt=float_diff_or_zero_within_tolerance(
                float_1=records_1[i].wer_gpt,
                float_2=records_2[i].wer_gpt,
                tolerance=tolerance,
                ndigits_past_tolerance=ndigits_past_tolerance,
                return_none_if_input_is_none=return_none_if_input_is_none,
            ),
            bleu_gpt=float_diff_or_zero_within_tolerance(
                float_1=records_1[i].bleu_gpt,
                float_2=records_2[i].bleu_gpt,
                tolerance=tolerance,
                ndigits_past_tolerance=ndigits_past_tolerance,
                return_none_if_input_is_none=return_none_if_input_is_none,
            ),
            rouge1_gpt=float_diff_or_zero_within_tolerance(
                float_1=records_1[i].rouge1_gpt,
                float_2=records_2[i].rouge1_gpt,
                tolerance=tolerance,
                ndigits_past_tolerance=ndigits_past_tolerance,
                return_none_if_input_is_none=return_none_if_input_is_none,
            ),
            rouge2_gpt=float_diff_or_zero_within_tolerance(
                float_1=records_1[i].rouge2_gpt,
                float_2=records_2[i].rouge2_gpt,
                tolerance=tolerance,
                ndigits_past_tolerance=ndigits_past_tolerance,
                return_none_if_input_is_none=return_none_if_input_is_none,
            ),
            rougeL_gpt=float_diff_or_zero_within_tolerance(
                float_1=records_1[i].rougeL_gpt,
                float_2=records_2[i].rougeL_gpt,
                tolerance=tolerance,
                ndigits_past_tolerance=ndigits_past_tolerance,
                return_none_if_input_is_none=return_none_if_input_is_none,
            ),
            rougeLsum_gpt=float_diff_or_zero_within_tolerance(
                float_1=records_1[i].rougeLsum_gpt,
                float_2=records_2[i].rougeLsum_gpt,
                tolerance=tolerance,
                ndigits_past_tolerance=ndigits_past_tolerance,
                return_none_if_input_is_none=return_none_if_input_is_none,
            ),
            wer_llama2=float_diff_or_zero_within_tolerance(
                float_1=records_1[i].wer_llama2,
                float_2=records_2[i].wer_llama2,
                tolerance=tolerance,
                ndigits_past_tolerance=ndigits_past_tolerance,
                return_none_if_input_is_none=return_none_if_input_is_none,
            ),
            bleu_llama2=float_diff_or_zero_within_tolerance(
                float_1=records_1[i].bleu_llama2,
                float_2=records_2[i].bleu_llama2,
                tolerance=tolerance,
                ndigits_past_tolerance=ndigits_past_tolerance,
                return_none_if_input_is_none=return_none_if_input_is_none,
            ),
            rouge1_llama2=float_diff_or_zero_within_tolerance(
                float_1=records_1[i].rouge1_llama2,
                float_2=records_2[i].rouge1_llama2,
                tolerance=tolerance,
                ndigits_past_tolerance=ndigits_past_tolerance,
                return_none_if_input_is_none=return_none_if_input_is_none,
            ),
            rouge2_llama2=float_diff_or_zero_within_tolerance(
                float_1=records_1[i].rouge2_llama2,
                float_2=records_2[i].rouge2_llama2,
                tolerance=tolerance,
                ndigits_past_tolerance=ndigits_past_tolerance,
                return_none_if_input_is_none=return_none_if_input_is_none,
            ),
            rougeL_llama2=float_diff_or_zero_within_tolerance(
                float_1=records_1[i].rougeL_llama2,
                float_2=records_2[i].rougeL_llama2,
                tolerance=tolerance,
                ndigits_past_tolerance=ndigits_past_tolerance,
                return_none_if_input_is_none=return_none_if_input_is_none,
            ),
            rougeLsum_llama2=float_diff_or_zero_within_tolerance(
                float_1=records_1[i].rougeLsum_llama2,
                float_2=records_2[i].rougeLsum_llama2,
                tolerance=tolerance,
                ndigits_past_tolerance=ndigits_past_tolerance,
                return_none_if_input_is_none=return_none_if_input_is_none,
            ),
            wer_asr=float_diff_or_zero_within_tolerance(
                float_1=records_1[i].wer_asr,
                float_2=records_2[i].wer_asr,
                tolerance=tolerance,
                ndigits_past_tolerance=ndigits_past_tolerance,
                return_none_if_input_is_none=return_none_if_input_is_none,
            ),
            bleu_asr=float_diff_or_zero_within_tolerance(
                float_1=records_1[i].bleu_asr,
                float_2=records_2[i].bleu_asr,
                tolerance=tolerance,
                ndigits_past_tolerance=ndigits_past_tolerance,
                return_none_if_input_is_none=return_none_if_input_is_none,
            ),
            rouge1_asr=float_diff_or_zero_within_tolerance(
                float_1=records_1[i].rouge1_asr,
                float_2=records_2[i].rouge1_asr,
                tolerance=tolerance,
                ndigits_past_tolerance=ndigits_past_tolerance,
                return_none_if_input_is_none=return_none_if_input_is_none,
            ),
            rouge2_asr=float_diff_or_zero_within_tolerance(
                float_1=records_1[i].rouge2_asr,
                float_2=records_2[i].rouge2_asr,
                tolerance=tolerance,
                ndigits_past_tolerance=ndigits_past_tolerance,
                return_none_if_input_is_none=return_none_if_input_is_none,
            ),
            rougeL_asr=float_diff_or_zero_within_tolerance(
                float_1=records_1[i].rougeL_asr,
                float_2=records_2[i].rougeL_asr,
                tolerance=tolerance,
                ndigits_past_tolerance=ndigits_past_tolerance,
                return_none_if_input_is_none=return_none_if_input_is_none,
            ),
            rougeLsum_asr=float_diff_or_zero_within_tolerance(
                float_1=records_1[i].rougeLsum_asr,
                float_2=records_2[i].rougeLsum_asr,
                tolerance=tolerance,
                ndigits_past_tolerance=ndigits_past_tolerance,
                return_none_if_input_is_none=return_none_if_input_is_none,
            ),
        )

        # Mutates the RecordList (records_diff).
        records_diff.append(record)

    return None


def main() -> None:
    # This main function is meant be modified:

    ############################################################################

    print("Old records (DHH), duplicated:")

    # Old records (DHH), duplicated.
    # Combination of DHH records and DHH sources (corrected).
    # Fetching: NO, keep old youtube captions (DHH).
    # Metadata: NO.
    # Prompting: NO, keep old GPT and Llama captions (DHH).
    # Evaluation: NO, keep old evaluation results (DHH).
    records_old_duplicated: RecordList = RecordList(
        tsv_file_path=None,
    )

    # Loads DHH records, video ID, and extra data from the combination of DHH
    # records and DHH sources (corrected).
    initialize_records_from_dhh_records_and_sources(
        records=records_old_duplicated,
    )

    # Saves a copy of the records.
    records_old_duplicated.save(
        tsv_file_path=OLD_DUPLICATED_PATH,
        skip_asr_results=True,
    )

    ############################################################################

    print("Old records (DHH), reevaluated:")

    # Old records (DHH), reevaluated.
    # Combination of DHH records and DHH sources (corrected).
    # Fetching: NO, keep old youtube captions (DHH).
    # Metadata: NO.
    # Prompting: NO, keep old GPT and Llama captions (DHH).
    # Evaluation: YES, evaluate by metrics WER, BLEU, and ROUGE.
    records_old_reevaluated: RecordList = RecordList(
        tsv_file_path=None,
    )

    # Loads the duplicated old records that were saved earlier.
    records_old_reevaluated.load(
        tsv_file_path=OLD_DUPLICATED_PATH,
        clear=True,
    )

    # Evaluates the records (and clears existing results).
    evaluate_records(
        records=records_old_reevaluated,
    )

    # Saves a copy of the records.
    records_old_reevaluated.save(
        tsv_file_path=OLD_REEVALUATED_PATH,
        skip_asr_results=False,
    )

    ############################################################################

    print("Old records (DHH), reprompted GPT 3.5 (DHH prompt, DHH parameters):")

    # Old records (DHH), reprompted GPT 3.5 (DHH prompt, DHH parameters).
    # Combination of DHH records and DHH sources (corrected).
    # Fetching: NO, keep old youtube captions (DHH).
    # Metadata: NO, no metadata and no category in prompt.
    # Prompting: YES, prompt GPT 3.5 with DHH prompt and DHH parameters.
    # Evaluation: YES, evaluate by metrics WER, BLEU, and ROUGE.
    records_old_reprompt_gpt_3_5_dhh: RecordList = RecordList(
        tsv_file_path=None,
    )

    # Loads the duplicated old records that were saved earlier.
    records_old_reprompt_gpt_3_5_dhh.load(
        tsv_file_path=OLD_DUPLICATED_PATH,
        clear=True,
    )

    # Clears old LLM captions.
    clear_llm_captions_from_records(
        records=records_old_reprompt_gpt_3_5_dhh,
    )

    # Generates new GPT captions.
    # Prompts GPT 3.5 with DHH prompt and DHH parameters.
    # Provides no metadata and no category in prompt.
    generate_openai_gpt_caption_for_records(
        records=records_old_reprompt_gpt_3_5_dhh,
        model=DEFAULT_MODEL,
        dhh_model=True, # True: force the use of DHH_MODEL (gpt-3.5-turbo-0125).
        dhh_prompt=True, # True: use DHH prompt (get_dhh_prompt function).
        dhh_parameters=True, # True: use temperature = 0.1 and top_p = 0.95.
        provide_category=False,
        provide_title=False,
        provide_description=False,
        provide_top_comments=False,
        wait_milliseconds_min=3000, # 3 seconds, precaution to avoid ban.
        wait_milliseconds_max=6000, # 6 seconds, precaution to avoid ban.
    )

    # Evaluates the records (and clears existing results).
    evaluate_records(
        records=records_old_reprompt_gpt_3_5_dhh,
    )

    # Saves a copy of the records.
    records_old_reprompt_gpt_3_5_dhh.save(
        tsv_file_path=OLD_REPROMPT_GPT_3_5_DHH_PATH,
        skip_asr_results=False,
    )

    ############################################################################

    print("Old records (DHH), reprompted GPT 3.5 (new prompt, DHH parameters):")

    # Old records (DHH), reprompted GPT 3.5 (new prompt, DHH parameters).
    # Combination of DHH records and DHH sources (corrected).
    # Fetching: NO, keep old youtube captions (DHH).
    # Metadata: NO, no metadata and no category in prompt.
    # Prompting: YES, prompt GPT 3.5 with new prompt and DHH parameters.
    # Evaluation: YES, evaluate by metrics WER, BLEU, and ROUGE.
    records_old_reprompt_gpt_3_5_new: RecordList = RecordList(
        tsv_file_path=None,
    )

    # Loads the duplicated old records that were saved earlier.
    records_old_reprompt_gpt_3_5_new.load(
        tsv_file_path=OLD_DUPLICATED_PATH,
        clear=True,
    )

    # Clears old LLM captions.
    clear_llm_captions_from_records(
        records=records_old_reprompt_gpt_3_5_new,
    )

    # Generates new GPT captions.
    # Prompts GPT 3.5 with new prompt and DHH parameters.
    # Provides no metadata and no category in prompt.
    generate_openai_gpt_caption_for_records(
        records=records_old_reprompt_gpt_3_5_new,
        model=DEFAULT_MODEL,
        dhh_model=True, # True: force the use of DHH_MODEL (gpt-3.5-turbo-0125).
        dhh_prompt=False, # False: use new prompt (get_prompt function).
        dhh_parameters=True, # True: use temperature = 0.1 and top_p = 0.95.
        provide_category=False,
        provide_title=False,
        provide_description=False,
        provide_top_comments=False,
        wait_milliseconds_min=3000, # 3 seconds, precaution to avoid ban.
        wait_milliseconds_max=6000, # 6 seconds, precaution to avoid ban.
    )

    # Evaluates the records (and clears existing results).
    evaluate_records(
        records=records_old_reprompt_gpt_3_5_new,
    )

    # Saves a copy of the records.
    records_old_reprompt_gpt_3_5_new.save(
        tsv_file_path=OLD_REPROMPT_GPT_3_5_NEW_PATH,
        skip_asr_results=False,
    )

    ############################################################################

    print("New records, with metadata, without LLM captions, without results:")

    # New records, with metadata, without LLM captions, without results.
    # Based on combination of DHH records and DHH sources (corrected).
    # Fetching: YES, fetch new available transcripts and blacklist unavailable.
    # Metadata: YES.
    # Prompting: NO, all LLM captions are cleared.
    # Evaluation: NO, all evaluation results are cleared.
    captions_new_no_llm: RecordList = RecordList(
        tsv_file_path=None,
    )

    # TODO: uncomment/comment:
    # Bases new records on the duplicated old records that were saved earlier.
    #captions_new_no_llm.load(
    #    tsv_file_path=OLD_DUPLICATED_PATH,
    #    clear=True,
    #)

    # TODO: comment/uncomment:
    # Loads existing new records (existing new transcripts), to avoid getting
    # banned for fetching all new transcripts again.
    captions_new_no_llm: RecordList = RecordList(
        tsv_file_path=NEW_NO_LLM_PATH,
    )

    # Clears old results.
    clear_results_from_records(
        records=captions_new_no_llm,
    )

    # Clears old LLM captions.
    clear_llm_captions_from_records(
        records=captions_new_no_llm,
    )

    # Fetches new autogen transcripts (autogen closed captions).
    # Old transcripts are retained until replaced with a new autogen transcript.
    # Skips videos that have CC status 200, meaning that a new autogen
    # transcript has already been saved for that video.
    # Getting IP banned (IpBlocked) does happen, at least when the
    # randomized_wait() function is not set to wait long enough.
    # Returns early if you get banned, to allow saving fetched transcripts
    # instead of crashing and losing all unsaved progress.
    fetch_transcripts_for_records(
        records=captions_new_no_llm,
        clear_unavailable_transcripts=True,
        wait_for_input_to_continue=False,
        wait_milliseconds_min=120000, # 120 seconds, precaution to avoid ban.
        wait_milliseconds_max=240000, # 240 seconds, precaution to avoid ban.
    )

    # Saves a copy of the records (new autogen transcripts).
    captions_new_no_llm.save(
        tsv_file_path=NEW_NO_LLM_PATH,
        skip_asr_results=False,
    )

    # TODO: uncomment/comment:
    # Fetches video metadata: title, description, and top comments (relevance).
    # Overwrites existing metadata, if any, with newly fetched metadata.
    #fetch_metadata_for_records(
    #    records=captions_new_no_llm,
    #    wait_milliseconds_min=6000,
    #    wait_milliseconds_max=12000,
    #)

    # Saves a copy of the records (new autogen transcripts and metadata).
    captions_new_no_llm.save(
        tsv_file_path=NEW_NO_LLM_PATH,
        skip_asr_results=False,
    )

    ############################################################################

    print("New records, GPT 3.5 (new prompt, DHH parameters), without"
          " metadata:")

    # New records, GPT 3.5 (new prompt, DHH parameters), without metadata.
    # Based on combination of DHH records and DHH sources (corrected).
    # Fetching: YES, fetch new available transcripts and blacklist unavailable.
    # Metadata: NO, no metadata and no category in prompt.
    # Prompting: YES, prompt GPT 3.5 with new prompt and DHH parameters.
    # Evaluation: YES, evaluate by metrics WER, BLEU, and ROUGE.
    records_new_gpt_3_5_no_meta: RecordList = RecordList(
        tsv_file_path=None,
    )

    # Loads the new records that were saved earlier (new autogen transcripts and
    # metadata).
    records_new_gpt_3_5_no_meta.load(
        tsv_file_path=NEW_NO_LLM_PATH,
        clear=True,
    )

    # Clears old LLM captions, just in case they are not cleared.
    clear_llm_captions_from_records(
        records=records_new_gpt_3_5_no_meta,
    )

    # Generates new GPT captions.
    # Prompts GPT 3.5 with new prompt and DHH parameters.
    # Provides no metadata and no category in prompt.
    generate_openai_gpt_caption_for_records(
        records=records_new_gpt_3_5_no_meta,
        model=DEFAULT_MODEL,
        dhh_model=True, # True: force the use of DHH_MODEL (gpt-3.5-turbo-0125).
        dhh_prompt=False, # False: use new prompt (get_prompt function).
        dhh_parameters=True, # True: use temperature = 0.1 and top_p = 0.95.
        provide_category=False,
        provide_title=False,
        provide_description=False,
        provide_top_comments=False,
        wait_milliseconds_min=3000, # 3 seconds, precaution to avoid ban.
        wait_milliseconds_max=6000, # 6 seconds, precaution to avoid ban.
    )

    # Evaluates the records (and clears existing results).
    evaluate_records(
        records=records_new_gpt_3_5_no_meta,
    )

    # Saves a copy of the records.
    records_new_gpt_3_5_no_meta.save(
        tsv_file_path=NEW_GPT_3_5_NO_META_PATH,
        skip_asr_results=False,
    )

    ############################################################################

    print("New records, GPT 3.5 (new prompt, DHH parameters), with metadata:")

    # New records, GPT 3.5 (new prompt, DHH parameters), with metadata.
    # Based on combination of DHH records and DHH sources (corrected).
    # Fetching: YES, fetch new available transcripts and blacklist unavailable.
    # Metadata: YES, all metadata in prompt, no category in prompt.
    # Prompting: YES, prompt GPT 3.5 with new prompt and DHH parameters.
    # Evaluation: YES, evaluate by metrics WER, BLEU, and ROUGE.
    records_new_gpt_3_5_meta: RecordList = RecordList(
        tsv_file_path=None,
    )

    # Loads the new records that were saved earlier (new autogen transcripts and
    # metadata).
    records_new_gpt_3_5_meta.load(
        tsv_file_path=NEW_NO_LLM_PATH,
        clear=True,
    )

    # Clears old LLM captions, just in case they are not cleared.
    clear_llm_captions_from_records(
        records=records_new_gpt_3_5_meta,
    )

    # Generates new GPT captions.
    # Prompts GPT 3.5 with new prompt and DHH parameters.
    # Provides all metadata in prompt, no category in prompt.
    generate_openai_gpt_caption_for_records(
        records=records_new_gpt_3_5_meta,
        model=DEFAULT_MODEL,
        dhh_model=True, # True: force the use of DHH_MODEL (gpt-3.5-turbo-0125).
        dhh_prompt=False, # False: use new prompt (get_prompt function).
        dhh_parameters=True, # True: use temperature = 0.1 and top_p = 0.95.
        provide_category=False,
        provide_title=True,
        provide_description=True,
        provide_top_comments=True,
        wait_milliseconds_min=3000, # 3 seconds, precaution to avoid ban.
        wait_milliseconds_max=6000, # 6 seconds, precaution to avoid ban.
    )

    # Evaluates the records (and clears existing results).
    evaluate_records(
        records=records_new_gpt_3_5_meta,
    )

    # Saves a copy of the records.
    records_new_gpt_3_5_meta.save(
        tsv_file_path=NEW_GPT_3_5_META_PATH,
        skip_asr_results=False,
    )

    ############################################################################

    print("New records, GPT 5.4 (new prompt), without metadata:")

    # New records, GPT 5.4 (new prompt), without metadata.
    # Based on combination of DHH records and DHH sources (corrected).
    # Fetching: YES, fetch new available transcripts and blacklist unavailable.
    # Metadata: NO, no metadata and no category in prompt.
    # Prompting: YES, prompt GPT 5.4 with new prompt and default parameters.
    # Evaluation: YES, evaluate by metrics WER, BLEU, and ROUGE.
    records_new_gpt_5_4_no_meta: RecordList = RecordList(
        tsv_file_path=None,
    )

    # Loads the new records that were saved earlier (new autogen transcripts and
    # metadata).
    records_new_gpt_5_4_no_meta.load(
        tsv_file_path=NEW_NO_LLM_PATH,
        clear=True,
    )

    # Clears old LLM captions, just in case they are not cleared.
    clear_llm_captions_from_records(
        records=records_new_gpt_5_4_no_meta,
    )

    # Generates new GPT captions.
    # Prompts GPT 5.4 with new prompt and default parameters.
    # Provides no metadata and no category in prompt.
    generate_openai_gpt_caption_for_records(
        records=records_new_gpt_5_4_no_meta,
        model=GPT_5_4_MODEL,
        dhh_model=False,
        dhh_prompt=False, # False: use new prompt (get_prompt function).
        dhh_parameters=False, # Only applicable to GPT 3.5 (gpt-3.5-turbo-0125).
        provide_category=False,
        provide_title=False,
        provide_description=False,
        provide_top_comments=False,
        wait_milliseconds_min=3000, # 3 seconds, precaution to avoid ban.
        wait_milliseconds_max=6000, # 6 seconds, precaution to avoid ban.
    )

    # Evaluates the records (and clears existing results).
    evaluate_records(
        records=records_new_gpt_5_4_no_meta,
    )

    # Saves a copy of the records.
    records_new_gpt_5_4_no_meta.save(
        tsv_file_path=NEW_GPT_5_4_NO_META_PATH,
        skip_asr_results=False,
    )

    ############################################################################

    print("New records, GPT 5.4 (new prompt), with metadata:")

    # New records, GPT 5.4 (new prompt), with metadata.
    # Based on combination of DHH records and DHH sources (corrected).
    # Fetching: YES, fetch new available transcripts and blacklist unavailable.
    # Metadata: YES, all metadata in prompt, no category in prompt.
    # Prompting: YES, prompt GPT 5.4 with new prompt and default parameters.
    # Evaluation: YES, evaluate by metrics WER, BLEU, and ROUGE.
    records_new_gpt_5_4_meta: RecordList = RecordList(
        tsv_file_path=None,
    )

    # Loads the new records that were saved earlier (new autogen transcripts and
    # metadata).
    records_new_gpt_5_4_meta.load(
        tsv_file_path=NEW_NO_LLM_PATH,
        clear=True,
    )

    # Clears old LLM captions, just in case they are not cleared.
    clear_llm_captions_from_records(
        records=records_new_gpt_5_4_meta,
    )

    # Generates new GPT captions.
    # Prompts GPT 5.4 with new prompt and default parameters.
    # Provides all metadata in prompt, no category in prompt.
    generate_openai_gpt_caption_for_records(
        records=records_new_gpt_5_4_meta,
        model=GPT_5_4_MODEL,
        dhh_model=False,
        dhh_prompt=False, # False: use new prompt (get_prompt function).
        dhh_parameters=False, # Only applicable to GPT 3.5 (gpt-3.5-turbo-0125).
        provide_category=False,
        provide_title=True,
        provide_description=True,
        provide_top_comments=True,
        wait_milliseconds_min=3000, # 3 seconds, precaution to avoid ban.
        wait_milliseconds_max=6000, # 6 seconds, precaution to avoid ban.
    )

    # Evaluates the records (and clears existing results).
    evaluate_records(
        records=records_new_gpt_5_4_meta,
    )

    # Saves a copy of the records.
    records_new_gpt_5_4_meta.save(
        tsv_file_path=NEW_GPT_5_4_META_PATH,
        skip_asr_results=False,
    )

    ############################################################################

    print("New records, GPT 5.4 (new prompt), with metadata without top"
          " comments:")

    # New records, GPT 5.4 (new prompt), with metadata without top comments.
    # Based on combination of DHH records and DHH sources (corrected).
    # Fetching: YES, fetch new available transcripts and blacklist unavailable.
    # Metadata: YES, metadata w/o top comments in prompt, no category in prompt.
    # Prompting: YES, prompt GPT 5.4 with new prompt and default parameters.
    # Evaluation: YES, evaluate by metrics WER, BLEU, and ROUGE.
    records_new_gpt_5_4_meta_no_comments: RecordList = RecordList(
        tsv_file_path=None,
    )

    # Loads the new records that were saved earlier (new autogen transcripts and
    # metadata).
    records_new_gpt_5_4_meta_no_comments.load(
        tsv_file_path=NEW_NO_LLM_PATH,
        clear=True,
    )

    # Clears old LLM captions, just in case they are not cleared.
    clear_llm_captions_from_records(
        records=records_new_gpt_5_4_meta_no_comments,
    )

    # Generates new GPT captions.
    # Prompts GPT 5.4 with new prompt and default parameters.
    # Provides metadata without top comments in prompt, no category in prompt.
    generate_openai_gpt_caption_for_records(
        records=records_new_gpt_5_4_meta_no_comments,
        model=GPT_5_4_MODEL,
        dhh_model=False,
        dhh_prompt=False, # False: use new prompt (get_prompt function).
        dhh_parameters=False, # Only applicable to GPT 3.5 (gpt-3.5-turbo-0125).
        provide_category=False,
        provide_title=True,
        provide_description=True,
        provide_top_comments=False,
        wait_milliseconds_min=3000, # 3 seconds, precaution to avoid ban.
        wait_milliseconds_max=6000, # 6 seconds, precaution to avoid ban.
    )

    # Evaluates the records (and clears existing results).
    evaluate_records(
        records=records_new_gpt_5_4_meta_no_comments,
    )

    # Saves a copy of the records.
    records_new_gpt_5_4_meta_no_comments.save(
        tsv_file_path=NEW_GPT_5_4_META_NO_COMMENTS_PATH,
        skip_asr_results=False,
    )

    ############################################################################

    print("Difference of")
    print('"Old records (DHH), reevaluated" versus')
    print('"Old records (DHH), duplicated":')

    # Difference of
    # "Old records (DHH), reevaluated" versus
    # "Old records (DHH), duplicated".
    diff_old_reevaluated_vs_old_duplicated: RecordList = RecordList(
        tsv_file_path=None,
    )

    # Old records (DHH), reevaluated.
    if "records_old_reevaluated" not in locals():
        records_old_reevaluated: RecordList = RecordList(
            tsv_file_path=OLD_REEVALUATED_PATH,
        )

    # Old records (DHH), duplicated.
    if "records_old_duplicated" not in locals():
        records_old_duplicated: RecordList = RecordList(
            tsv_file_path=OLD_DUPLICATED_PATH,
        )

    # Calculates the differences of the results of the records. A difference is
    # taken as zero if the absolute difference is less than the specified
    # tolerance. This is to remove noise and floating point artifacts.
    diff_results_of_records(
        records_1=records_old_reevaluated,
        records_2=records_old_duplicated,
        records_diff=diff_old_reevaluated_vs_old_duplicated,
        tolerance=DIFF_TOLERANCE,
        ndigits_past_tolerance=None,
        return_none_if_input_is_none=True,
    )

    # Saves a copy of the differences that were just calculated.
    diff_old_reevaluated_vs_old_duplicated.save(
        tsv_file_path=DIFF_OLD_REEVALUATED_VS_OLD_DUPLICATED_PATH,
        skip_asr_results=False,
    )

    ############################################################################

    print("Difference of")
    print('"Old records (DHH), reprompted GPT 3.5 (DHH prompt, DHH'
          ' parameters)" versus')
    print('"Old records (DHH), reevaluated":')

    # Difference of
    # "Old records (DHH), reprompted GPT 3.5 (DHH prompt, DHH parameters)" vs.
    # "Old records (DHH), reevaluated".
    diff_old_reprompt_gpt_3_5_dhh_vs_old_reevaluated: RecordList = RecordList(
        tsv_file_path=None,
    )

    # Old records (DHH), reprompted GPT 3.5 (DHH prompt, DHH parameters).
    if "records_old_reprompt_gpt_3_5_dhh" not in locals():
        records_old_reprompt_gpt_3_5_dhh: RecordList = RecordList(
            tsv_file_path=OLD_REPROMPT_GPT_3_5_DHH_PATH,
        )

    # Old records (DHH), reevaluated.
    if "records_old_reevaluated" not in locals():
        records_old_reevaluated: RecordList = RecordList(
            tsv_file_path=OLD_REEVALUATED_PATH,
        )

    # Calculates the differences of the results of the records. A difference is
    # taken as zero if the absolute difference is less than the specified
    # tolerance. This is to remove noise and floating point artifacts.
    diff_results_of_records(
        records_1=records_old_reprompt_gpt_3_5_dhh,
        records_2=records_old_reevaluated,
        records_diff=diff_old_reprompt_gpt_3_5_dhh_vs_old_reevaluated,
        tolerance=DIFF_TOLERANCE,
        ndigits_past_tolerance=None,
        return_none_if_input_is_none=True,
    )

    # Saves a copy of the differences that were just calculated.
    diff_old_reprompt_gpt_3_5_dhh_vs_old_reevaluated.save(
        tsv_file_path=DIFF_OLD_REPROMPT_GPT_3_5_DHH_VS_OLD_REEVALUATED_PATH,
        skip_asr_results=False,
    )

    ############################################################################

    print("Difference of")
    print('"Old records (DHH), reprompted GPT 3.5 (new prompt, DHH'
          ' parameters)" versus')
    print('"Old records (DHH), reevaluated":')

    # Difference of
    # "Old records (DHH), reprompted GPT 3.5 (new prompt, DHH parameters)" vs.
    # "Old records (DHH), reevaluated".
    diff_old_reprompt_gpt_3_5_new_vs_old_reevaluated: RecordList = RecordList(
        tsv_file_path=None,
    )

    # Old records (DHH), reprompted GPT 3.5 (new prompt, DHH parameters).
    if "records_old_reprompt_gpt_3_5_new" not in locals():
        records_old_reprompt_gpt_3_5_new: RecordList = RecordList(
            tsv_file_path=OLD_REPROMPT_GPT_3_5_NEW_PATH,
        )

    # Old records (DHH), reevaluated.
    if "records_old_reevaluated" not in locals():
        records_old_reevaluated: RecordList = RecordList(
            tsv_file_path=OLD_REEVALUATED_PATH,
        )

    # Calculates the differences of the results of the records. A difference is
    # taken as zero if the absolute difference is less than the specified
    # tolerance. This is to remove noise and floating point artifacts.
    diff_results_of_records(
        records_1=records_old_reprompt_gpt_3_5_new,
        records_2=records_old_reevaluated,
        records_diff=diff_old_reprompt_gpt_3_5_new_vs_old_reevaluated,
        tolerance=DIFF_TOLERANCE,
        ndigits_past_tolerance=None,
        return_none_if_input_is_none=True,
    )

    # Saves a copy of the differences that were just calculated.
    diff_old_reprompt_gpt_3_5_new_vs_old_reevaluated.save(
        tsv_file_path=DIFF_OLD_REPROMPT_GPT_3_5_NEW_VS_OLD_REEVALUATED_PATH,
        skip_asr_results=False,
    )

    ############################################################################

    print("Difference of")
    print('"Old records (DHH), reprompted GPT 3.5 (new prompt, DHH'
          ' parameters)" versus')
    print('"Old records (DHH), reprompted GPT 3.5 (DHH prompt, DHH'
          ' parameters)":')

    # Difference of
    # "Old records (DHH), reprompted GPT 3.5 (new prompt, DHH parameters)" vs.
    # "Old records (DHH), reprompted GPT 3.5 (DHH prompt, DHH parameters)".
    diff_old_reprompt_gpt_3_5_new_vs_old_reprompt_gpt_3_5_dhh: RecordList
    diff_old_reprompt_gpt_3_5_new_vs_old_reprompt_gpt_3_5_dhh = RecordList(
        tsv_file_path=None,
    )

    # Old records (DHH), reprompted GPT 3.5 (new prompt, DHH parameters).
    if "records_old_reprompt_gpt_3_5_new" not in locals():
        records_old_reprompt_gpt_3_5_new: RecordList = RecordList(
            tsv_file_path=OLD_REPROMPT_GPT_3_5_NEW_PATH,
        )

    # Old records (DHH), reprompted GPT 3.5 (DHH prompt, DHH parameters).
    if "records_old_reprompt_gpt_3_5_dhh" not in locals():
        records_old_reprompt_gpt_3_5_dhh: RecordList = RecordList(
            tsv_file_path=OLD_REPROMPT_GPT_3_5_DHH_PATH,
        )

    # Calculates the differences of the results of the records. A difference is
    # taken as zero if the absolute difference is less than the specified
    # tolerance. This is to remove noise and floating point artifacts.
    diff_results_of_records(
        records_1=records_old_reprompt_gpt_3_5_new,
        records_2=records_old_reprompt_gpt_3_5_dhh,
        records_diff=diff_old_reprompt_gpt_3_5_new_vs_old_reprompt_gpt_3_5_dhh,
        tolerance=DIFF_TOLERANCE,
        ndigits_past_tolerance=None,
        return_none_if_input_is_none=True,
    )

    # Saves a copy of the differences that were just calculated.
    diff_old_reprompt_gpt_3_5_new_vs_old_reprompt_gpt_3_5_dhh.save(
        tsv_file_path=(
            DIFF_OLD_REPROMPT_GPT_3_5_NEW_VS_OLD_REPROMPT_GPT_3_5_DHH_PATH
        ),
        skip_asr_results=False,
    )

    ############################################################################

    print("Difference of")
    print('"New records, GPT 3.5 (new prompt, DHH parameters), without'
          ' metadata" versus')
    print('"Old records (DHH), reprompted GPT 3.5 (new prompt, DHH'
          ' parameters)":')

    # Difference of
    # "New records, GPT 3.5 (new prompt, DHH parameters), without metadata" vs.
    # "Old records (DHH), reprompted GPT 3.5 (new prompt, DHH parameters)".
    diff_new_gpt_3_5_no_meta_vs_old_reprompt_gpt_3_5_new: RecordList
    diff_new_gpt_3_5_no_meta_vs_old_reprompt_gpt_3_5_new = RecordList(
        tsv_file_path=None,
    )

    # New records, GPT 3.5 (new prompt, DHH parameters), without metadata.
    if "records_new_gpt_3_5_no_meta" not in locals():
        records_new_gpt_3_5_no_meta: RecordList = RecordList(
            tsv_file_path=NEW_GPT_3_5_NO_META_PATH,
        )

    # Old records (DHH), reprompted GPT 3.5 (new prompt, DHH parameters).
    if "records_old_reprompt_gpt_3_5_new" not in locals():
        records_old_reprompt_gpt_3_5_new: RecordList = RecordList(
            tsv_file_path=OLD_REPROMPT_GPT_3_5_NEW_PATH,
        )

    # Calculates the differences of the results of the records. A difference is
    # taken as zero if the absolute difference is less than the specified
    # tolerance. This is to remove noise and floating point artifacts.
    diff_results_of_records(
        records_1=records_new_gpt_3_5_no_meta,
        records_2=records_old_reprompt_gpt_3_5_new,
        records_diff=diff_new_gpt_3_5_no_meta_vs_old_reprompt_gpt_3_5_new,
        tolerance=DIFF_TOLERANCE,
        ndigits_past_tolerance=None,
        return_none_if_input_is_none=True,
    )

    # Saves a copy of the differences that were just calculated.
    diff_new_gpt_3_5_no_meta_vs_old_reprompt_gpt_3_5_new.save(
        tsv_file_path=DIFF_NEW_GPT_3_5_NO_META_VS_OLD_REPROMPT_GPT_3_5_NEW_PATH,
        skip_asr_results=False,
    )

    ############################################################################

    print("Difference of")
    print('"New records, GPT 3.5 (new prompt, DHH parameters), with metadata"'
          ' versus')
    print('"New records, GPT 3.5 (new prompt, DHH parameters), without'
          ' metadata":')

    # Difference of
    # "New records, GPT 3.5 (new prompt, DHH parameters), with metadata" versus
    # "New records, GPT 3.5 (new prompt, DHH parameters), without metadata".
    diff_new_gpt_3_5_meta_vs_new_gpt_3_5_no_meta: RecordList = RecordList(
        tsv_file_path=None,
    )

    # New records, GPT 3.5 (new prompt, DHH parameters), with metadata.
    if "records_new_gpt_3_5_meta" not in locals():
        records_new_gpt_3_5_meta: RecordList = RecordList(
            tsv_file_path=NEW_GPT_3_5_META_PATH,
        )

    # New records, GPT 3.5 (new prompt, DHH parameters), without metadata.
    if "records_new_gpt_3_5_no_meta" not in locals():
        records_new_gpt_3_5_no_meta: RecordList = RecordList(
            tsv_file_path=NEW_GPT_3_5_NO_META_PATH,
        )

    # Calculates the differences of the results of the records. A difference is
    # taken as zero if the absolute difference is less than the specified
    # tolerance. This is to remove noise and floating point artifacts.
    diff_results_of_records(
        records_1=records_new_gpt_3_5_meta,
        records_2=records_new_gpt_3_5_no_meta,
        records_diff=diff_new_gpt_3_5_meta_vs_new_gpt_3_5_no_meta,
        tolerance=DIFF_TOLERANCE,
        ndigits_past_tolerance=None,
        return_none_if_input_is_none=True,
    )

    # Saves a copy of the differences that were just calculated.
    diff_new_gpt_3_5_meta_vs_new_gpt_3_5_no_meta.save(
        tsv_file_path=DIFF_NEW_GPT_3_5_META_VS_NEW_GPT_3_5_NO_META_PATH,
        skip_asr_results=False,
    )

    ############################################################################

    print("Difference of")
    print('"New records, GPT 5.4 (new prompt), without metadata" versus')
    print('"New records, GPT 3.5 (new prompt, DHH parameters), without'
          ' metadata":')

    # Difference of
    # "New records, GPT 5.4 (new prompt), without metadata" versus
    # "New records, GPT 3.5 (new prompt, DHH parameters), without metadata".
    diff_new_gpt_5_4_no_meta_vs_new_gpt_3_5_no_meta: RecordList = RecordList(
        tsv_file_path=None,
    )

    # New records, GPT 5.4 (new prompt), without metadata.
    if "records_new_gpt_5_4_no_meta" not in locals():
        records_new_gpt_5_4_no_meta: RecordList = RecordList(
            tsv_file_path=NEW_GPT_5_4_NO_META_PATH,
        )

    # New records, GPT 3.5 (new prompt, DHH parameters), without metadata.
    if "records_new_gpt_3_5_no_meta" not in locals():
        records_new_gpt_3_5_no_meta: RecordList = RecordList(
            tsv_file_path=NEW_GPT_3_5_NO_META_PATH,
        )

    # Calculates the differences of the results of the records. A difference is
    # taken as zero if the absolute difference is less than the specified
    # tolerance. This is to remove noise and floating point artifacts.
    diff_results_of_records(
        records_1=records_new_gpt_5_4_no_meta,
        records_2=records_new_gpt_3_5_no_meta,
        records_diff=diff_new_gpt_5_4_no_meta_vs_new_gpt_3_5_no_meta,
        tolerance=DIFF_TOLERANCE,
        ndigits_past_tolerance=None,
        return_none_if_input_is_none=True,
    )

    # Saves a copy of the differences that were just calculated.
    diff_new_gpt_5_4_no_meta_vs_new_gpt_3_5_no_meta.save(
        tsv_file_path=DIFF_NEW_GPT_5_4_NO_META_VS_NEW_GPT_3_5_NO_META_PATH,
        skip_asr_results=False,
    )

    ############################################################################

    print("Difference of")
    print('"New records, GPT 5.4 (new prompt), with metadata" versus')
    print('"New records, GPT 3.5 (new prompt, DHH parameters), with metadata":')

    # Difference of
    # "New records, GPT 5.4 (new prompt), with metadata" versus
    # "New records, GPT 3.5 (new prompt, DHH parameters), with metadata".
    diff_new_gpt_5_4_meta_vs_new_gpt_3_5_meta: RecordList = RecordList(
        tsv_file_path=None,
    )

    # New records, GPT 5.4 (new prompt), with metadata.
    if "records_new_gpt_5_4_meta" not in locals():
        records_new_gpt_5_4_meta: RecordList = RecordList(
            tsv_file_path=NEW_GPT_5_4_META_PATH,
        )

    # New records, GPT 3.5 (new prompt, DHH parameters), with metadata.
    if "records_new_gpt_3_5_meta" not in locals():
        records_new_gpt_3_5_meta: RecordList = RecordList(
            tsv_file_path=NEW_GPT_3_5_META_PATH,
        )

    # Calculates the differences of the results of the records. A difference is
    # taken as zero if the absolute difference is less than the specified
    # tolerance. This is to remove noise and floating point artifacts.
    diff_results_of_records(
        records_1=records_new_gpt_5_4_meta,
        records_2=records_new_gpt_3_5_meta,
        records_diff=diff_new_gpt_5_4_meta_vs_new_gpt_3_5_meta,
        tolerance=DIFF_TOLERANCE,
        ndigits_past_tolerance=None,
        return_none_if_input_is_none=True,
    )

    # Saves a copy of the differences that were just calculated.
    diff_new_gpt_5_4_meta_vs_new_gpt_3_5_meta.save(
        tsv_file_path=DIFF_NEW_GPT_5_4_META_VS_NEW_GPT_3_5_META_PATH,
        skip_asr_results=False,
    )

    ############################################################################

    print("Difference of")
    print('"New records, GPT 5.4 (new prompt), with metadata" versus')
    print('"New records, GPT 5.4 (new prompt), without metadata":')

    # Difference of
    # "New records, GPT 5.4 (new prompt), with metadata" versus
    # "New records, GPT 5.4 (new prompt), without metadata".
    diff_new_gpt_5_4_meta_vs_new_gpt_5_4_no_meta: RecordList = RecordList(
        tsv_file_path=None,
    )

    # New records, GPT 5.4 (new prompt), with metadata.
    if "records_new_gpt_5_4_meta" not in locals():
        records_new_gpt_5_4_meta: RecordList = RecordList(
            tsv_file_path=NEW_GPT_5_4_META_PATH,
        )

    # New records, GPT 5.4 (new prompt), without metadata.
    if "records_new_gpt_5_4_no_meta" not in locals():
        records_new_gpt_5_4_no_meta: RecordList = RecordList(
            tsv_file_path=NEW_GPT_5_4_NO_META_PATH,
        )

    # Calculates the differences of the results of the records. A difference is
    # taken as zero if the absolute difference is less than the specified
    # tolerance. This is to remove noise and floating point artifacts.
    diff_results_of_records(
        records_1=records_new_gpt_5_4_meta,
        records_2=records_new_gpt_5_4_no_meta,
        records_diff=diff_new_gpt_5_4_meta_vs_new_gpt_5_4_no_meta,
        tolerance=DIFF_TOLERANCE,
        ndigits_past_tolerance=None,
        return_none_if_input_is_none=True,
    )

    # Saves a copy of the differences that were just calculated.
    diff_new_gpt_5_4_meta_vs_new_gpt_5_4_no_meta.save(
        tsv_file_path=DIFF_NEW_GPT_5_4_META_VS_NEW_GPT_5_4_NO_META_PATH,
        skip_asr_results=False,
    )

    ############################################################################

    print("Difference of")
    print('"New records, GPT 5.4 (new prompt), with metadata without top'
          ' comments" versus')
    print('"New records, GPT 5.4 (new prompt), with metadata":')

    # Difference of
    # "New records, GPT 5.4 (new prompt), with metadata w/o top comments" versus
    # "New records, GPT 5.4 (new prompt), with metadata".
    diff_new_gpt_5_4_meta_no_comments_vs_new_gpt_5_4_meta: RecordList
    diff_new_gpt_5_4_meta_no_comments_vs_new_gpt_5_4_meta = RecordList(
        tsv_file_path=None,
    )

    # New records, GPT 5.4 (new prompt), with metadata without top comments.
    if "records_new_gpt_5_4_meta_no_comments" not in locals():
        records_new_gpt_5_4_meta_no_comments: RecordList = RecordList(
            tsv_file_path=NEW_GPT_5_4_META_NO_COMMENTS_PATH,
        )

    # New records, GPT 5.4 (new prompt), with metadata.
    if "records_new_gpt_5_4_meta" not in locals():
        records_new_gpt_5_4_meta: RecordList = RecordList(
            tsv_file_path=NEW_GPT_5_4_META_PATH,
        )

    # Calculates the differences of the results of the records. A difference is
    # taken as zero if the absolute difference is less than the specified
    # tolerance. This is to remove noise and floating point artifacts.
    diff_results_of_records(
        records_1=records_new_gpt_5_4_meta_no_comments,
        records_2=records_new_gpt_5_4_meta,
        records_diff=diff_new_gpt_5_4_meta_no_comments_vs_new_gpt_5_4_meta,
        tolerance=DIFF_TOLERANCE,
        ndigits_past_tolerance=None,
        return_none_if_input_is_none=True,
    )

    # Saves a copy of the differences that were just calculated.
    diff_new_gpt_5_4_meta_no_comments_vs_new_gpt_5_4_meta.save(
        tsv_file_path=(
            DIFF_NEW_GPT_5_4_META_NO_COMMENTS_VS_NEW_GPT_5_4_META_PATH
        ),
        skip_asr_results=False,
    )

    ############################################################################

    print("Difference of")
    print('"New records, GPT 5.4 (new prompt), with metadata without top'
          ' comments" versus')
    print('"New records, GPT 5.4 (new prompt), without metadata":')

    # Difference of
    # "New records, GPT 5.4 (new prompt), with metadata w/o top comments" versus
    # "New records, GPT 5.4 (new prompt), without metadata".
    diff_new_gpt_5_4_meta_no_comments_vs_new_gpt_5_4_no_meta: RecordList
    diff_new_gpt_5_4_meta_no_comments_vs_new_gpt_5_4_no_meta = RecordList(
        tsv_file_path=None,
    )

    # New records, GPT 5.4 (new prompt), with metadata without top comments.
    if "records_new_gpt_5_4_meta_no_comments" not in locals():
        records_new_gpt_5_4_meta_no_comments: RecordList = RecordList(
            tsv_file_path=NEW_GPT_5_4_META_NO_COMMENTS_PATH,
        )

    # New records, GPT 5.4 (new prompt), without metadata.
    if "records_new_gpt_5_4_no_meta" not in locals():
        records_new_gpt_5_4_no_meta: RecordList = RecordList(
            tsv_file_path=NEW_GPT_5_4_NO_META_PATH,
        )

    # Calculates the differences of the results of the records. A difference is
    # taken as zero if the absolute difference is less than the specified
    # tolerance. This is to remove noise and floating point artifacts.
    diff_results_of_records(
        records_1=records_new_gpt_5_4_meta_no_comments,
        records_2=records_new_gpt_5_4_no_meta,
        records_diff=diff_new_gpt_5_4_meta_no_comments_vs_new_gpt_5_4_no_meta,
        tolerance=DIFF_TOLERANCE,
        ndigits_past_tolerance=None,
        return_none_if_input_is_none=True,
    )

    # Saves a copy of the differences that were just calculated.
    diff_new_gpt_5_4_meta_no_comments_vs_new_gpt_5_4_no_meta.save(
        tsv_file_path=(
            DIFF_NEW_GPT_5_4_META_NO_COMMENTS_VS_NEW_GPT_5_4_NO_META_PATH
        ),
        skip_asr_results=False,
    )

    ############################################################################

    return None


if __name__ == "__main__":
    main()
