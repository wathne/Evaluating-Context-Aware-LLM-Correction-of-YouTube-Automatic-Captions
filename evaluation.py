# Evaluation module.
#
# See the main function at the end. The main function is meant be modified.
#
# Set your environment variables or populate "./private_api_keys.py" with your
# own private API keys if you want to use LLM functionality (GPT, Llama, Claude,
# and Gemini). YouTube transcripts (closed captions) are fetched without an API
# key, but other video metadata requires a private API key. Modules will try to
# read a wanted API key from your environment variables if not found in
# "./private_api_keys.py".
#
# Complete list of API key environment variables:
# ANTHROPIC_CLAUDE_API_KEY
# GOOGLE_GEMINI_API_KEY
# GOOGLE_YOUTUBE_API_KEY
# META_LLAMA_API_KEY
# OPENAI_GPT_API_KEY

# Module requirements (included in requirements.txt):
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


from captions_with_evaluation_results_wrapper import clear_results_from_records
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

    return None


if __name__ == "__main__":
    main()
