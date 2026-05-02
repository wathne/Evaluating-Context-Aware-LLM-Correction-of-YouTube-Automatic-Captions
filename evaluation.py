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
from evaluation_specs import DIFF_TOLERANCE
from evaluation_specs import DIR
from evaluation_specs import EXT
from evaluation_specs import Specification
from evaluation_specs import specifications
from helpers import float_or_none
from helpers import float_or_zero_if_none
from helpers import int_or_none
from helpers import int_or_zero_if_none
from helpers import str_or_none_if_empty
from jiwer import wer
from llm_openai_gpt import DEFAULT_MODEL
from llm_openai_gpt import generate_openai_gpt_caption_for_records
from llm_openai_gpt import GPT_5_4_MODEL
from math import log10
from math import sqrt
from mean_evaluation_results_wrapper import Record as MeanRecord
from mean_evaluation_results_wrapper import RecordList as MeanRecordList
from plot import bar_chart_mean_records
from statistics import mean
from statistics import variance
from yt_metadata import fetch_metadata_for_records
from yt_transcripts import fetch_transcripts_for_records


# Local BLEU metric.
# Pulled from Hugging Face Evaluate repo, 2026 April 9, at commit a7dd338:
# https://github.com/huggingface/evaluate/tree/a7dd338/metrics/bleu
METRIC_BLEU: str = "./metrics/bleu/bleu.py"

# Local ROUGE metric.
# Pulled from Hugging Face Evaluate repo, 2026 April 9, at commit a7dd338:
# https://github.com/huggingface/evaluate/tree/a7dd338/metrics/rouge
METRIC_ROUGE: str = "./metrics/rouge/rouge.py"


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


# Supported column key integers and strings:
# 10 - "wer_gpt"
# 11 - "bleu_gpt"
# 12 - "rouge1_gpt"
# 13 - "rouge2_gpt"
# 14 - "rougeL_gpt"
# 15 - "rougeLsum_gpt"
# 16 - "wer_llama2"
# 17 - "bleu_llama2"
# 18 - "rouge1_llama2"
# 19 - "rouge2_llama2"
# 20 - "rougeL_llama2"
# 21 - "rougeLsum_llama2"
# 22 - "wer_asr"
# 23 - "bleu_asr"
# 24 - "rouge1_asr"
# 25 - "rouge2_asr"
# 26 - "rougeL_asr"
# 27 - "rougeLsum_asr"
def calculate_mean_of_results_of_records(
    records: RecordList,
    mean_record: MeanRecord,
    column_key: str | int,
    ndigits: int | None = 10,
) -> None:
    ndigits = int_or_none(ndigits)

    column: list[float] = records.list_result_column_only_float(column_key)

    column_mean: float
    column_stdev: float
    column_variance: float

    if len(column) > 0:
        column_mean = mean(data=column)

        mean_record.mean = round(number=column_mean, ndigits=ndigits)
    else:
        mean_record.mean = None

    if len(column) > 1:
        column_variance = variance(data=column, xbar=column_mean)
        column_stdev = sqrt(column_variance)

        mean_record.stdev = round(number=column_stdev, ndigits=ndigits)
        mean_record.variance = round(number=column_variance, ndigits=ndigits)
    else:
        mean_record.stdev = None
        mean_record.variance = None

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
    old_duplicated: Specification = specifications["old_duplicated"]

    # Initializes an empty Recordlist.
    old_duplicated.records = RecordList(
        tsv_file_path=None, # None: initialize an empty Recordlist.
    )

    # Loads DHH records, video ID, and extra data from the combination of DHH
    # records and DHH sources (corrected).
    initialize_records_from_dhh_records_and_sources(
        records=old_duplicated.records,
    )

    # Saves a copy of the records.
    old_duplicated.records.save(
        tsv_file_path=old_duplicated.path,
        skip_asr_results=True, # True: omit YouTube ASR result columns.
    )

    ############################################################################

    print("Old records (DHH), reevaluated:")

    # Old records (DHH), reevaluated.
    # Combination of DHH records and DHH sources (corrected).
    # Fetching: NO, keep old youtube captions (DHH).
    # Metadata: NO.
    # Prompting: NO, keep old GPT and Llama captions (DHH).
    # Evaluation: YES, evaluate by metrics WER, BLEU, and ROUGE.
    old_reevaluated: Specification = specifications["old_reevaluated"]

    # Initializes an empty Recordlist.
    old_reevaluated.records = RecordList(
        tsv_file_path=None, # None: initialize an empty Recordlist.
    )

    # Loads the duplicated old records that were saved earlier.
    old_reevaluated.records.load(
        tsv_file_path=specifications["old_duplicated"].path,
        clear=True,
    )

    # Evaluates the records (and clears existing results).
    evaluate_records(
        records=old_reevaluated.records,
    )

    # Saves a copy of the records.
    old_reevaluated.records.save(
        tsv_file_path=old_reevaluated.path,
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
    old_reprompt_gpt_3_5_dhh: Specification = specifications[
        "old_reprompt_gpt_3_5_dhh"
    ]

    # Initializes an empty Recordlist.
    old_reprompt_gpt_3_5_dhh.records = RecordList(
        tsv_file_path=None, # None: initialize an empty Recordlist.
    )

    # Loads the duplicated old records that were saved earlier.
    old_reprompt_gpt_3_5_dhh.records.load(
        tsv_file_path=specifications["old_duplicated"].path,
        clear=True,
    )

    # Clears old LLM captions.
    clear_llm_captions_from_records(
        records=old_reprompt_gpt_3_5_dhh.records,
    )

    # Generates new GPT captions.
    # Prompts GPT 3.5 with DHH prompt and DHH parameters.
    # Provides no metadata and no category in prompt.
    generate_openai_gpt_caption_for_records(
        records=old_reprompt_gpt_3_5_dhh.records,
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
        records=old_reprompt_gpt_3_5_dhh.records,
    )

    # Saves a copy of the records.
    old_reprompt_gpt_3_5_dhh.records.save(
        tsv_file_path=old_reprompt_gpt_3_5_dhh.path,
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
    old_reprompt_gpt_3_5_new: Specification = specifications[
        "old_reprompt_gpt_3_5_new"
    ]

    # Initializes an empty Recordlist.
    old_reprompt_gpt_3_5_new.records = RecordList(
        tsv_file_path=None, # None: initialize an empty Recordlist.
    )

    # Loads the duplicated old records that were saved earlier.
    old_reprompt_gpt_3_5_new.records.load(
        tsv_file_path=specifications["old_duplicated"].path,
        clear=True,
    )

    # Clears old LLM captions.
    clear_llm_captions_from_records(
        records=old_reprompt_gpt_3_5_new.records,
    )

    # Generates new GPT captions.
    # Prompts GPT 3.5 with new prompt and DHH parameters.
    # Provides no metadata and no category in prompt.
    generate_openai_gpt_caption_for_records(
        records=old_reprompt_gpt_3_5_new.records,
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
        records=old_reprompt_gpt_3_5_new.records,
    )

    # Saves a copy of the records.
    old_reprompt_gpt_3_5_new.records.save(
        tsv_file_path=old_reprompt_gpt_3_5_new.path,
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
    new_no_llm: Specification = specifications["new_no_llm"]

    # Initializes an empty Recordlist.
    new_no_llm.records = RecordList(
        tsv_file_path=None, # None: initialize an empty Recordlist.
    )

    # TODO: uncomment/comment:
    # Bases new records on the duplicated old records that were saved earlier.
    #new_no_llm.records.load(
    #    tsv_file_path=specifications["old_duplicated"].path,
    #    clear=True,
    #)

    # TODO: comment/uncomment:
    # Loads existing new records (existing new transcripts), to avoid getting
    # banned for fetching all new transcripts again.
    new_no_llm.records.load(
        tsv_file_path=new_no_llm.path,
        clear=True,
    )

    # Clears old results.
    clear_results_from_records(
        records=new_no_llm.records,
    )

    # Clears old LLM captions.
    clear_llm_captions_from_records(
        records=new_no_llm.records,
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
        records=new_no_llm.records,
        clear_unavailable_transcripts=True,
        wait_for_input_to_continue=False,
        wait_milliseconds_min=120000, # 120 seconds, precaution to avoid ban.
        wait_milliseconds_max=240000, # 240 seconds, precaution to avoid ban.
    )

    # Saves a copy of the records (new autogen transcripts).
    new_no_llm.records.save(
        tsv_file_path=new_no_llm.path,
        skip_asr_results=False,
    )

    # TODO: uncomment/comment:
    # Fetches video metadata: title, description, and top comments (relevance).
    # Overwrites existing metadata, if any, with newly fetched metadata.
    #fetch_metadata_for_records(
    #    records=new_no_llm.records,
    #    wait_milliseconds_min=6000,
    #    wait_milliseconds_max=12000,
    #)

    # Saves a copy of the records (new autogen transcripts and metadata).
    new_no_llm.records.save(
        tsv_file_path=new_no_llm.path,
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
    new_gpt_3_5_no_meta: Specification = specifications["new_gpt_3_5_no_meta"]

    # Initializes an empty Recordlist.
    new_gpt_3_5_no_meta.records = RecordList(
        tsv_file_path=None, # None: initialize an empty Recordlist.
    )

    # Loads the new records that were saved earlier (new autogen transcripts and
    # metadata).
    new_gpt_3_5_no_meta.records.load(
        tsv_file_path=specifications["new_no_llm"].path,
        clear=True,
    )

    # Clears old LLM captions, just in case they are not cleared.
    clear_llm_captions_from_records(
        records=new_gpt_3_5_no_meta.records,
    )

    # Generates new GPT captions.
    # Prompts GPT 3.5 with new prompt and DHH parameters.
    # Provides no metadata and no category in prompt.
    generate_openai_gpt_caption_for_records(
        records=new_gpt_3_5_no_meta.records,
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
        records=new_gpt_3_5_no_meta.records,
    )

    # Saves a copy of the records.
    new_gpt_3_5_no_meta.records.save(
        tsv_file_path=new_gpt_3_5_no_meta.path,
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
    new_gpt_3_5_meta: Specification = specifications["new_gpt_3_5_meta"]

    # Initializes an empty Recordlist.
    new_gpt_3_5_meta.records = RecordList(
        tsv_file_path=None, # None: initialize an empty Recordlist.
    )

    # Loads the new records that were saved earlier (new autogen transcripts and
    # metadata).
    new_gpt_3_5_meta.records.load(
        tsv_file_path=specifications["new_no_llm"].path,
        clear=True,
    )

    # Clears old LLM captions, just in case they are not cleared.
    clear_llm_captions_from_records(
        records=new_gpt_3_5_meta.records,
    )

    # Generates new GPT captions.
    # Prompts GPT 3.5 with new prompt and DHH parameters.
    # Provides all metadata in prompt, no category in prompt.
    generate_openai_gpt_caption_for_records(
        records=new_gpt_3_5_meta.records,
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
        records=new_gpt_3_5_meta.records,
    )

    # Saves a copy of the records.
    new_gpt_3_5_meta.records.save(
        tsv_file_path=new_gpt_3_5_meta.path,
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
    new_gpt_5_4_no_meta: Specification = specifications["new_gpt_5_4_no_meta"]

    # Initializes an empty Recordlist.
    new_gpt_5_4_no_meta.records = RecordList(
        tsv_file_path=None, # None: initialize an empty Recordlist.
    )

    # Loads the new records that were saved earlier (new autogen transcripts and
    # metadata).
    new_gpt_5_4_no_meta.records.load(
        tsv_file_path=specifications["new_no_llm"].path,
        clear=True,
    )

    # Clears old LLM captions, just in case they are not cleared.
    clear_llm_captions_from_records(
        records=new_gpt_5_4_no_meta.records,
    )

    # Generates new GPT captions.
    # Prompts GPT 5.4 with new prompt and default parameters.
    # Provides no metadata and no category in prompt.
    generate_openai_gpt_caption_for_records(
        records=new_gpt_5_4_no_meta.records,
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
        records=new_gpt_5_4_no_meta.records,
    )

    # Saves a copy of the records.
    new_gpt_5_4_no_meta.records.save(
        tsv_file_path=new_gpt_5_4_no_meta.path,
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
    new_gpt_5_4_meta: Specification = specifications["new_gpt_5_4_meta"]

    # Initializes an empty Recordlist.
    new_gpt_5_4_meta.records = RecordList(
        tsv_file_path=None, # None: initialize an empty Recordlist.
    )

    # Loads the new records that were saved earlier (new autogen transcripts and
    # metadata).
    new_gpt_5_4_meta.records.load(
        tsv_file_path=specifications["new_no_llm"].path,
        clear=True,
    )

    # Clears old LLM captions, just in case they are not cleared.
    clear_llm_captions_from_records(
        records=new_gpt_5_4_meta.records,
    )

    # Generates new GPT captions.
    # Prompts GPT 5.4 with new prompt and default parameters.
    # Provides all metadata in prompt, no category in prompt.
    generate_openai_gpt_caption_for_records(
        records=new_gpt_5_4_meta.records,
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
        records=new_gpt_5_4_meta.records,
    )

    # Saves a copy of the records.
    new_gpt_5_4_meta.records.save(
        tsv_file_path=new_gpt_5_4_meta.path,
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
    new_gpt_5_4_meta_no_comments: Specification = specifications[
        "new_gpt_5_4_meta_no_comments"
    ]

    # Initializes an empty Recordlist.
    new_gpt_5_4_meta_no_comments.records = RecordList(
        tsv_file_path=None, # None: initialize an empty Recordlist.
    )

    # Loads the new records that were saved earlier (new autogen transcripts and
    # metadata).
    new_gpt_5_4_meta_no_comments.records.load(
        tsv_file_path=specifications["new_no_llm"].path,
        clear=True,
    )

    # Clears old LLM captions, just in case they are not cleared.
    clear_llm_captions_from_records(
        records=new_gpt_5_4_meta_no_comments.records,
    )

    # Generates new GPT captions.
    # Prompts GPT 5.4 with new prompt and default parameters.
    # Provides metadata without top comments in prompt, no category in prompt.
    generate_openai_gpt_caption_for_records(
        records=new_gpt_5_4_meta_no_comments.records,
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
        records=new_gpt_5_4_meta_no_comments.records,
    )

    # Saves a copy of the records.
    new_gpt_5_4_meta_no_comments.records.save(
        tsv_file_path=new_gpt_5_4_meta_no_comments.path,
        skip_asr_results=False,
    )

    ############################################################################

    print("Difference of")
    print('"Old records (DHH), reevaluated" versus')
    print('"Old records (DHH), duplicated":')

    # Difference of
    # "Old records (DHH), reevaluated" versus
    # "Old records (DHH), duplicated".
    diff_old_reevaluated_vs_old_duplicated: Specification = specifications[
        "diff_old_reevaluated_vs_old_duplicated"
    ]

    # Initializes an empty Recordlist.
    diff_old_reevaluated_vs_old_duplicated.records = RecordList(
        tsv_file_path=None, # None: initialize an empty Recordlist.
    )

    # Old records (DHH), reevaluated.
    if specifications["old_reevaluated"].records is None:
        specifications["old_reevaluated"].records = RecordList(
            tsv_file_path=specifications["old_reevaluated"].path,
        )

    # Old records (DHH), duplicated.
    if specifications["old_duplicated"].records is None:
        specifications["old_duplicated"].records = RecordList(
            tsv_file_path=specifications["old_duplicated"].path,
        )

    # Calculates the differences of the results of the records. A difference is
    # taken as zero if the absolute difference is less than the specified
    # tolerance. This is to remove noise and floating point artifacts.
    diff_results_of_records(
        records_1=specifications["old_reevaluated"].records,
        records_2=specifications["old_duplicated"].records,
        records_diff=diff_old_reevaluated_vs_old_duplicated.records,
        tolerance=DIFF_TOLERANCE,
        ndigits_past_tolerance=None,
        return_none_if_input_is_none=True,
    )

    # Saves a copy of the differences that were just calculated.
    diff_old_reevaluated_vs_old_duplicated.records.save(
        tsv_file_path=diff_old_reevaluated_vs_old_duplicated.path,
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
    diff_old_reprompt_gpt_3_5_dhh_vs_old_reevaluated: Specification
    diff_old_reprompt_gpt_3_5_dhh_vs_old_reevaluated = specifications[
        "diff_old_reprompt_gpt_3_5_dhh_vs_old_reevaluated"
    ]

    # Initializes an empty Recordlist.
    diff_old_reprompt_gpt_3_5_dhh_vs_old_reevaluated.records = RecordList(
        tsv_file_path=None, # None: initialize an empty Recordlist.
    )

    # Old records (DHH), reprompted GPT 3.5 (DHH prompt, DHH parameters).
    if specifications["old_reprompt_gpt_3_5_dhh"].records is None:
        specifications["old_reprompt_gpt_3_5_dhh"].records = RecordList(
            tsv_file_path=specifications["old_reprompt_gpt_3_5_dhh"].path,
        )

    # Old records (DHH), reevaluated.
    if specifications["old_reevaluated"].records is None:
        specifications["old_reevaluated"].records = RecordList(
            tsv_file_path=specifications["old_reevaluated"].path,
        )

    # Calculates the differences of the results of the records. A difference is
    # taken as zero if the absolute difference is less than the specified
    # tolerance. This is to remove noise and floating point artifacts.
    diff_results_of_records(
        records_1=specifications["old_reprompt_gpt_3_5_dhh"].records,
        records_2=specifications["old_reevaluated"].records,
        records_diff=diff_old_reprompt_gpt_3_5_dhh_vs_old_reevaluated.records,
        tolerance=DIFF_TOLERANCE,
        ndigits_past_tolerance=None,
        return_none_if_input_is_none=True,
    )

    # Saves a copy of the differences that were just calculated.
    diff_old_reprompt_gpt_3_5_dhh_vs_old_reevaluated.records.save(
        tsv_file_path=diff_old_reprompt_gpt_3_5_dhh_vs_old_reevaluated.path,
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
    diff_old_reprompt_gpt_3_5_new_vs_old_reevaluated: Specification
    diff_old_reprompt_gpt_3_5_new_vs_old_reevaluated = specifications[
        "diff_old_reprompt_gpt_3_5_new_vs_old_reevaluated"
    ]

    # Initializes an empty Recordlist.
    diff_old_reprompt_gpt_3_5_new_vs_old_reevaluated.records = RecordList(
        tsv_file_path=None, # None: initialize an empty Recordlist.
    )

    # Old records (DHH), reprompted GPT 3.5 (new prompt, DHH parameters).
    if specifications["old_reprompt_gpt_3_5_new"].records is None:
        specifications["old_reprompt_gpt_3_5_new"].records = RecordList(
            tsv_file_path=specifications["old_reprompt_gpt_3_5_new"].path,
        )

    # Old records (DHH), reevaluated.
    if specifications["old_reevaluated"].records is None:
        specifications["old_reevaluated"].records = RecordList(
            tsv_file_path=specifications["old_reevaluated"].path,
        )

    # Calculates the differences of the results of the records. A difference is
    # taken as zero if the absolute difference is less than the specified
    # tolerance. This is to remove noise and floating point artifacts.
    diff_results_of_records(
        records_1=specifications["old_reprompt_gpt_3_5_new"].records,
        records_2=specifications["old_reevaluated"].records,
        records_diff=diff_old_reprompt_gpt_3_5_new_vs_old_reevaluated.records,
        tolerance=DIFF_TOLERANCE,
        ndigits_past_tolerance=None,
        return_none_if_input_is_none=True,
    )

    # Saves a copy of the differences that were just calculated.
    diff_old_reprompt_gpt_3_5_new_vs_old_reevaluated.records.save(
        tsv_file_path=diff_old_reprompt_gpt_3_5_new_vs_old_reevaluated.path,
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
    diff_old_reprompt_gpt_3_5_new_vs_old_reprompt_gpt_3_5_dhh: Specification
    diff_old_reprompt_gpt_3_5_new_vs_old_reprompt_gpt_3_5_dhh = specifications[
        "diff_old_reprompt_gpt_3_5_new_vs_old_reprompt_gpt_3_5_dhh"
    ]

    # Initializes an empty Recordlist.
    diff_old_reprompt_gpt_3_5_new_vs_old_reprompt_gpt_3_5_dhh.records = (
        RecordList(
            tsv_file_path=None, # None: initialize an empty Recordlist.
        )
    )

    # Old records (DHH), reprompted GPT 3.5 (new prompt, DHH parameters).
    if specifications["old_reprompt_gpt_3_5_new"].records is None:
        specifications["old_reprompt_gpt_3_5_new"].records = RecordList(
            tsv_file_path=specifications["old_reprompt_gpt_3_5_new"].path,
        )

    # Old records (DHH), reprompted GPT 3.5 (DHH prompt, DHH parameters).
    if specifications["old_reprompt_gpt_3_5_dhh"].records is None:
        specifications["old_reprompt_gpt_3_5_dhh"].records = RecordList(
            tsv_file_path=specifications["old_reprompt_gpt_3_5_dhh"].path,
        )

    # Calculates the differences of the results of the records. A difference is
    # taken as zero if the absolute difference is less than the specified
    # tolerance. This is to remove noise and floating point artifacts.
    diff_results_of_records(
        records_1=specifications["old_reprompt_gpt_3_5_new"].records,
        records_2=specifications["old_reprompt_gpt_3_5_dhh"].records,
        records_diff=(
            diff_old_reprompt_gpt_3_5_new_vs_old_reprompt_gpt_3_5_dhh.records
        ),
        tolerance=DIFF_TOLERANCE,
        ndigits_past_tolerance=None,
        return_none_if_input_is_none=True,
    )

    # Saves a copy of the differences that were just calculated.
    diff_old_reprompt_gpt_3_5_new_vs_old_reprompt_gpt_3_5_dhh.records.save(
        tsv_file_path=(
            diff_old_reprompt_gpt_3_5_new_vs_old_reprompt_gpt_3_5_dhh.path
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
    diff_new_gpt_3_5_no_meta_vs_old_reprompt_gpt_3_5_new: Specification
    diff_new_gpt_3_5_no_meta_vs_old_reprompt_gpt_3_5_new = specifications[
        "diff_new_gpt_3_5_no_meta_vs_old_reprompt_gpt_3_5_new"
    ]

    # Initializes an empty Recordlist.
    diff_new_gpt_3_5_no_meta_vs_old_reprompt_gpt_3_5_new.records = RecordList(
        tsv_file_path=None, # None: initialize an empty Recordlist.
    )

    # New records, GPT 3.5 (new prompt, DHH parameters), without metadata.
    if specifications["new_gpt_3_5_no_meta"].records is None:
        specifications["new_gpt_3_5_no_meta"].records = RecordList(
            tsv_file_path=specifications["new_gpt_3_5_no_meta"].path,
        )

    # Old records (DHH), reprompted GPT 3.5 (new prompt, DHH parameters).
    if specifications["old_reprompt_gpt_3_5_new"].records is None:
        specifications["old_reprompt_gpt_3_5_new"].records = RecordList(
            tsv_file_path=specifications["old_reprompt_gpt_3_5_new"].path,
        )

    # Calculates the differences of the results of the records. A difference is
    # taken as zero if the absolute difference is less than the specified
    # tolerance. This is to remove noise and floating point artifacts.
    diff_results_of_records(
        records_1=specifications["new_gpt_3_5_no_meta"].records,
        records_2=specifications["old_reprompt_gpt_3_5_new"].records,
        records_diff=(
            diff_new_gpt_3_5_no_meta_vs_old_reprompt_gpt_3_5_new.records
        ),
        tolerance=DIFF_TOLERANCE,
        ndigits_past_tolerance=None,
        return_none_if_input_is_none=True,
    )

    # Saves a copy of the differences that were just calculated.
    diff_new_gpt_3_5_no_meta_vs_old_reprompt_gpt_3_5_new.records.save(
        tsv_file_path=diff_new_gpt_3_5_no_meta_vs_old_reprompt_gpt_3_5_new.path,
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
    diff_new_gpt_3_5_meta_vs_new_gpt_3_5_no_meta: Specification
    diff_new_gpt_3_5_meta_vs_new_gpt_3_5_no_meta = specifications[
        "diff_new_gpt_3_5_meta_vs_new_gpt_3_5_no_meta"
    ]

    # Initializes an empty Recordlist.
    diff_new_gpt_3_5_meta_vs_new_gpt_3_5_no_meta.records = RecordList(
        tsv_file_path=None, # None: initialize an empty Recordlist.
    )

    # New records, GPT 3.5 (new prompt, DHH parameters), with metadata.
    if specifications["new_gpt_3_5_meta"].records is None:
        specifications["new_gpt_3_5_meta"].records = RecordList(
            tsv_file_path=specifications["new_gpt_3_5_meta"].path,
        )

    # New records, GPT 3.5 (new prompt, DHH parameters), without metadata.
    if specifications["new_gpt_3_5_no_meta"].records is None:
        specifications["new_gpt_3_5_no_meta"].records = RecordList(
            tsv_file_path=specifications["new_gpt_3_5_no_meta"].path,
        )

    # Calculates the differences of the results of the records. A difference is
    # taken as zero if the absolute difference is less than the specified
    # tolerance. This is to remove noise and floating point artifacts.
    diff_results_of_records(
        records_1=specifications["new_gpt_3_5_meta"].records,
        records_2=specifications["new_gpt_3_5_no_meta"].records,
        records_diff=diff_new_gpt_3_5_meta_vs_new_gpt_3_5_no_meta.records,
        tolerance=DIFF_TOLERANCE,
        ndigits_past_tolerance=None,
        return_none_if_input_is_none=True,
    )

    # Saves a copy of the differences that were just calculated.
    diff_new_gpt_3_5_meta_vs_new_gpt_3_5_no_meta.records.save(
        tsv_file_path=diff_new_gpt_3_5_meta_vs_new_gpt_3_5_no_meta.path,
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
    diff_new_gpt_5_4_no_meta_vs_new_gpt_3_5_no_meta: Specification
    diff_new_gpt_5_4_no_meta_vs_new_gpt_3_5_no_meta = specifications[
        "diff_new_gpt_5_4_no_meta_vs_new_gpt_3_5_no_meta"
    ]

    # Initializes an empty Recordlist.
    diff_new_gpt_5_4_no_meta_vs_new_gpt_3_5_no_meta.records = RecordList(
        tsv_file_path=None, # None: initialize an empty Recordlist.
    )

    # New records, GPT 5.4 (new prompt), without metadata.
    if specifications["new_gpt_5_4_no_meta"].records is None:
        specifications["new_gpt_5_4_no_meta"].records = RecordList(
            tsv_file_path=specifications["new_gpt_5_4_no_meta"].path,
        )

    # New records, GPT 3.5 (new prompt, DHH parameters), without metadata.
    if specifications["new_gpt_3_5_no_meta"].records is None:
        specifications["new_gpt_3_5_no_meta"].records = RecordList(
            tsv_file_path=specifications["new_gpt_3_5_no_meta"].path,
        )

    # Calculates the differences of the results of the records. A difference is
    # taken as zero if the absolute difference is less than the specified
    # tolerance. This is to remove noise and floating point artifacts.
    diff_results_of_records(
        records_1=specifications["new_gpt_5_4_no_meta"].records,
        records_2=specifications["new_gpt_3_5_no_meta"].records,
        records_diff=diff_new_gpt_5_4_no_meta_vs_new_gpt_3_5_no_meta.records,
        tolerance=DIFF_TOLERANCE,
        ndigits_past_tolerance=None,
        return_none_if_input_is_none=True,
    )

    # Saves a copy of the differences that were just calculated.
    diff_new_gpt_5_4_no_meta_vs_new_gpt_3_5_no_meta.records.save(
        tsv_file_path=diff_new_gpt_5_4_no_meta_vs_new_gpt_3_5_no_meta.path,
        skip_asr_results=False,
    )

    ############################################################################

    print("Difference of")
    print('"New records, GPT 5.4 (new prompt), with metadata" versus')
    print('"New records, GPT 3.5 (new prompt, DHH parameters), with metadata":')

    # Difference of
    # "New records, GPT 5.4 (new prompt), with metadata" versus
    # "New records, GPT 3.5 (new prompt, DHH parameters), with metadata".
    diff_new_gpt_5_4_meta_vs_new_gpt_3_5_meta: Specification = specifications[
        "diff_new_gpt_5_4_meta_vs_new_gpt_3_5_meta"
    ]

    # Initializes an empty Recordlist.
    diff_new_gpt_5_4_meta_vs_new_gpt_3_5_meta.records = RecordList(
        tsv_file_path=None, # None: initialize an empty Recordlist.
    )

    # New records, GPT 5.4 (new prompt), with metadata.
    if specifications["new_gpt_5_4_meta"].records is None:
        specifications["new_gpt_5_4_meta"].records = RecordList(
            tsv_file_path=specifications["new_gpt_5_4_meta"].path,
        )

    # New records, GPT 3.5 (new prompt, DHH parameters), with metadata.
    if specifications["new_gpt_3_5_meta"].records is None:
        specifications["new_gpt_3_5_meta"].records = RecordList(
            tsv_file_path=specifications["new_gpt_3_5_meta"].path,
        )

    # Calculates the differences of the results of the records. A difference is
    # taken as zero if the absolute difference is less than the specified
    # tolerance. This is to remove noise and floating point artifacts.
    diff_results_of_records(
        records_1=specifications["new_gpt_5_4_meta"].records,
        records_2=specifications["new_gpt_3_5_meta"].records,
        records_diff=diff_new_gpt_5_4_meta_vs_new_gpt_3_5_meta.records,
        tolerance=DIFF_TOLERANCE,
        ndigits_past_tolerance=None,
        return_none_if_input_is_none=True,
    )

    # Saves a copy of the differences that were just calculated.
    diff_new_gpt_5_4_meta_vs_new_gpt_3_5_meta.records.save(
        tsv_file_path=diff_new_gpt_5_4_meta_vs_new_gpt_3_5_meta.path,
        skip_asr_results=False,
    )

    ############################################################################

    print("Difference of")
    print('"New records, GPT 5.4 (new prompt), with metadata" versus')
    print('"New records, GPT 5.4 (new prompt), without metadata":')

    # Difference of
    # "New records, GPT 5.4 (new prompt), with metadata" versus
    # "New records, GPT 5.4 (new prompt), without metadata".
    diff_new_gpt_5_4_meta_vs_new_gpt_5_4_no_meta: Specification
    diff_new_gpt_5_4_meta_vs_new_gpt_5_4_no_meta = specifications[
        "diff_new_gpt_5_4_meta_vs_new_gpt_5_4_no_meta"
    ]

    # Initializes an empty Recordlist.
    diff_new_gpt_5_4_meta_vs_new_gpt_5_4_no_meta.records = RecordList(
        tsv_file_path=None, # None: initialize an empty Recordlist.
    )

    # New records, GPT 5.4 (new prompt), with metadata.
    if specifications["new_gpt_5_4_meta"].records is None:
        specifications["new_gpt_5_4_meta"].records = RecordList(
            tsv_file_path=specifications["new_gpt_5_4_meta"].path,
        )

    # New records, GPT 5.4 (new prompt), without metadata.
    if specifications["new_gpt_5_4_no_meta"].records is None:
        specifications["new_gpt_5_4_no_meta"].records = RecordList(
            tsv_file_path=specifications["new_gpt_5_4_no_meta"].path,
        )

    # Calculates the differences of the results of the records. A difference is
    # taken as zero if the absolute difference is less than the specified
    # tolerance. This is to remove noise and floating point artifacts.
    diff_results_of_records(
        records_1=specifications["new_gpt_5_4_meta"].records,
        records_2=specifications["new_gpt_5_4_no_meta"].records,
        records_diff=diff_new_gpt_5_4_meta_vs_new_gpt_5_4_no_meta.records,
        tolerance=DIFF_TOLERANCE,
        ndigits_past_tolerance=None,
        return_none_if_input_is_none=True,
    )

    # Saves a copy of the differences that were just calculated.
    diff_new_gpt_5_4_meta_vs_new_gpt_5_4_no_meta.records.save(
        tsv_file_path=diff_new_gpt_5_4_meta_vs_new_gpt_5_4_no_meta.path,
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
    diff_new_gpt_5_4_meta_no_comments_vs_new_gpt_5_4_meta: Specification
    diff_new_gpt_5_4_meta_no_comments_vs_new_gpt_5_4_meta = specifications[
        "diff_new_gpt_5_4_meta_no_comments_vs_new_gpt_5_4_meta"
    ]

    # Initializes an empty Recordlist.
    diff_new_gpt_5_4_meta_no_comments_vs_new_gpt_5_4_meta.records = RecordList(
        tsv_file_path=None, # None: initialize an empty Recordlist.
    )

    # New records, GPT 5.4 (new prompt), with metadata without top comments.
    if specifications["new_gpt_5_4_meta_no_comments"].records is None:
        specifications["new_gpt_5_4_meta_no_comments"].records = RecordList(
            tsv_file_path=specifications["new_gpt_5_4_meta_no_comments"].path,
        )

    # New records, GPT 5.4 (new prompt), with metadata.
    if specifications["new_gpt_5_4_meta"].records is None:
        specifications["new_gpt_5_4_meta"].records = RecordList(
            tsv_file_path=specifications["new_gpt_5_4_meta"].path,
        )

    # Calculates the differences of the results of the records. A difference is
    # taken as zero if the absolute difference is less than the specified
    # tolerance. This is to remove noise and floating point artifacts.
    diff_results_of_records(
        records_1=specifications["new_gpt_5_4_meta_no_comments"].records,
        records_2=specifications["new_gpt_5_4_meta"].records,
        records_diff=(
            diff_new_gpt_5_4_meta_no_comments_vs_new_gpt_5_4_meta.records
        ),
        tolerance=DIFF_TOLERANCE,
        ndigits_past_tolerance=None,
        return_none_if_input_is_none=True,
    )

    # Saves a copy of the differences that were just calculated.
    diff_new_gpt_5_4_meta_no_comments_vs_new_gpt_5_4_meta.records.save(
        tsv_file_path=(
            diff_new_gpt_5_4_meta_no_comments_vs_new_gpt_5_4_meta.path
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
    diff_new_gpt_5_4_meta_no_comments_vs_new_gpt_5_4_no_meta: Specification
    diff_new_gpt_5_4_meta_no_comments_vs_new_gpt_5_4_no_meta = specifications[
        "diff_new_gpt_5_4_meta_no_comments_vs_new_gpt_5_4_no_meta"
    ]

    # Initializes an empty Recordlist.
    diff_new_gpt_5_4_meta_no_comments_vs_new_gpt_5_4_no_meta.records = (
        RecordList(
            tsv_file_path=None, # None: initialize an empty Recordlist.
        )
    )

    # New records, GPT 5.4 (new prompt), with metadata without top comments.
    if specifications["new_gpt_5_4_meta_no_comments"].records is None:
        specifications["new_gpt_5_4_meta_no_comments"].records = RecordList(
            tsv_file_path=specifications["new_gpt_5_4_meta_no_comments"].path,
        )

    # New records, GPT 5.4 (new prompt), without metadata.
    if specifications["new_gpt_5_4_no_meta"].records is None:
        specifications["new_gpt_5_4_no_meta"].records = RecordList(
            tsv_file_path=specifications["new_gpt_5_4_no_meta"].path,
        )

    # Calculates the differences of the results of the records. A difference is
    # taken as zero if the absolute difference is less than the specified
    # tolerance. This is to remove noise and floating point artifacts.
    diff_results_of_records(
        records_1=specifications["new_gpt_5_4_meta_no_comments"].records,
        records_2=specifications["new_gpt_5_4_no_meta"].records,
        records_diff=(
            diff_new_gpt_5_4_meta_no_comments_vs_new_gpt_5_4_no_meta.records
        ),
        tolerance=DIFF_TOLERANCE,
        ndigits_past_tolerance=None,
        return_none_if_input_is_none=True,
    )

    # Saves a copy of the differences that were just calculated.
    diff_new_gpt_5_4_meta_no_comments_vs_new_gpt_5_4_no_meta.records.save(
        tsv_file_path=(
            diff_new_gpt_5_4_meta_no_comments_vs_new_gpt_5_4_no_meta.path
        ),
        skip_asr_results=False,
    )

    ############################################################################

    # Calculate mean of results and generate bar charts.

    whitelisted_specification_keys: list[str] = [
        "old_duplicated",               # 1
        "old_reevaluated",              # 2
        "old_reprompt_gpt_3_5_dhh",     # 3
        "old_reprompt_gpt_3_5_new",     # 4
        #"new_no_llm",
        "new_gpt_3_5_no_meta",          # 5
        "new_gpt_3_5_meta",             # 6
        "new_gpt_5_4_no_meta",          # 7
        "new_gpt_5_4_meta",             # 8
        "new_gpt_5_4_meta_no_comments", # 9
        #"diff_old_reevaluated_vs_old_duplicated",
        #"diff_old_reprompt_gpt_3_5_dhh_vs_old_reevaluated",
        #"diff_old_reprompt_gpt_3_5_new_vs_old_reevaluated",
        #"diff_old_reprompt_gpt_3_5_new_vs_old_reprompt_gpt_3_5_dhh",
        #"diff_new_gpt_3_5_no_meta_vs_old_reprompt_gpt_3_5_new",
        #"diff_new_gpt_3_5_meta_vs_new_gpt_3_5_no_meta",
        #"diff_new_gpt_5_4_no_meta_vs_new_gpt_3_5_no_meta",
        #"diff_new_gpt_5_4_meta_vs_new_gpt_3_5_meta",
        #"diff_new_gpt_5_4_meta_vs_new_gpt_5_4_no_meta",
        #"diff_new_gpt_5_4_meta_no_comments_vs_new_gpt_5_4_meta",
        #"diff_new_gpt_5_4_meta_no_comments_vs_new_gpt_5_4_no_meta",
    ]

    # This is a list of result keys, and it dictates the processing order.
    # Beware that result key "wer_asr" should be processed, and reference record
    # (ASR mean) set aside, before result keys "wer_gpt" and "wer_llama2" are
    # processed and plotted. The same goes for the rest: "*_asr" before "*_gpt"
    # and "*_llama2".
    result_keys: list[str] = [
        "wer_asr",
        "wer_gpt",
        "wer_llama2",

        "bleu_asr",
        "bleu_gpt",
        "bleu_llama2",

        "rouge1_asr",
        "rouge1_gpt",
        "rouge1_llama2",

        "rouge2_asr",
        "rouge2_gpt",
        "rouge2_llama2",

        "rougeL_asr",
        "rougeL_gpt",
        "rougeL_llama2",

        "rougeLsum_asr",
        "rougeLsum_gpt",
        "rougeLsum_llama2",
    ]

    asr_result_keys: list[str] = [
        "wer_asr",
        "bleu_asr",
        "rouge1_asr",
        "rouge2_asr",
        "rougeL_asr",
        "rougeLsum_asr",
    ]

    # This is a dictionary key to an arbitrarily chosen specification from which
    # to select benchmark values, specifically ASR means. All of the "new_*"
    # specifications have identical YouTube (ASR) captions and therefore
    # identical ASR means. Any one of them will be fine, for example
    # "new_gpt_5_4_no_meta". All ASR means of this specification will be set
    # aside. The idea is, for example, that a mean wer_gpt bar chart could
    # include an extra bar for wer_asr. And unlike wer_gpt, to cover wer_asr
    # we only need one extra bar because ASR means are invariant for all "new_*"
    # specifications. In other words, the mean value of wer_asr is identical for
    # "new_gpt_5_4_no_meta", "new_gpt_3_5_meta", or any other "new_*"
    # specification.
    # Beware that result key "wer_asr" must be processed, and reference record
    # (ASR mean) set aside, before result keys "wer_gpt" and "wer_llama2" are
    # processed and plotted. The same goes for the rest: "*_asr" before "*_gpt"
    # and "*_llama2".
    reference_new_specification_key: str = "new_gpt_5_4_no_meta"
    reference_new_asr_mean_record: MeanRecord | None
    reference_new_asr_mean_records: dict[str, MeanRecord | None] = {
        "wer_asr": None,
        "bleu_asr": None,
        "rouge1_asr": None,
        "rouge2_asr": None,
        "rougeL_asr": None,
        "rougeLsum_asr": None,
    }

    # There are also the invariant ASR means for "old_*" specifications. See the
    # explanation above, it is the same concept, just for "old_*"
    # specifications.
    # We can later include both new and old ASR means in all bar charts.
    reference_old_specification_key: str = "old_reevaluated"
    reference_old_asr_mean_record: MeanRecord | None
    reference_old_asr_mean_records: dict[str, MeanRecord | None] = {
        "wer_asr": None,
        "bleu_asr": None,
        "rouge1_asr": None,
        "rouge2_asr": None,
        "rougeL_asr": None,
        "rougeLsum_asr": None,
    }

    reference_result_key: str | None
    surjective_reference_result_key_map: dict[str, str | None] = {
        "wer_asr": "wer_asr", # Set None to skip ASR reference bars.
        "wer_gpt": "wer_asr",
        "wer_llama2": "wer_asr",

        "bleu_asr": "bleu_asr", # Set None to skip ASR reference bars.
        "bleu_gpt": "bleu_asr",
        "bleu_llama2": "bleu_asr",

        "rouge1_asr": "rouge1_asr", # Set None to skip ASR reference bars.
        "rouge1_gpt": "rouge1_asr",
        "rouge1_llama2": "rouge1_asr",

        "rouge2_asr": "rouge2_asr", # Set None to skip ASR reference bars.
        "rouge2_gpt": "rouge2_asr",
        "rouge2_llama2": "rouge2_asr",

        "rougeL_asr": "rougeL_asr", # Set None to skip ASR reference bars.
        "rougeL_gpt": "rougeL_asr",
        "rougeL_llama2": "rougeL_asr",

        "rougeLsum_asr": "rougeLsum_asr", # Set None to skip ASR reference bars.
        "rougeLsum_gpt": "rougeLsum_asr",
        "rougeLsum_llama2": "rougeLsum_asr",
    }

    bar_legends: list[str]
    bar_colors: list[str]
    reference_new_bar_legend: str | None
    reference_new_bar_color: str | None = "#696969" # darkgrey
    reference_old_bar_legend: str | None
    reference_old_bar_color: str | None = "#a9a9a9" # dimgrey

    mean_records: MeanRecordList
    mean_records_path: str
    mean_records_bar_chart_path: str
    column_key: str
    # Iterates over result keys.
    for column_key in result_keys:
        print(
            f"Calculating {column_key} mean and generating bar chart .",
            end="",
            flush=True,
        )

        bar_legends = []
        bar_colors = []

        mean_records = MeanRecordList(
            tsv_file_path=None,
            skip_variance=False,
        )

        mean_record: MeanRecord
        key: str
        specification: Specification
        # Iterates over imported specifications, see "./evaluation_specs.py".
        for key, specification in specifications.items():

            if key not in whitelisted_specification_keys:
                continue

            mean_record = MeanRecord(
                identifier=specification.identifier,
                mean=None,
                stdev=None,
                variance=None,
                description=specification.description,
            )

            if specification.records is None:
                specification.records = RecordList(
                    tsv_file_path=specification.path,
                )

            # Calculates the mean of the results (column_key) of the records.
            calculate_mean_of_results_of_records(
                records=specification.records,
                mean_record=mean_record,
                column_key=column_key,
                ndigits=4,
            )

            mean_records.append(mean_record)

            if specification.chart_legend is None:
                if specification.identifier is None:
                    bar_legends.append(key)
                else:
                    bar_legends.append(
                        str(specification.identifier) +
                        " - " +
                        key
                    )
            else:
                bar_legends.append(specification.chart_legend)

            if specification.chart_color is None:
                bar_colors.append("#7f7f7f") # grey (Tableu)
            else:
                bar_colors.append(specification.chart_color)

            # Sets aside reference record of new ASR mean.
            if key == reference_new_specification_key:
                if column_key in asr_result_keys:
                    reference_new_asr_mean_records[column_key] = mean_record

            # Sets aside reference record of old ASR mean.
            if key == reference_old_specification_key:
                if column_key in asr_result_keys:
                    reference_old_asr_mean_records[column_key] = mean_record

        print(".", end="", flush=True)

        mean_records_path = (
            DIR +
            "mean_evaluation_results_" +
            column_key +
            EXT
        )

        mean_records.save(
            tsv_file_path=mean_records_path,
            skip_variance=True,
        )

        print(".", end="", flush=True)

        mean_records_bar_chart_path = (
            DIR +
            "bar_charts/" +
            "bar_chart_mean_" +
            column_key +
            ".png"
        )

        reference_result_key = surjective_reference_result_key_map.get(
            column_key,
            None,
        )

        if reference_result_key is None:
            reference_new_asr_mean_record = None
            reference_old_asr_mean_record = None
            reference_new_bar_legend = None
            reference_old_bar_legend = None
        else:
            reference_new_asr_mean_record = reference_new_asr_mean_records.get(
                reference_result_key,
                None,
            )
            reference_old_asr_mean_record = reference_old_asr_mean_records.get(
                reference_result_key,
                None,
            )
            reference_new_bar_legend = (
                "*" + reference_result_key + " - new YT captions (no LLM)"
            )
            reference_old_bar_legend = (
                "*" + reference_result_key + " - old YT captions (no LLM)"
            )

        bar_chart_mean_records(
            mean_records=mean_records,
            bar_legends=bar_legends,
            bar_colors=bar_colors,
            bar_chart_path=mean_records_bar_chart_path,
            bar_chart_title=column_key,
            show_stdev=True,
            show_variance=False,
            reference_new_asr_mean_record=reference_new_asr_mean_record,
            reference_new_bar_legend=reference_new_bar_legend,
            reference_new_bar_color=reference_new_bar_color,
            reference_old_asr_mean_record=reference_old_asr_mean_record,
            reference_old_bar_legend=reference_old_bar_legend,
            reference_old_bar_color=reference_old_bar_color,
        )

        print("done.")

    ############################################################################

    return None


if __name__ == "__main__":
    main()
