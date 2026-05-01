# Evaluation specifications module.

# For convenience, let "DHH" be a shortened reference to the following study:
# "Empowering the Deaf and Hard of Hearing Community: Enhancing Video Captions
# Using Large Language Models".
# https://arxiv.org/abs/2412.00342
# https://github.com/monikabhole001/Improving-the-Quality-of-Video-Captions-for-the-DHH-Community-Using-LLM

# TODO: Include full filenames (as comments) to make it easier to search for
# filenames.


from captions_with_evaluation_results_wrapper import RecordList


DIR: str = "./captions_with_evaluation_results/"
BASE: str = "captions_with_evaluation_results"
EXT: str = ".tsv"

# "./captions_with_evaluation_results/captions_with_evaluation_results.tsv"
DEFAULT_PATH: str = DIR + BASE + EXT

# Difference tolerance.
# A difference is taken as zero if the absolute difference is less than the
# specified tolerance. This is to remove noise and floating point artifacts.
DIFF_TOLERANCE: float = 0.001


class Specification:
    def __init__(
        self,
        path: str = DEFAULT_PATH,
        records: RecordList | None = None,
        identifier: int | None = None,
        description: str | None = None,
        description_extended: str | None = None,
        chart_legend: str | None = None,
        chart_color: str | None = None,
    ) -> None:
        self.path: str = path
        self.records: RecordList | None = records
        self.identifier: int | None = identifier
        self.description: str | None = description
        self.description_extended: str | None = description_extended
        self.chart_legend: str | None = chart_legend
        self.chart_color: str | None = chart_color

        return None


specifications: dict[str, Specification] = {}

# Old records (DHH), duplicated.
# Combination of DHH records and DHH sources (corrected).
# Fetching: NO, keep old youtube captions (DHH).
# Metadata: NO.
# Prompting: NO, keep old GPT and Llama captions (DHH).
# Evaluation: NO, keep old evaluation results (DHH).
specifications["old_duplicated"] = Specification(
    path=(DIR + BASE + "_old_duplicated" + EXT),
    records=None,
    identifier=1,
    description="Old records (DHH), duplicated",
    description_extended=None,
    chart_legend="1 - old duplicated",
    chart_color="#1f77b4", # blue (Tableu)
)

# Old records (DHH), reevaluated.
# Combination of DHH records and DHH sources (corrected).
# Fetching: NO, keep old youtube captions (DHH).
# Metadata: NO.
# Prompting: NO, keep old GPT and Llama captions (DHH).
# Evaluation: YES, evaluate by metrics WER, BLEU, and ROUGE.
specifications["old_reevaluated"] = Specification(
    path=(DIR + BASE + "_old_reevaluated" + EXT),
    records=None,
    identifier=2,
    description="Old records (DHH), reevaluated",
    description_extended=None,
    chart_legend="2 - old reevaluated",
    chart_color="#ff7f0e", # orange (Tableu)
)

# Old records (DHH), reprompted GPT 3.5 (DHH prompt, DHH parameters).
# Combination of DHH records and DHH sources (corrected).
# Fetching: NO, keep old youtube captions (DHH).
# Metadata: NO, no metadata and no category in prompt.
# Prompting: YES, prompt GPT 3.5 with DHH prompt and DHH parameters.
# Evaluation: YES, evaluate by metrics WER, BLEU, and ROUGE.
specifications["old_reprompt_gpt_3_5_dhh"] = Specification(
    path=(DIR + BASE + "_old_reprompt_gpt_3_5_dhh" + EXT),
    records=None,
    identifier=3,
    description=(
        "Old records (DHH), reprompted GPT 3.5 (DHH prompt, DHH parameters)"
    ),
    description_extended=None,
    chart_legend="3 - old reprompt GPT 3.5 DHH prompt",
    chart_color="#2ca02c", # green (Tableu)
)

# Old records (DHH), reprompted GPT 3.5 (new prompt, DHH parameters).
# Combination of DHH records and DHH sources (corrected).
# Fetching: NO, keep old youtube captions (DHH).
# Metadata: NO, no metadata and no category in prompt.
# Prompting: YES, prompt GPT 3.5 with new prompt and DHH parameters.
# Evaluation: YES, evaluate by metrics WER, BLEU, and ROUGE.
specifications["old_reprompt_gpt_3_5_new"] = Specification(
    path=(DIR + BASE + "_old_reprompt_gpt_3_5_new" + EXT),
    records=None,
    identifier=4,
    description=(
        "Old records (DHH), reprompted GPT 3.5 (new prompt, DHH parameters)"
    ),
    description_extended=None,
    chart_legend="4 - old reprompt GPT 3.5 new prompt",
    chart_color="#d62728", # red (Tableu)
)

# New records, with metadata, without LLM captions, without results.
# Based on combination of DHH records and DHH sources (corrected).
# Fetching: YES, fetch new available transcripts and blacklist unavailable.
# Metadata: YES.
# Prompting: NO, all LLM captions are cleared.
# Evaluation: NO, all evaluation results are cleared.
specifications["new_no_llm"] = Specification(
    path=(DIR + "captions_new_no_llm" + EXT),
    records=None,
    identifier=None,
    description=(
        "New records, with metadata, without LLM captions, without results"
    ),
    description_extended=None,
    chart_legend=None,
    chart_color=None,
)

# New records, GPT 3.5 (new prompt, DHH parameters), without metadata.
# Based on combination of DHH records and DHH sources (corrected).
# Fetching: YES, fetch new available transcripts and blacklist unavailable.
# Metadata: NO, no metadata and no category in prompt.
# Prompting: YES, prompt GPT 3.5 with new prompt and DHH parameters.
# Evaluation: YES, evaluate by metrics WER, BLEU, and ROUGE.
specifications["new_gpt_3_5_no_meta"] = Specification(
    path=(DIR + BASE + "_new_gpt_3_5_no_metadata" + EXT),
    records=None,
    identifier=5,
    description=(
        "New records, GPT 3.5 (new prompt, DHH parameters), without metadata"
    ),
    description_extended=None,
    chart_legend="5 - new GPT 3.5 no meta",
    chart_color="#9467bd", # purple (Tableu)
)

# New records, GPT 3.5 (new prompt, DHH parameters), with metadata.
# Based on combination of DHH records and DHH sources (corrected).
# Fetching: YES, fetch new available transcripts and blacklist unavailable.
# Metadata: YES, all metadata in prompt, no category in prompt.
# Prompting: YES, prompt GPT 3.5 with new prompt and DHH parameters.
# Evaluation: YES, evaluate by metrics WER, BLEU, and ROUGE.
specifications["new_gpt_3_5_meta"] = Specification(
    path=(DIR + BASE + "_new_gpt_3_5_metadata" + EXT),
    records=None,
    identifier=6,
    description=(
        "New records, GPT 3.5 (new prompt, DHH parameters), with metadata"
    ),
    description_extended=None,
    chart_legend="6 - new GPT 3.5 meta",
    chart_color="#8c564b", # brown (Tableu)
)

# New records, GPT 5.4 (new prompt), without metadata.
# Based on combination of DHH records and DHH sources (corrected).
# Fetching: YES, fetch new available transcripts and blacklist unavailable.
# Metadata: NO, no metadata and no category in prompt.
# Prompting: YES, prompt GPT 5.4 with new prompt and default parameters.
# Evaluation: YES, evaluate by metrics WER, BLEU, and ROUGE.
specifications["new_gpt_5_4_no_meta"] = Specification(
    path=(DIR + BASE + "_new_gpt_5_4_no_metadata" + EXT),
    records=None,
    identifier=7,
    description="New records, GPT 5.4 (new prompt), without metadata",
    description_extended=None,
    chart_legend="7 - new GPT 5.4 no meta",
    chart_color="#e377c2", # pink (Tableu)
)

# New records, GPT 5.4 (new prompt), with metadata.
# Based on combination of DHH records and DHH sources (corrected).
# Fetching: YES, fetch new available transcripts and blacklist unavailable.
# Metadata: YES, all metadata in prompt, no category in prompt.
# Prompting: YES, prompt GPT 5.4 with new prompt and default parameters.
# Evaluation: YES, evaluate by metrics WER, BLEU, and ROUGE.
specifications["new_gpt_5_4_meta"] = Specification(
    path=(DIR + BASE + "_new_gpt_5_4_metadata" + EXT),
    records=None,
    identifier=8,
    description="New records, GPT 5.4 (new prompt), with metadata",
    description_extended=None,
    chart_legend="8 - new GPT 5.4 meta",
    chart_color="#bcbd22", # yellow (Tableu)
)

# New records, GPT 5.4 (new prompt), with metadata without top comments.
# Based on combination of DHH records and DHH sources (corrected).
# Fetching: YES, fetch new available transcripts and blacklist unavailable.
# Metadata: YES, metadata without top comments in prompt, no category in prompt.
# Prompting: YES, prompt GPT 5.4 with new prompt and default parameters.
# Evaluation: YES, evaluate by metrics WER, BLEU, and ROUGE.
specifications["new_gpt_5_4_meta_no_comments"] = Specification(
    path=(DIR + BASE + "_new_gpt_5_4_metadata_no_comments" + EXT),
    records=None,
    identifier=9,
    description=(
        "New records, GPT 5.4 (new prompt), with metadata without top comments"
    ),
    description_extended=None,
    chart_legend="9 - new GPT 5.4 meta no comments",
    chart_color="#17becf", # cyan (Tableu)
)

# Difference of
# "Old records (DHH), reevaluated" versus
# "Old records (DHH), duplicated".
# See also the DIFF_TOLERANCE constant.
specifications["diff_old_reevaluated_vs_old_duplicated"] = Specification(
    path=(DIR + "diff_old_reevaluated_vs_old_duplicated" + EXT),
    records=None,
    identifier=None,
    description=(
        'Difference of'
        ' "Old records (DHH), reevaluated" versus'
        ' "Old records (DHH), duplicated"'
    ),
    description_extended=None,
    chart_legend=None,
    chart_color=None,
)

# Difference of
# "Old records (DHH), reprompted GPT 3.5 (DHH prompt, DHH parameters)" versus
# "Old records (DHH), reevaluated".
# See also the DIFF_TOLERANCE constant.
specifications[
    "diff_old_reprompt_gpt_3_5_dhh_vs_old_reevaluated"
] = Specification(
    path=(DIR + "diff_old_reprompt_gpt_3_5_dhh_vs_old_reevaluated" + EXT),
    records=None,
    identifier=None,
    description=(
        'Difference of'
        ' "Old records (DHH), reprompted GPT 3.5 (DHH prompt, DHH parameters)"'
        ' versus'
        ' "Old records (DHH), reevaluated"'
    ),
    description_extended=None,
    chart_legend=None,
    chart_color=None,
)

# Difference of
# "Old records (DHH), reprompted GPT 3.5 (new prompt, DHH parameters)" versus
# "Old records (DHH), reevaluated".
# See also the DIFF_TOLERANCE constant.
specifications[
    "diff_old_reprompt_gpt_3_5_new_vs_old_reevaluated"
] = Specification(
    path=(DIR + "diff_old_reprompt_gpt_3_5_new_vs_old_reevaluated" + EXT),
    records=None,
    identifier=None,
    description=(
        'Difference of'
        ' "Old records (DHH), reprompted GPT 3.5 (new prompt, DHH parameters)"'
        ' versus'
        ' "Old records (DHH), reevaluated"'
    ),
    description_extended=None,
    chart_legend=None,
    chart_color=None,
)

# Difference of
# "Old records (DHH), reprompted GPT 3.5 (new prompt, DHH parameters)" versus
# "Old records (DHH), reprompted GPT 3.5 (DHH prompt, DHH parameters)".
# See also the DIFF_TOLERANCE constant.
specifications[
    "diff_old_reprompt_gpt_3_5_new_vs_old_reprompt_gpt_3_5_dhh"
] = Specification(
    path=(
        DIR + "diff_old_reprompt_gpt_3_5_new_vs_old_reprompt_gpt_3_5_dhh" + EXT
    ),
    records=None,
    identifier=None,
    description=(
        'Difference of'
        ' "Old records (DHH), reprompted GPT 3.5 (new prompt, DHH parameters)"'
        ' versus'
        ' "Old records (DHH), reprompted GPT 3.5 (DHH prompt, DHH parameters)"'
    ),
    description_extended=None,
    chart_legend=None,
    chart_color=None,
)

# Difference of
# "New records, GPT 3.5 (new prompt, DHH parameters), without metadata" versus
# "Old records (DHH), reprompted GPT 3.5 (new prompt, DHH parameters)".
# See also the DIFF_TOLERANCE constant.
specifications[
    "diff_new_gpt_3_5_no_meta_vs_old_reprompt_gpt_3_5_new"
] = Specification(
    path=(
        DIR + "diff_new_gpt_3_5_no_metadata_vs_old_reprompt_gpt_3_5_new" + EXT
    ),
    records=None,
    identifier=None,
    description=(
        'Difference of'
        ' "New records, GPT 3.5 (new prompt, DHH parameters), without metadata"'
        ' versus'
        ' "Old records (DHH), reprompted GPT 3.5 (new prompt, DHH parameters)"'
    ),
    description_extended=None,
    chart_legend=None,
    chart_color=None,
)

# Difference of
# "New records, GPT 3.5 (new prompt, DHH parameters), with metadata" versus
# "New records, GPT 3.5 (new prompt, DHH parameters), without metadata".
# See also the DIFF_TOLERANCE constant.
specifications[
    "diff_new_gpt_3_5_meta_vs_new_gpt_3_5_no_meta"
] = Specification(
    path=(DIR + "diff_new_gpt_3_5_metadata_vs_new_gpt_3_5_no_metadata" + EXT),
    records=None,
    identifier=None,
    description=(
        'Difference of'
        ' "New records, GPT 3.5 (new prompt, DHH parameters), with metadata"'
        ' versus'
        ' "New records, GPT 3.5 (new prompt, DHH parameters), without metadata"'
    ),
    description_extended=None,
    chart_legend=None,
    chart_color=None,
)

# Difference of
# "New records, GPT 5.4 (new prompt), without metadata" versus
# "New records, GPT 3.5 (new prompt, DHH parameters), without metadata".
# See also the DIFF_TOLERANCE constant.
specifications[
    "diff_new_gpt_5_4_no_meta_vs_new_gpt_3_5_no_meta"
] = Specification(
    path=(
        DIR + "diff_new_gpt_5_4_no_metadata_vs_new_gpt_3_5_no_metadata" + EXT
    ),
    records=None,
    identifier=None,
    description=(
        'Difference of'
        ' "New records, GPT 5.4 (new prompt), without metadata" versus'
        ' "New records, GPT 3.5 (new prompt, DHH parameters), without metadata"'
    ),
    description_extended=None,
    chart_legend=None,
    chart_color=None,
)

# Difference of
# "New records, GPT 5.4 (new prompt), with metadata" versus
# "New records, GPT 3.5 (new prompt, DHH parameters), with metadata".
# See also the DIFF_TOLERANCE constant.
specifications["diff_new_gpt_5_4_meta_vs_new_gpt_3_5_meta"] = Specification(
    path=(DIR + "diff_new_gpt_5_4_metadata_vs_new_gpt_3_5_metadata" + EXT),
    records=None,
    identifier=None,
    description=(
        'Difference of'
        ' "New records, GPT 5.4 (new prompt), with metadata" versus'
        ' "New records, GPT 3.5 (new prompt, DHH parameters), with metadata"'
    ),
    description_extended=None,
    chart_legend=None,
    chart_color=None,
)

# Difference of
# "New records, GPT 5.4 (new prompt), with metadata" versus
# "New records, GPT 5.4 (new prompt), without metadata".
# See also the DIFF_TOLERANCE constant.
specifications["diff_new_gpt_5_4_meta_vs_new_gpt_5_4_no_meta"] = Specification(
    path=(DIR + "diff_new_gpt_5_4_metadata_vs_new_gpt_5_4_no_metadata" + EXT),
    records=None,
    identifier=None,
    description=(
        'Difference of'
        ' "New records, GPT 5.4 (new prompt), with metadata" versus'
        ' "New records, GPT 5.4 (new prompt), without metadata"'
    ),
    description_extended=None,
    chart_legend=None,
    chart_color=None,
)

# Difference of
# "New records, GPT 5.4 (new prompt), with metadata without top comments" versus
# "New records, GPT 5.4 (new prompt), with metadata".
# See also the DIFF_TOLERANCE constant.
specifications[
    "diff_new_gpt_5_4_meta_no_comments_vs_new_gpt_5_4_meta"
] = Specification(
    path=(
        DIR +
        "diff_new_gpt_5_4_metadata_no_comments_vs_new_gpt_5_4_metadata" +
        EXT
    ),
    records=None,
    identifier=None,
    description=(
        'Difference of'
        ' "New records, GPT 5.4 (new prompt), with metadata without top'
        ' comments" versus'
        ' "New records, GPT 5.4 (new prompt), with metadata"'
    ),
    description_extended=None,
    chart_legend=None,
    chart_color=None,
)

# Difference of
# "New records, GPT 5.4 (new prompt), with metadata without top comments" versus
# "New records, GPT 5.4 (new prompt), without metadata".
# See also the DIFF_TOLERANCE constant.
specifications[
    "diff_new_gpt_5_4_meta_no_comments_vs_new_gpt_5_4_no_meta"
] = Specification(
    path=(
        DIR +
        "diff_new_gpt_5_4_metadata_no_comments_vs_new_gpt_5_4_no_metadata" +
        EXT
    ),
    records=None,
    identifier=None,
    description=(
        'Difference of'
        ' "New records, GPT 5.4 (new prompt), with metadata without top'
        ' comments" versus'
        ' "New records, GPT 5.4 (new prompt), without metadata"'
    ),
    description_extended=None,
    chart_legend=None,
    chart_color=None,
)
