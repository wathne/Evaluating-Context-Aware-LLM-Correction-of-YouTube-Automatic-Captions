# Read-write wrapper for "captions_with_evaluation_results*.tsv".
#
# These records are based on the same structure as the DHH records, but with a
# "Video_Id" column instead of the "File" column. The "Category" column is now
# just before the "Label" column. The "Ground_Truth" column is now just after
# the "youtube_caption" column. New columns "CC_Status", "Note", "Metadata",
# "wer_asr", "bleu_asr", "rouge1_asr", "rouge2_asr", "rougeL_asr", and
# "rougeLsum_asr" have been added:
# "CC_Status" - Status code for new closed captions.
# "Note" - Anything noteworthy or an explanation for the CC status.
# "Metadata" - JSON representation of video metadata: title, description, and
# top comments (relevance).
# "*_asr" - Evaluation results for YouTube ASR captions (pre LLM).
#
# This wrapper removes all tabs (delimiter) and newlines (lineterminator) from
# strings just before they are written to the TSV file. Removing tabs and
# newlines is necessary because no escape character has been set for the CSV
# reader/writer. The escape character is intentionally omitted to make it more
# likely that a viewer (program) will be able to display the TSV file correctly
# on default settings.
# The CSV reader/writer is configured for tab-separated values (TSV). The quote
# character (quotechar) and escape character (escapechar) are both undefined
# (None). "strict" mode is enabled and will raise an exception Error on bad
# input or output.

# For convenience, let "DHH" be a shortened reference to the following study:
# "Empowering the Deaf and Hard of Hearing Community: Enhancing Video Captions
# Using Large Language Models".
# https://arxiv.org/abs/2412.00342
# https://github.com/monikabhole001/Improving-the-Quality-of-Video-Captions-for-the-DHH-Community-Using-LLM


from collections.abc import MutableMapping
from collections.abc import MutableSequence
from csv import QUOTE_NONE
from csv import reader
from csv import writer
from DHH_data_source_wrapper import DEFAULT_PATH as DHH_SOURCES_PATH
from DHH_data_source_wrapper import Record as DHHSource
from DHH_data_source_wrapper import RecordList as DHHSourceList
from DHH_final_captions_with_evaluation_results_wrapper import DEFAULT_PATH as DHH_RECORDS_PATH
from DHH_final_captions_with_evaluation_results_wrapper import Record as DHHRecord
from DHH_final_captions_with_evaluation_results_wrapper import RecordList as DHHRecordList
from helpers import float_or_none
from helpers import int_or_none
from helpers import str_or_empty_if_none
from helpers import str_or_none
from helpers import str_or_none_if_empty
from pathlib import Path
from typing import Generator
from typing import TextIO


DIR: str = "./captions_with_evaluation_results/"
DEFAULT_PATH: str = DIR + "captions_with_evaluation_results.tsv"


# TODO(wathne): Rearranging or adding new columns is cumbersome. Make it easier.
class Record(MutableMapping):
    # MutableMapping must implement methods __getitem__, __setitem__,
    # __delitem__, __iter__, and __len__.

    def __init__(
        self,
        video_id: str | None = None,
        cc_status: int | None = None,
        note: str | None = None,
        category: str | None = None,
        label: int | None = None,
        metadata: str | None = None,
        youtube_caption: str | None = None,
        ground_truth: str | None = None,
        chatgpt_caption: str | None = None,
        llama2_caption: str | None = None,
        wer_gpt: float | None = None,
        bleu_gpt: float | None = None,
        rouge1_gpt: float | None = None,
        rouge2_gpt: float | None = None,
        rougeL_gpt: float | None = None,
        rougeLsum_gpt: float | None = None,
        wer_llama2: float | None = None,
        bleu_llama2: float | None = None,
        rouge1_llama2: float | None = None,
        rouge2_llama2: float | None = None,
        rougeL_llama2: float | None = None,
        rougeLsum_llama2: float | None = None,
        wer_asr: float | None = None,
        bleu_asr: float | None = None,
        rouge1_asr: float | None = None,
        rouge2_asr: float | None = None,
        rougeL_asr: float | None = None,
        rougeLsum_asr: float | None = None,
    ) -> None:
        self.video_id: str | None = video_id
        self.cc_status: int | None = cc_status
        self.note: str | None = note
        self.category: str | None = category
        self.label: int | None = label
        self.metadata: str | None = metadata
        self.youtube_caption: str | None = youtube_caption
        self.ground_truth: str | None = ground_truth
        self.chatgpt_caption: str | None = chatgpt_caption
        self.llama2_caption: str | None = llama2_caption
        self.wer_gpt: float | None = wer_gpt
        self.bleu_gpt: float | None = bleu_gpt
        self.rouge1_gpt: float | None = rouge1_gpt
        self.rouge2_gpt: float | None = rouge2_gpt
        self.rougeL_gpt: float | None = rougeL_gpt
        self.rougeLsum_gpt: float | None = rougeLsum_gpt
        self.wer_llama2: float | None = wer_llama2
        self.bleu_llama2: float | None = bleu_llama2
        self.rouge1_llama2: float | None = rouge1_llama2
        self.rouge2_llama2: float | None = rouge2_llama2
        self.rougeL_llama2: float | None = rougeL_llama2
        self.rougeLsum_llama2: float | None = rougeLsum_llama2
        self.wer_asr: float | None = wer_asr
        self.bleu_asr: float | None = bleu_asr
        self.rouge1_asr: float | None = rouge1_asr
        self.rouge2_asr: float | None = rouge2_asr
        self.rougeL_asr: float | None = rougeL_asr
        self.rougeLsum_asr: float | None = rougeLsum_asr

        super().__init__()
        return None

    def __getitem__(self, key: str | int) -> str | int | float | None:
        if isinstance(key, str):
            if (key == "video_id" or key == "Video_Id" or key == "Video_ID"):
                return self.video_id
            if (key == "cc_status" or key == "CC_Status"):
                return self.cc_status
            if (key == "note" or key == "Note"):
                return self.note
            if (key == "category" or key == "Category"):
                return self.category
            if (key == "label" or key == "Label"):
                return self.label
            if (key == "metadata" or key == "Metadata"):
                return self.metadata
            if (key == "youtube_caption"):
                return self.youtube_caption
            if (key == "ground_truth" or key == "Ground_Truth"):
                return self.ground_truth
            if (key == "chatgpt_caption" or key == "Chatgpt_caption"):
                return self.chatgpt_caption
            if (key == "llama2_caption" or key == "Llama2_caption"):
                return self.llama2_caption
            if (key == "wer_gpt"):
                return self.wer_gpt
            if (key == "bleu_gpt"):
                return self.bleu_gpt
            if (key == "rouge1_gpt"):
                return self.rouge1_gpt
            if (key == "rouge2_gpt"):
                return self.rouge2_gpt
            if (key == "rougeL_gpt"):
                return self.rougeL_gpt
            if (key == "rougeLsum_gpt"):
                return self.rougeLsum_gpt
            if (key == "wer_llama2"):
                return self.wer_llama2
            if (key == "bleu_llama2"):
                return self.bleu_llama2
            if (key == "rouge1_llama2"):
                return self.rouge1_llama2
            if (key == "rouge2_llama2"):
                return self.rouge2_llama2
            if (key == "rougeL_llama2"):
                return self.rougeL_llama2
            if (key == "rougeLsum_llama2"):
                return self.rougeLsum_llama2
            if (key == "wer_asr"):
                return self.wer_asr
            if (key == "bleu_asr"):
                return self.bleu_asr
            if (key == "rouge1_asr"):
                return self.rouge1_asr
            if (key == "rouge2_asr"):
                return self.rouge2_asr
            if (key == "rougeL_asr"):
                return self.rougeL_asr
            if (key == "rougeLsum_asr"):
                return self.rougeLsum_asr

            raise KeyError

        if isinstance(key, int):
            if key == 0:
                return self.video_id
            if key == 1:
                return self.cc_status
            if key == 2:
                return self.note
            if key == 3:
                return self.category
            if key == 4:
                return self.label
            if key == 5:
                return self.metadata
            if key == 6:
                return self.youtube_caption
            if key == 7:
                return self.ground_truth
            if key == 8:
                return self.chatgpt_caption
            if key == 9:
                return self.llama2_caption
            if key == 10:
                return self.wer_gpt
            if key == 11:
                return self.bleu_gpt
            if key == 12:
                return self.rouge1_gpt
            if key == 13:
                return self.rouge2_gpt
            if key == 14:
                return self.rougeL_gpt
            if key == 15:
                return self.rougeLsum_gpt
            if key == 16:
                return self.wer_llama2
            if key == 17:
                return self.bleu_llama2
            if key == 18:
                return self.rouge1_llama2
            if key == 19:
                return self.rouge2_llama2
            if key == 20:
                return self.rougeL_llama2
            if key == 21:
                return self.rougeLsum_llama2
            if key == 22:
                return self.wer_asr
            if key == 23:
                return self.bleu_asr
            if key == 24:
                return self.rouge1_asr
            if key == 25:
                return self.rouge2_asr
            if key == 26:
                return self.rougeL_asr
            if key == 27:
                return self.rougeLsum_asr

            raise KeyError

        raise KeyError

    def __setitem__(
        self, key: str | int,
        value: str | int | float | None,
    ) -> None:
        if isinstance(key, str):
            if (key == "video_id" or key == "Video_Id" or key == "Video_ID"):
                self.video_id = str_or_none(value)
                return None
            if (key == "cc_status" or key == "CC_Status"):
                self.cc_status = int_or_none(value)
                return None
            if (key == "note" or key == "Note"):
                self.note = str_or_none(value)
                return None
            if (key == "category" or key == "Category"):
                self.category = str_or_none(value)
                return None
            if (key == "label" or key == "Label"):
                self.label = int_or_none(value)
                return None
            if (key == "metadata" or key == "Metadata"):
                self.metadata = str_or_none(value)
                return None
            if (key == "youtube_caption"):
                self.youtube_caption = str_or_none(value)
                return None
            if (key == "ground_truth" or key == "Ground_Truth"):
                self.ground_truth = str_or_none(value)
                return None
            if (key == "chatgpt_caption" or key == "Chatgpt_caption"):
                self.chatgpt_caption = str_or_none(value)
                return None
            if (key == "llama2_caption" or key == "Llama2_caption"):
                self.llama2_caption = str_or_none(value)
                return None
            if (key == "wer_gpt"):
                self.wer_gpt = float_or_none(value)
                return None
            if (key == "bleu_gpt"):
                self.bleu_gpt = float_or_none(value)
                return None
            if (key == "rouge1_gpt"):
                self.rouge1_gpt = float_or_none(value)
                return None
            if (key == "rouge2_gpt"):
                self.rouge2_gpt = float_or_none(value)
                return None
            if (key == "rougeL_gpt"):
                self.rougeL_gpt = float_or_none(value)
                return None
            if (key == "rougeLsum_gpt"):
                self.rougeLsum_gpt = float_or_none(value)
                return None
            if (key == "wer_llama2"):
                self.wer_llama2 = float_or_none(value)
                return None
            if (key == "bleu_llama2"):
                self.bleu_llama2 = float_or_none(value)
                return None
            if (key == "rouge1_llama2"):
                self.rouge1_llama2 = float_or_none(value)
                return None
            if (key == "rouge2_llama2"):
                self.rouge2_llama2 = float_or_none(value)
                return None
            if (key == "rougeL_llama2"):
                self.rougeL_llama2 = float_or_none(value)
                return None
            if (key == "rougeLsum_llama2"):
                self.rougeLsum_llama2 = float_or_none(value)
                return None
            if (key == "wer_asr"):
                self.wer_asr = float_or_none(value)
                return None
            if (key == "bleu_asr"):
                self.bleu_asr = float_or_none(value)
                return None
            if (key == "rouge1_asr"):
                self.rouge1_asr = float_or_none(value)
                return None
            if (key == "rouge2_asr"):
                self.rouge2_asr = float_or_none(value)
                return None
            if (key == "rougeL_asr"):
                self.rougeL_asr = float_or_none(value)
                return None
            if (key == "rougeLsum_asr"):
                self.rougeLsum_asr = float_or_none(value)
                return None

            raise KeyError

        if isinstance(key, int):
            if (key == 0):
                self.video_id = str_or_none(value)
                return None
            if (key == 1):
                self.cc_status = int_or_none(value)
                return None
            if (key == 2):
                self.note = str_or_none(value)
                return None
            if (key == 3):
                self.category = str_or_none(value)
                return None
            if (key == 4):
                self.label = int_or_none(value)
                return None
            if (key == 5):
                self.metadata = str_or_none(value)
                return None
            if (key == 6):
                self.youtube_caption = str_or_none(value)
                return None
            if (key == 7):
                self.ground_truth = str_or_none(value)
                return None
            if (key == 8):
                self.chatgpt_caption = str_or_none(value)
                return None
            if (key == 9):
                self.llama2_caption = str_or_none(value)
                return None
            if (key == 10):
                self.wer_gpt = float_or_none(value)
                return None
            if (key == 11):
                self.bleu_gpt = float_or_none(value)
                return None
            if (key == 12):
                self.rouge1_gpt = float_or_none(value)
                return None
            if (key == 13):
                self.rouge2_gpt = float_or_none(value)
                return None
            if (key == 14):
                self.rougeL_gpt = float_or_none(value)
                return None
            if (key == 15):
                self.rougeLsum_gpt = float_or_none(value)
                return None
            if (key == 16):
                self.wer_llama2 = float_or_none(value)
                return None
            if (key == 17):
                self.bleu_llama2 = float_or_none(value)
                return None
            if (key == 18):
                self.rouge1_llama2 = float_or_none(value)
                return None
            if (key == 19):
                self.rouge2_llama2 = float_or_none(value)
                return None
            if (key == 20):
                self.rougeL_llama2 = float_or_none(value)
                return None
            if (key == 21):
                self.rougeLsum_llama2 = float_or_none(value)
                return None
            if (key == 22):
                self.wer_asr = float_or_none(value)
                return None
            if (key == 23):
                self.bleu_asr = float_or_none(value)
                return None
            if (key == 24):
                self.rouge1_asr = float_or_none(value)
                return None
            if (key == 25):
                self.rouge2_asr = float_or_none(value)
                return None
            if (key == 26):
                self.rougeL_asr = float_or_none(value)
                return None
            if (key == 27):
                self.rougeLsum_asr = float_or_none(value)
                return None

            raise KeyError

        raise KeyError

    def __delitem__(self, key: str | int) -> None:
        if isinstance(key, str):
            if (key == "video_id" or key == "Video_Id" or key == "Video_ID"):
                self.video_id = None
                return None
            if (key == "cc_status" or key == "CC_Status"):
                self.cc_status = None
                return None
            if (key == "note" or key == "Note"):
                self.note = None
                return None
            if (key == "category" or key == "Category"):
                self.category = None
                return None
            if (key == "label" or key == "Label"):
                self.label = None
                return None
            if (key == "metadata" or key == "Metadata"):
                self.metadata = None
                return None
            if (key == "youtube_caption"):
                self.youtube_caption = None
                return None
            if (key == "ground_truth" or key == "Ground_Truth"):
                self.ground_truth = None
                return None
            if (key == "chatgpt_caption" or key == "Chatgpt_caption"):
                self.chatgpt_caption = None
                return None
            if (key == "llama2_caption" or key == "Llama2_caption"):
                self.llama2_caption = None
                return None
            if (key == "wer_gpt"):
                self.wer_gpt = None
                return None
            if (key == "bleu_gpt"):
                self.bleu_gpt = None
                return None
            if (key == "rouge1_gpt"):
                self.rouge1_gpt = None
                return None
            if (key == "rouge2_gpt"):
                self.rouge2_gpt = None
                return None
            if (key == "rougeL_gpt"):
                self.rougeL_gpt = None
                return None
            if (key == "rougeLsum_gpt"):
                self.rougeLsum_gpt = None
                return None
            if (key == "wer_llama2"):
                self.wer_llama2 = None
                return None
            if (key == "bleu_llama2"):
                self.bleu_llama2 = None
                return None
            if (key == "rouge1_llama2"):
                self.rouge1_llama2 = None
                return None
            if (key == "rouge2_llama2"):
                self.rouge2_llama2 = None
                return None
            if (key == "rougeL_llama2"):
                self.rougeL_llama2 = None
                return None
            if (key == "rougeLsum_llama2"):
                self.rougeLsum_llama2 = None
                return None
            if (key == "wer_asr"):
                self.wer_asr = None
                return None
            if (key == "bleu_asr"):
                self.bleu_asr = None
                return None
            if (key == "rouge1_asr"):
                self.rouge1_asr = None
                return None
            if (key == "rouge2_asr"):
                self.rouge2_asr = None
                return None
            if (key == "rougeL_asr"):
                self.rougeL_asr = None
                return None
            if (key == "rougeLsum_asr"):
                self.rougeLsum_asr = None
                return None

            raise KeyError

        if isinstance(key, int):
            if key == 0:
                self.video_id = None
                return None
            if key == 1:
                self.cc_status = None
                return None
            if key == 2:
                self.note = None
                return None
            if key == 3:
                self.category = None
                return None
            if key == 4:
                self.label = None
                return None
            if key == 5:
                self.metadata = None
                return None
            if key == 6:
                self.youtube_caption = None
                return None
            if key == 7:
                self.ground_truth = None
                return None
            if key == 8:
                self.chatgpt_caption = None
                return None
            if key == 9:
                self.llama2_caption = None
                return None
            if key == 10:
                self.wer_gpt = None
                return None
            if key == 11:
                self.bleu_gpt = None
                return None
            if key == 12:
                self.rouge1_gpt = None
                return None
            if key == 13:
                self.rouge2_gpt = None
                return None
            if key == 14:
                self.rougeL_gpt = None
                return None
            if key == 15:
                self.rougeLsum_gpt = None
                return None
            if key == 16:
                self.wer_llama2 = None
                return None
            if key == 17:
                self.bleu_llama2 = None
                return None
            if key == 18:
                self.rouge1_llama2 = None
                return None
            if key == 19:
                self.rouge2_llama2 = None
                return None
            if key == 20:
                self.rougeL_llama2 = None
                return None
            if key == 21:
                self.rougeLsum_llama2 = None
                return None
            if key == 22:
                self.wer_asr = None
                return None
            if key == 23:
                self.bleu_asr = None
                return None
            if key == 24:
                self.rouge1_asr = None
                return None
            if key == 25:
                self.rouge2_asr = None
                return None
            if key == 26:
                self.rougeL_asr = None
                return None
            if key == 27:
                self.rougeLsum_asr = None
                return None

            raise KeyError

        raise KeyError

    # There is a notion that an __iter__ function must return an Iterator (self)
    # instance where self implements a __next__ function. Returning a simple
    # Generator function is just as valid as returning an Iterator instance.
    def __iter__(self) -> Generator[str | int | float | None]:
        yield self.video_id
        yield self.cc_status
        yield self.note
        yield self.category
        yield self.label
        yield self.metadata
        yield self.youtube_caption
        yield self.ground_truth
        yield self.chatgpt_caption
        yield self.llama2_caption
        yield self.wer_gpt
        yield self.bleu_gpt
        yield self.rouge1_gpt
        yield self.rouge2_gpt
        yield self.rougeL_gpt
        yield self.rougeLsum_gpt
        yield self.wer_llama2
        yield self.bleu_llama2
        yield self.rouge1_llama2
        yield self.rouge2_llama2
        yield self.rougeL_llama2
        yield self.rougeLsum_llama2
        yield self.wer_asr
        yield self.bleu_asr
        yield self.rouge1_asr
        yield self.rouge2_asr
        yield self.rougeL_asr
        yield self.rougeLsum_asr

    def iterate_and_remove_tabs_newlines(
        self,
        skip_asr_results: bool = False,
    ) -> Generator[str | int | float | None]:
        try:
            skip_asr_results = bool(skip_asr_results)
        except (TypeError, ValueError):
            skip_asr_results = False

        items: list[str | int | float | None]
        if skip_asr_results:
            items = [
                self.video_id,
                self.cc_status,
                self.note,
                self.category,
                self.label,
                self.metadata,
                self.youtube_caption,
                self.ground_truth,
                self.chatgpt_caption,
                self.llama2_caption,
                self.wer_gpt,
                self.bleu_gpt,
                self.rouge1_gpt,
                self.rouge2_gpt,
                self.rougeL_gpt,
                self.rougeLsum_gpt,
                self.wer_llama2,
                self.bleu_llama2,
                self.rouge1_llama2,
                self.rouge2_llama2,
                self.rougeL_llama2,
                self.rougeLsum_llama2,
            ]
        else:
            items = [
                self.video_id,
                self.cc_status,
                self.note,
                self.category,
                self.label,
                self.metadata,
                self.youtube_caption,
                self.ground_truth,
                self.chatgpt_caption,
                self.llama2_caption,
                self.wer_gpt,
                self.bleu_gpt,
                self.rouge1_gpt,
                self.rouge2_gpt,
                self.rougeL_gpt,
                self.rougeLsum_gpt,
                self.wer_llama2,
                self.bleu_llama2,
                self.rouge1_llama2,
                self.rouge2_llama2,
                self.rougeL_llama2,
                self.rougeLsum_llama2,
                self.wer_asr,
                self.bleu_asr,
                self.rouge1_asr,
                self.rouge2_asr,
                self.rougeL_asr,
                self.rougeLsum_asr,
            ]

        string: str
        item: str | int | float | None
        for item in items:
            if isinstance(item, str):
                string = str(item)

                # Removes tabs.
                string = string.replace("\t", " ")

                # Removes newlines.
                string = " ".join(string.splitlines(keepends=False))

                yield string

            elif isinstance(item, int | float):
                yield item

            else:
                yield None

    def __len__(self) -> int:
        return 28

    def __str__(self) -> str:
        return (
            f"{str_or_empty_if_none(self.video_id)}\t"
            f"{str_or_empty_if_none(self.cc_status)}\t"
            f"{str_or_empty_if_none(self.note)}\t"
            f"{str_or_empty_if_none(self.category)}\t"
            f"{str_or_empty_if_none(self.label)}\t"
            f"{str_or_empty_if_none(self.metadata)}\t"
            f"{str_or_empty_if_none(self.youtube_caption)}\t"
            f"{str_or_empty_if_none(self.ground_truth)}\t"
            f"{str_or_empty_if_none(self.chatgpt_caption)}\t"
            f"{str_or_empty_if_none(self.llama2_caption)}\t"
            f"{str_or_empty_if_none(self.wer_gpt)}\t"
            f"{str_or_empty_if_none(self.bleu_gpt)}\t"
            f"{str_or_empty_if_none(self.rouge1_gpt)}\t"
            f"{str_or_empty_if_none(self.rouge2_gpt)}\t"
            f"{str_or_empty_if_none(self.rougeL_gpt)}\t"
            f"{str_or_empty_if_none(self.rougeLsum_gpt)}\t"
            f"{str_or_empty_if_none(self.wer_llama2)}\t"
            f"{str_or_empty_if_none(self.bleu_llama2)}\t"
            f"{str_or_empty_if_none(self.rouge1_llama2)}\t"
            f"{str_or_empty_if_none(self.rouge2_llama2)}\t"
            f"{str_or_empty_if_none(self.rougeL_llama2)}\t"
            f"{str_or_empty_if_none(self.rougeLsum_llama2)}\t"
            f"{str_or_empty_if_none(self.wer_asr)}\t"
            f"{str_or_empty_if_none(self.bleu_asr)}\t"
            f"{str_or_empty_if_none(self.rouge1_asr)}\t"
            f"{str_or_empty_if_none(self.rouge2_asr)}\t"
            f"{str_or_empty_if_none(self.rougeL_asr)}\t"
            f"{str_or_empty_if_none(self.rougeLsum_asr)}"
        )


def _read_tsv(
    records: list[Record], # Reference/pointer to a mutable list.
    tsv_file_path: str | Path = Path(DEFAULT_PATH),
) -> None:
    tsv_file: TextIO
    with open(
        file=tsv_file_path,
        mode="rt",
        buffering=-1,
        encoding="utf-8",
        errors=None,
        newline="",
        closefd=True,
        opener=None,
    ) as tsv_file:
        tsv_reader = reader(
            tsv_file,
            dialect="excel-tab",
            delimiter="\t",
            quotechar=None,
            escapechar=None,
            doublequote=False,
            skipinitialspace=False,
            lineterminator="\r\n",
            quoting=QUOTE_NONE,
            strict=True,
        )

        record_length: int = len(Record())

        record: Record
        i: int
        tsv_row: list[str]
        for i, tsv_row in enumerate(tsv_reader):
            # Skips the header row.
            if i == 0:
                continue

            # Bulks up tsv_row to prevent "IndexError: list index out of range".
            while len(tsv_row) < record_length:
                tsv_row.append("")

            record = Record(
                video_id=str_or_none_if_empty(tsv_row[0]),
                cc_status=int_or_none(tsv_row[1]),
                note=str_or_none_if_empty(tsv_row[2]),
                category=str_or_none_if_empty(tsv_row[3]),
                label=int_or_none(tsv_row[4]),
                metadata=str_or_none_if_empty(tsv_row[5]),
                youtube_caption=str_or_none_if_empty(tsv_row[6]),
                ground_truth=str_or_none_if_empty(tsv_row[7]),
                chatgpt_caption=str_or_none_if_empty(tsv_row[8]),
                llama2_caption=str_or_none_if_empty(tsv_row[9]),
                wer_gpt=float_or_none(tsv_row[10]),
                bleu_gpt=float_or_none(tsv_row[11]),
                rouge1_gpt=float_or_none(tsv_row[12]),
                rouge2_gpt=float_or_none(tsv_row[13]),
                rougeL_gpt=float_or_none(tsv_row[14]),
                rougeLsum_gpt=float_or_none(tsv_row[15]),
                wer_llama2=float_or_none(tsv_row[16]),
                bleu_llama2=float_or_none(tsv_row[17]),
                rouge1_llama2=float_or_none(tsv_row[18]),
                rouge2_llama2=float_or_none(tsv_row[19]),
                rougeL_llama2=float_or_none(tsv_row[20]),
                rougeLsum_llama2=float_or_none(tsv_row[21]),
                wer_asr=float_or_none(tsv_row[22]),
                bleu_asr=float_or_none(tsv_row[23]),
                rouge1_asr=float_or_none(tsv_row[24]),
                rouge2_asr=float_or_none(tsv_row[25]),
                rougeL_asr=float_or_none(tsv_row[26]),
                rougeLsum_asr=float_or_none(tsv_row[27]),
            )

            # Mutates the list.
            records.append(record)

    return None


def _write_tsv(
    records: list[Record],
    tsv_file_path: str | Path = Path(DEFAULT_PATH),
    skip_asr_results: bool = False,
) -> None:
    try:
        skip_asr_results = bool(skip_asr_results)
    except (TypeError, ValueError):
        skip_asr_results = False

    tsv_file: TextIO
    with open(
        file=tsv_file_path,
        mode="wt",
        buffering=-1,
        encoding="utf-8",
        errors=None,
        newline="",
        closefd=True,
        opener=None,
    ) as tsv_file:
        tsv_writer = writer(
            tsv_file,
            dialect="excel-tab",
            delimiter="\t",
            quotechar=None,
            escapechar=None,
            doublequote=False,
            skipinitialspace=False,
            lineterminator="\r\n",
            quoting=QUOTE_NONE,
            strict=True,
        )

        header_row: list[str]
        if skip_asr_results:
            header_row = [
                "Video_Id",
                "CC_Status",
                "Note",
                "Category",
                "Label",
                "Metadata",
                "youtube_caption",
                "Ground_Truth",
                "Chatgpt_caption",
                "Llama2_caption",
                "wer_gpt",
                "bleu_gpt",
                "rouge1_gpt",
                "rouge2_gpt",
                "rougeL_gpt",
                "rougeLsum_gpt",
                "wer_llama2",
                "bleu_llama2",
                "rouge1_llama2",
                "rouge2_llama2",
                "rougeL_llama2",
                "rougeLsum_llama2",
            ]
        else:
            header_row = [
                "Video_Id",
                "CC_Status",
                "Note",
                "Category",
                "Label",
                "Metadata",
                "youtube_caption",
                "Ground_Truth",
                "Chatgpt_caption",
                "Llama2_caption",
                "wer_gpt",
                "bleu_gpt",
                "rouge1_gpt",
                "rouge2_gpt",
                "rougeL_gpt",
                "rougeLsum_gpt",
                "wer_llama2",
                "bleu_llama2",
                "rouge1_llama2",
                "rouge2_llama2",
                "rougeL_llama2",
                "rougeLsum_llama2",
                "wer_asr",
                "bleu_asr",
                "rouge1_asr",
                "rouge2_asr",
                "rougeL_asr",
                "rougeLsum_asr",
            ]

        # Writes the header row.
        tsv_writer.writerow(header_row)

        record: Record
        for record in records:
            # The None type is written as an empty string. All other non-string
            # data are stringified with str() before being written. Note that an
            # empty string is eventually read back as an empty string, not as
            # None type. See the str_or_None_if_empty() helper function, it will
            # take any string as input and then return the string as it is or
            # return None instead if the string is empty.
            tsv_writer.writerow(record.iterate_and_remove_tabs_newlines(
                skip_asr_results=skip_asr_results,
            ))

    return None


class RecordList(MutableSequence):
    # MutableSequence must implement methods __init__, __getitem__, __setitem__,
    # __delitem__, __len__, and insert.

    def load(
        self,
        tsv_file_path: str | Path = Path(DEFAULT_PATH),
        clear: bool = False,
    ) -> None:
        try:
            clear = bool(clear)
        except (TypeError, ValueError):
            clear = False
        if clear:
            # Clears the mutable list.
            self.records.clear()

        # Mutates the list.
        _read_tsv(
            records=self.records,
            tsv_file_path=tsv_file_path,
        )

        return None

    def save(
        self,
        tsv_file_path: str | Path = Path(DEFAULT_PATH),
        skip_asr_results: bool = False,
    ) -> None:
        _write_tsv(
            records=self.records,
            tsv_file_path=tsv_file_path,
            skip_asr_results=skip_asr_results,
        )

        return None

    def __init__(
        self,
        tsv_file_path: str | Path | None = Path(DEFAULT_PATH),
    ) -> None:
        # Creates a reference/pointer to a new empty list (mutable).
        self.records: list[Record] = []

        if isinstance(tsv_file_path, str | Path):
            self.load(tsv_file_path=tsv_file_path, clear=False)

        super().__init__()
        return None

    def __getitem__(self, index: int) -> Record:
        return self.records[index]

    def __setitem__(self, index: int, value: Record) -> None:
        # Mutates the list.
        self.records[index] = value
        return None

    def __delitem__(self, index: int) -> None:
        # Mutates the list.
        self.records.pop(index)
        return None

    def __len__(self) -> int:
        return len(self.records)

    def insert(self, index: int, value: Record) -> None:
        # Mutates the list.
        self.records.insert(index, value)
        return None

    def __str__(self) -> str:
        return (
            "Video_Id\t"
            "CC_Status\t"
            "Note\t"
            "Category\t"
            "Label\t"
            "Metadata\t"
            "youtube_caption\t"
            "Ground_Truth\t"
            "Chatgpt_caption\t"
            "Llama2_caption\t"
            "wer_gpt\t"
            "bleu_gpt\t"
            "rouge1_gpt\t"
            "rouge2_gpt\t"
            "rougeL_gpt\t"
            "rougeLsum_gpt\t"
            "wer_llama2\t"
            "bleu_llama2\t"
            "rouge1_llama2\t"
            "rouge2_llama2\t"
            "rougeL_llama2\t"
            "rougeLsum_llama2\t"
            "wer_asr\t"
            "bleu_asr\t"
            "rouge1_asr\t"
            "rouge2_asr\t"
            "rougeL_asr\t"
            "rougeLsum_asr"
        )


# TODO(wathne): Test this function (it should work).
def get_records(
    records: list[Record] | None = None, # Reference/pointer to a mutable list.
    tsv_file_path: str | Path = Path(DEFAULT_PATH),
) -> None | list[Record]:
    if isinstance(records, list):
        # Clears the mutable list.
        records.clear()

        # Mutates the list.
        _read_tsv(
            records=records,
            tsv_file_path=tsv_file_path,
        )

        return None

    # Creates a reference/pointer to a new empty list (mutable) if there is
    # none.
    elif records is None:
        records = []

        # Mutates the new list.
        _read_tsv(
            records=records,
            tsv_file_path=tsv_file_path,
        )

        # Returns a reference/pointer to the new list.
        return records

    raise TypeError


# TODO(wathne): Test this function (it should work).
def set_records(
    records: list[Record],
    tsv_file_path: str | Path = Path(DEFAULT_PATH),
    skip_asr_results: bool = False,
) -> None:
    _write_tsv(
        records=records,
        tsv_file_path=tsv_file_path,
        skip_asr_results=skip_asr_results,
    )

    return None


def clear_results_from_records(
    records: RecordList,
) -> None:
    record: Record # Reference/pointer to a mutable Record.
    for record in records:
        # Mutates the Record.
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

    return None


def clear_llm_captions_from_records(
    records: RecordList,
) -> None:
    record: Record # Reference/pointer to a mutable Record.
    for record in records:
        # Mutates the Record.
        record.chatgpt_caption = None
        record.llama2_caption = None

    return None


def clear_youtube_caption_from_records(
    records: RecordList,
) -> None:
    record: Record # Reference/pointer to a mutable Record.
    for record in records:
        # Mutates the Record.
        record.youtube_caption = None

    return None


def initialize_records_from_dhh_records_and_sources(
    records: RecordList, # Reference/pointer to a mutable RecordList.
) -> None:
    # Clears the mutable RecordList.
    records.clear()

    dhh_records: DHHRecordList = DHHRecordList(
        tsv_file_path=Path(DHH_RECORDS_PATH),
    )
    dhh_sources: DHHSourceList = DHHSourceList(
        tsv_file_path=Path(DHH_SOURCES_PATH),
    )

    video_id: str | None
    cc_status: int | None
    note: str | None
    category: str
    label: int
    record: Record
    dhh_source: DHHSource
    dhh_record: DHHRecord
    for dhh_record in dhh_records:
        video_id = None
        cc_status = None
        note = None
        category = str(dhh_record.category)
        label = int(dhh_record.label)

        # Gets the correct video_id by matching category and label from DHH
        # records with category and label from DHH sources.
        for dhh_source in dhh_sources:
            # (category == dhh_source.category AND label == dhh_source.label)
            # The following negation logic is equivalent and should be faster
            # because a string comparison returns early for non-equal strings.
            if not (
                category != str(dhh_source.category) or
                label != int(dhh_source.label)
            ):
                video_id = str_or_none_if_empty(dhh_source.video_id)
                cc_status = int_or_none(dhh_source.cc_status)
                note = str_or_none_if_empty(dhh_source.note)

        # The DHH records and sources are bijective. The video ID must be found.
        if video_id is None:
            raise Exception(
                "The DHH records and sources are bijective. The video ID must"
                " be found.")

        record = Record(
            video_id=video_id,
            cc_status=cc_status,
            note=note,
            category=category,
            label=label,
            metadata=None,
            youtube_caption=str(dhh_record.youtube_caption),
            ground_truth=str(dhh_record.ground_truth),
            chatgpt_caption=str(dhh_record.chatgpt_caption),
            llama2_caption=str(dhh_record.llama2_caption),
            wer_gpt=float(dhh_record.wer_gpt),
            bleu_gpt=float(dhh_record.bleu_gpt),
            rouge1_gpt=float(dhh_record.rouge1_gpt),
            rouge2_gpt=float(dhh_record.rouge2_gpt),
            rougeL_gpt=float(dhh_record.rougeL_gpt),
            rougeLsum_gpt=float(dhh_record.rougeLsum_gpt),
            wer_llama2=float(dhh_record.wer_llama2),
            bleu_llama2=float(dhh_record.bleu_llama2),
            rouge1_llama2=float(dhh_record.rouge1_llama2),
            rouge2_llama2=float(dhh_record.rouge2_llama2),
            rougeL_llama2=float(dhh_record.rougeL_llama2),
            rougeLsum_llama2=float(dhh_record.rougeLsum_llama2),
            wer_asr=None,
            bleu_asr=None,
            rouge1_asr=None,
            rouge2_asr=None,
            rougeL_asr=None,
            rougeLsum_asr=None,
        )

        # Mutates the RecordList.
        records.append(record)

    return None
