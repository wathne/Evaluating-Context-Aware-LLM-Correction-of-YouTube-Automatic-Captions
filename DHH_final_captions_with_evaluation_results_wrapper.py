# Read-only wrapper for "DHH_final_captions_with_evaluation_results.tsv".

# For convenience, let "DHH" be a shortened reference to the following study:
# "Empowering the Deaf and Hard of Hearing Community: Enhancing Video Captions
# Using Large Language Models".
# https://arxiv.org/abs/2412.00342
# https://github.com/monikabhole001/Improving-the-Quality-of-Video-Captions-for-the-DHH-Community-Using-LLM


from collections.abc import Mapping
from collections.abc import Sequence
from csv import QUOTE_NONE
from csv import reader
from pathlib import Path
from typing import Generator
from typing import TextIO


DIR: str = "./DHH_records_and_sources/"
DEFAULT_PATH: str = DIR + "DHH_final_captions_with_evaluation_results.tsv"


class Record(Mapping):
    # Mapping must implement methods __getitem__, __iter__, and __len__.

    def __init__(
        self,
        file: str,
        label: int,
        youtube_caption: str,
        category: str,
        chatgpt_caption: str,
        ground_truth: str,
        llama2_caption: str,
        wer_gpt: float,
        bleu_gpt: float,
        rouge1_gpt: float,
        rouge2_gpt: float,
        rougeL_gpt: float,
        rougeLsum_gpt: float,
        wer_llama2: float,
        bleu_llama2: float,
        rouge1_llama2: float,
        rouge2_llama2: float,
        rougeL_llama2: float,
        rougeLsum_llama2: float,
    ) -> None:
        self.file: str = file
        self.label: int = label
        self.youtube_caption: str = youtube_caption
        self.category: str = category
        self.chatgpt_caption: str = chatgpt_caption
        self.ground_truth: str = ground_truth
        self.llama2_caption: str = llama2_caption
        self.wer_gpt: float = wer_gpt
        self.bleu_gpt: float = bleu_gpt
        self.rouge1_gpt: float = rouge1_gpt
        self.rouge2_gpt: float = rouge2_gpt
        self.rougeL_gpt: float = rougeL_gpt
        self.rougeLsum_gpt: float = rougeLsum_gpt
        self.wer_llama2: float = wer_llama2
        self.bleu_llama2: float = bleu_llama2
        self.rouge1_llama2: float = rouge1_llama2
        self.rouge2_llama2: float = rouge2_llama2
        self.rougeL_llama2: float = rougeL_llama2
        self.rougeLsum_llama2: float = rougeLsum_llama2

        super().__init__()
        return None

    def __getitem__(self, key: str | int) -> str | int | float:
        if isinstance(key, str):
            if (key == "file" or key == "File"):
                return self.file
            if (key == "label" or key == "Label"):
                return self.label
            if (key == "youtube_caption"):
                return self.youtube_caption
            if (key == "category" or key == "Category"):
                return self.category
            if (key == "chatgpt_caption" or key == "Chatgpt_caption"):
                return self.chatgpt_caption
            if (key == "ground_truth" or key == "Ground_Truth"):
                return self.ground_truth
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

            raise KeyError

        if isinstance(key, int):
            if key == 0:
                return self.file
            if key == 1:
                return self.label
            if key == 2:
                return self.youtube_caption
            if key == 3:
                return self.category
            if key == 4:
                return self.chatgpt_caption
            if key == 5:
                return self.ground_truth
            if key == 6:
                return self.llama2_caption
            if key == 7:
                return self.wer_gpt
            if key == 8:
                return self.bleu_gpt
            if key == 9:
                return self.rouge1_gpt
            if key == 10:
                return self.rouge2_gpt
            if key == 11:
                return self.rougeL_gpt
            if key == 12:
                return self.rougeLsum_gpt
            if key == 13:
                return self.wer_llama2
            if key == 14:
                return self.bleu_llama2
            if key == 15:
                return self.rouge1_llama2
            if key == 16:
                return self.rouge2_llama2
            if key == 17:
                return self.rougeL_llama2
            if key == 18:
                return self.rougeLsum_llama2

            raise KeyError

        raise KeyError

    # There is a notion that an __iter__ function must return an Iterator (self)
    # instance where self implements a __next__ function. Returning a simple
    # Generator function is just as valid as returning an Iterator instance.
    def __iter__(self) -> Generator[str | int | float]:
        yield self.file
        yield self.label
        yield self.youtube_caption
        yield self.category
        yield self.chatgpt_caption
        yield self.ground_truth
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

    def __len__(self) -> int:
        return 19

    def __str__(self) -> str:
        return (
            f"{self.file}\t"
            f"{self.label}\t"
            f"{self.youtube_caption}\t"
            f"{self.category}\t"
            f"{self.chatgpt_caption}\t"
            f"{self.ground_truth}\t"
            f"{self.llama2_caption}\t"
            f"{self.wer_gpt}\t"
            f"{self.bleu_gpt}\t"
            f"{self.rouge1_gpt}\t"
            f"{self.rouge2_gpt}\t"
            f"{self.rougeL_gpt}\t"
            f"{self.rougeLsum_gpt}\t"
            f"{self.wer_llama2}\t"
            f"{self.bleu_llama2}\t"
            f"{self.rouge1_llama2}\t"
            f"{self.rouge2_llama2}\t"
            f"{self.rougeL_llama2}\t"
            f"{self.rougeLsum_llama2}"
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

        record: Record
        i: int
        tsv_row: list[str]
        for i, tsv_row in enumerate(tsv_reader):
            # Skips the header row.
            if i == 0:
                continue

            record = Record(
                file=tsv_row[0],
                label=int(tsv_row[1]),
                youtube_caption=tsv_row[2],
                category=tsv_row[3],
                chatgpt_caption=tsv_row[4],
                ground_truth=tsv_row[5],
                llama2_caption=tsv_row[6],
                wer_gpt=float(tsv_row[7]),
                bleu_gpt=float(tsv_row[8]),
                rouge1_gpt=float(tsv_row[9]),
                rouge2_gpt=float(tsv_row[10]),
                rougeL_gpt=float(tsv_row[11]),
                rougeLsum_gpt=float(tsv_row[12]),
                wer_llama2=float(tsv_row[13]),
                bleu_llama2=float(tsv_row[14]),
                rouge1_llama2=float(tsv_row[15]),
                rouge2_llama2=float(tsv_row[16]),
                rougeL_llama2=float(tsv_row[17]),
                rougeLsum_llama2=float(tsv_row[18]),
            )

            # Mutates the list.
            records.append(record)

    return None


class RecordList(Sequence):
    # Sequence must implement methods __init__, __getitem__, and __len__.

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

    def __len__(self) -> int:
        return len(self.records)

    def __str__(self) -> str:
        return (
            "File\t"
            "Label\t"
            "youtube_caption\t"
            "Category\t"
            "Chatgpt_caption\t"
            "Ground_Truth\t"
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
            "rougeLsum_llama2"
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
