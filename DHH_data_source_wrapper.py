# Read-only wrapper for "DHH_data_source_corrected.tsv".

# For convenience, let "DHH" be a shortened reference to the following study:
# "Empowering the Deaf and Hard of Hearing Community: Enhancing Video Captions
# Using Large Language Models".
# https://arxiv.org/abs/2412.00342
# https://github.com/monikabhole001/Improving-the-Quality-of-Video-Captions-for-the-DHH-Community-Using-LLM


from collections.abc import Mapping
from collections.abc import Sequence
from csv import QUOTE_NONE
from csv import reader
from helpers import int_or_none
from helpers import str_or_empty_if_none
from helpers import str_or_none_if_empty
from pathlib import Path
from typing import Generator
from typing import TextIO


DIR: str = "./DHH_records_and_sources/"
DEFAULT_PATH: str = DIR + "DHH_data_source_corrected.tsv"


class Record(Mapping):
    # Mapping must implement methods __getitem__, __iter__, and __len__.

    def __init__(
        self,
        link: str,
        label: int,
        category: str,
        source: str,
        video_id: str,
        cc_status: int | None = None,
        note: str | None = None,
    ) -> None:
        self.link: str = link
        self.label: int = label
        self.category: str = category
        self.source: str = source
        self.video_id: str = video_id
        self.cc_status: int | None = cc_status
        self.note: str | None = note

        super().__init__()
        return None

    def __getitem__(self, key: str | int) -> str | int | None:
        if isinstance(key, str):
            if (key == "link" or key == "Link"):
                return self.link
            if (key == "label" or key == "Label"):
                return self.label
            if (key == "category" or key == "Category"):
                return self.category
            if (key == "source" or key == "Source"):
                return self.source
            if (key == "video_id" or key == "Video_Id" or key == "Video_ID"):
                return self.video_id
            if (key == "cc_status" or key == "CC_Status"):
                return self.cc_status
            if (key == "note" or key == "Note"):
                return self.note

            raise KeyError

        if isinstance(key, int):
            if key == 0:
                return self.link
            if key == 1:
                return self.label
            if key == 2:
                return self.category
            if key == 3:
                return self.source
            if key == 4:
                return self.video_id
            if key == 5:
                return self.cc_status
            if key == 6:
                return self.note

            raise KeyError

        raise KeyError

    # There is a notion that an __iter__ function must return an Iterator (self)
    # instance where self implements a __next__ function. Returning a simple
    # Generator function is just as valid as returning an Iterator instance.
    def __iter__(self) -> Generator[str | int | None]:
        yield self.link
        yield self.label
        yield self.category
        yield self.source
        yield self.video_id
        yield self.cc_status
        yield self.note

    def __len__(self) -> int:
        return 7

    def __str__(self) -> str:
        return (
            f"{self.link},"
            f"{self.label},"
            f"{self.category},"
            f"{self.source},"
            f"{self.video_id},"
            f"{str_or_empty_if_none(self.cc_status)},"
            f"{str_or_empty_if_none(self.note)}"
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
                link=tsv_row[0],
                label=int(tsv_row[1]),
                category=tsv_row[2],
                source=tsv_row[3],
                video_id=tsv_row[4],
                cc_status=int_or_none(tsv_row[5]),
                note=str_or_none_if_empty(tsv_row[6]),
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
        return ("Link,Label,Category,Source,Video_Id,CC_Status,Note")


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
