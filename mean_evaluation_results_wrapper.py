# Read-write wrapper for "mean_evaluation_results*.tsv".

from collections.abc import MutableMapping
from collections.abc import MutableSequence
from csv import QUOTE_NONE
from csv import reader
from csv import writer
from helpers import float_or_none
from helpers import int_or_none
from helpers import str_or_empty_if_none
from helpers import str_or_none
from helpers import str_or_none_if_empty
from pathlib import Path
from typing import Generator
from typing import TextIO


DIR: str = "./captions_with_evaluation_results/"
DEFAULT_PATH: str = DIR + "mean_evaluation_results.tsv"


class Record(MutableMapping):
    # MutableMapping must implement methods __getitem__, __setitem__,
    # __delitem__, __iter__, and __len__.

    def __init__(
        self,
        identifier: int | None = None,
        mean: float | None = None,
        stdev: float | None = None,
        variance: float | None = None,
        description: str | None = None,
    ) -> None:
        self.identifier: int | None = identifier
        self.mean: float | None = mean
        self.stdev: float | None = stdev
        self.variance: float | None = variance
        self.description: str | None = description

        super().__init__()
        return None

    def __getitem__(self, key: str | int) -> str | int | float | None:
        if isinstance(key, str):
            if (key == "identifier" or
                key == "id" or
                key == "Id" or
                key == "ID"
            ):
                return self.identifier
            if key == "mean":
                return self.mean
            if key == "stdev":
                return self.stdev
            if key == "variance":
                return self.variance
            if key == "description":
                return self.description

            raise KeyError

        if isinstance(key, int):
            if key == 0:
                return self.identifier
            if key == 1:
                return self.mean
            if key == 2:
                return self.stdev
            if key == 3:
                return self.variance
            if key == 4:
                return self.description

            raise KeyError

        raise KeyError

    def __setitem__(
        self,
        key: str | int,
        value: str | int | float | None,
    ) -> None:
        if isinstance(key, str):
            if (key == "identifier" or
                key == "id" or
                key == "Id" or
                key == "ID"
            ):
                self.identifier = int_or_none(value)
                return None
            if key == "mean":
                self.mean = float_or_none(value)
                return None
            if key == "stdev":
                self.stdev = float_or_none(value)
                return None
            if key == "variance":
                self.variance = float_or_none(value)
                return None
            if key == "description":
                self.description = str_or_none(value)
                return None

            raise KeyError

        if isinstance(key, int):
            if key == 0:
                self.identifier = int_or_none(value)
                return None
            if key == 1:
                self.mean = float_or_none(value)
                return None
            if key == 2:
                self.stdev = float_or_none(value)
                return None
            if key == 3:
                self.variance = float_or_none(value)
                return None
            if key == 4:
                self.description = str_or_none(value)
                return None

            raise KeyError

        raise KeyError

    def __delitem__(self, key: str | int) -> None:
        if isinstance(key, str):
            if (key == "identifier" or
                key == "id" or
                key == "Id" or
                key == "ID"
            ):
                self.identifier = None
                return None
            if key == "mean":
                self.mean = None
                return None
            if key == "stdev":
                self.stdev = None
                return None
            if key == "variance":
                self.variance = None
                return None
            if key == "description":
                self.description = None
                return None

            raise KeyError

        if isinstance(key, int):
            if key == 0:
                self.identifier = None
                return None
            if key == 1:
                self.mean = None
                return None
            if key == 2:
                self.stdev = None
                return None
            if key == 3:
                self.variance = None
                return None
            if key == 4:
                self.description = None
                return None

            raise KeyError

        raise KeyError

    # There is a notion that an __iter__ function must return an Iterator (self)
    # instance where self implements a __next__ function. Returning a simple
    # Generator function is just as valid as returning an Iterator instance.
    def __iter__(self) -> Generator[str | float | None]:
        yield self.identifier
        yield self.mean
        yield self.stdev
        yield self.variance
        yield self.description

    def iterate_and_remove_tabs_newlines(
        self,
        skip_variance: bool = False,
    ) -> Generator[str | int | float | None]:
        try:
            skip_variance = bool(skip_variance)
        except (TypeError, ValueError):
            skip_variance = False

        items: list[str | float | None]
        if skip_variance:
            items = [
                self.identifier,
                self.mean,
                self.stdev,
                self.description,
            ]
        else:
            items = [
                self.identifier,
                self.mean,
                self.stdev,
                self.variance,
                self.description,
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
        return 5

    def __str__(self) -> str:
        return (
            f"{str_or_empty_if_none(self.identifier)}\t"
            f"{str_or_empty_if_none(self.mean)}\t"
            f"{str_or_empty_if_none(self.stdev)}\t"
            f"{str_or_empty_if_none(self.variance)}\t"
            f"{str_or_empty_if_none(self.description)}"
        )


def _read_tsv(
    records: list[Record], # Reference/pointer to a mutable list.
    tsv_file_path: str | Path = Path(DEFAULT_PATH),
    skip_variance: bool = False,
) -> None:
    try:
        skip_variance = bool(skip_variance)
    except (TypeError, ValueError):
        skip_variance = False

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

        if skip_variance:
            record_length -= 1

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

            if skip_variance:
                record = Record(
                    identifier=int_or_none(tsv_row[0]),
                    mean=float_or_none(tsv_row[1]),
                    stdev=float_or_none(tsv_row[2]),
                    variance=None,
                    description=str_or_none_if_empty(tsv_row[3]),
                )
            else:
                record = Record(
                    identifier=int_or_none(tsv_row[0]),
                    mean=float_or_none(tsv_row[1]),
                    stdev=float_or_none(tsv_row[2]),
                    variance=float_or_none(tsv_row[3]),
                    description=str_or_none_if_empty(tsv_row[4]),
                )

            # Mutates the list.
            records.append(record)

    return None


def _write_tsv(
    records: list[Record],
    tsv_file_path: str | Path = Path(DEFAULT_PATH),
    skip_variance: bool = False,
) -> None:
    try:
        skip_variance = bool(skip_variance)
    except (TypeError, ValueError):
        skip_variance = False

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
        if skip_variance:
            header_row = [
                "id",
                "mean",
                "stdev",
                "description",
            ]
        else:
            header_row = [
                "id",
                "mean",
                "stdev",
                "variance",
                "description",
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
                skip_variance=skip_variance,
            ))

    return None


class RecordList(MutableSequence):
    # MutableSequence must implement methods __init__, __getitem__, __setitem__,
    # __delitem__, __len__, and insert.

    def load(
        self,
        tsv_file_path: str | Path = Path(DEFAULT_PATH),
        skip_variance: bool = False,
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
            skip_variance=skip_variance,
        )

        return None

    def save(
        self,
        tsv_file_path: str | Path = Path(DEFAULT_PATH),
        skip_variance: bool = False,
    ) -> None:
        _write_tsv(
            records=self.records,
            tsv_file_path=tsv_file_path,
            skip_variance=skip_variance,
        )

        return None

    def __init__(
        self,
        tsv_file_path: str | Path | None = Path(DEFAULT_PATH),
        skip_variance: bool = False,
    ) -> None:
        # Creates a reference/pointer to a new empty list (mutable).
        self.records: list[Record] = []

        if isinstance(tsv_file_path, str | Path):
            self.load(
                tsv_file_path=tsv_file_path,
                skip_variance=skip_variance,
                clear=False,
            )

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
            "id\t"
            "mean\t"
            "stdev\t"
            "variance\t"
            "description"
        )


# TODO(wathne): Test this function (it should work).
def get_records(
    records: list[Record] | None = None, # Reference/pointer to a mutable list.
    tsv_file_path: str | Path = Path(DEFAULT_PATH),
    skip_variance: bool = False,
) -> None | list[Record]:
    if isinstance(records, list):
        # Clears the mutable list.
        records.clear()

        # Mutates the list.
        _read_tsv(
            records=records,
            tsv_file_path=tsv_file_path,
            skip_variance=skip_variance,
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
            skip_variance=skip_variance,
        )

        # Returns a reference/pointer to the new list.
        return records

    raise TypeError


# TODO(wathne): Test this function (it should work).
def set_records(
    records: list[Record],
    tsv_file_path: str | Path = Path(DEFAULT_PATH),
    skip_variance: bool = False,
) -> None:
    _write_tsv(
        records=records,
        tsv_file_path=tsv_file_path,
        skip_variance=skip_variance,
    )

    return None
