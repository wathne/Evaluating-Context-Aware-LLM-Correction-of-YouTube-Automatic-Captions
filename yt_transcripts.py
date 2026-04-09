# YouTube transcripts module.

# Module requirements (included in requirements.txt):
# youtube-transcript-api


from captions_with_evaluation_results_wrapper import Record
from captions_with_evaluation_results_wrapper import RecordList
from helpers import input_to_continue
from helpers import int_or_none
from helpers import randomized_wait
from helpers import str_or_none_if_empty
from youtube_transcript_api import FetchedTranscript
from youtube_transcript_api import FetchedTranscriptSnippet
from youtube_transcript_api import Transcript
from youtube_transcript_api import TranscriptList
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import Formatter

# Known youtube_transcript_api exceptions:
from youtube_transcript_api import IpBlocked
from youtube_transcript_api import YouTubeRequestFailed
from youtube_transcript_api import FailedToCreateConsentCookie
from youtube_transcript_api import YouTubeDataUnparsable
from youtube_transcript_api import RequestBlocked
from youtube_transcript_api import AgeRestricted
from youtube_transcript_api import InvalidVideoId
from youtube_transcript_api import VideoUnavailable
from youtube_transcript_api import VideoUnplayable
from youtube_transcript_api import TranscriptsDisabled
from youtube_transcript_api import NoTranscriptFound
from youtube_transcript_api import PoTokenRequired


class StringFormatter(Formatter):
    # Formatter must implement methods format_transcript and format_transcripts.

    def format_transcript(
        self,
        transcript: FetchedTranscript,
        **kwargs,
    ) -> str:
        # A FetchedTranscript object is iterable, which allows iteration over
        # transcript snippets.
        snippet: FetchedTranscriptSnippet
        return " ".join(snippet.text for snippet in transcript)

    # TODO(wathne): Implement this method (not necessary).
    def format_transcripts(
        self,
        transcripts: list[FetchedTranscript],
        **kwargs,
    ) -> None:
        raise NotImplementedError(
            "A subclass of Formatter must implement "
            "their own .format_transcripts() method."
        )


# Known youtube_transcript_api exceptions:
# IpBlocked, YouTubeRequestFailed, FailedToCreateConsentCookie,
# YouTubeDataUnparsable, RequestBlocked, AgeRestricted, InvalidVideoId,
# VideoUnavailable, VideoUnplayable, TranscriptsDisabled, NoTranscriptFound, and
# PoTokenRequired.
def fetch_transcript_string(
    youtube_video_id: str | None,
    wait_milliseconds_min: int | None = 120000, # 120 seconds
    wait_milliseconds_max: int | None = 240000, # 240 seconds
) -> str | None:
    youtube_video_id = str_or_none_if_empty(youtube_video_id)
    if youtube_video_id is None:
        print("Fetching transcript ...", end="", flush=True)
        print("skipped (YouTube video ID must be a non-empty string).")

        return None

    # Initializes an instance of StringFormatter.
    string_formatter: StringFormatter = StringFormatter()

    # Initializes an instance of YouTubeTranscriptApi.
    ytt_api: YouTubeTranscriptApi = YouTubeTranscriptApi(
        proxy_config=None,
        http_client=None,
    )

    print(f"{youtube_video_id} Fetching transcript ...", end="", flush=True)

    # Retrieves the list of transcripts which are available for the given
    # YouTube video ID.
    # Known youtube_transcript_api exceptions:
    # IpBlocked, YouTubeRequestFailed, FailedToCreateConsentCookie,
    # YouTubeDataUnparsable, RequestBlocked, AgeRestricted, InvalidVideoId,
    # VideoUnavailable, VideoUnplayable, and TranscriptsDisabled.
    transcript_list: TranscriptList
    try:
        transcript_list = ytt_api.list(
            video_id=youtube_video_id,
        )
    except TranscriptsDisabled:
        print("skipped (transcripts are disabled for this video).")

        # Waits and hopefully avoids getting banned by YouTube.
        print(f"    ", end="", flush=True)
        randomized_wait(
            wait_milliseconds_min=wait_milliseconds_min,
            wait_milliseconds_max=wait_milliseconds_max,
            verbose=True,
        )

        return None

    # Finds an automatically generated transcript for the english language code.
    # We intentionally use this method to avoid manually created transcripts.
    # The returned Transcript object contains transcript metadata properties.
    transcript: Transcript
    try:
        transcript = transcript_list.find_generated_transcript(
            language_codes=["en"]
        )
    # Known youtube_transcript_api exceptions:
    # NoTranscriptFound.
    except NoTranscriptFound:
        print("skipped (english autogen transcript not found).")

        # Waits and hopefully avoids getting banned by YouTube.
        print(f"    ", end="", flush=True)
        randomized_wait(
            wait_milliseconds_min=wait_milliseconds_min,
            wait_milliseconds_max=wait_milliseconds_max,
            verbose=True,
        )

        return None

    # Fetches the actual transcript data as a FetchedTranscript object.
    # Known youtube_transcript_api exceptions:
    # PoTokenRequired, IpBlocked, and YouTubeRequestFailed.
    fetched_transcript: FetchedTranscript = transcript.fetch(
        preserve_formatting=True
    )

    # Formats the transcript as a String.
    string_formatted: str = string_formatter.format_transcript(
        transcript=fetched_transcript
    )

    print("done.")

    # Waits and hopefully avoids getting banned by YouTube.
    print(f"    ", end="", flush=True)
    randomized_wait(
        wait_milliseconds_min=wait_milliseconds_min,
        wait_milliseconds_max=wait_milliseconds_max,
        verbose=True,
    )

    return string_formatted


def fetch_transcripts_for_records(
    records: RecordList,
    clear_unavailable_transcripts: bool = False,
    wait_for_input_to_continue: bool = False,
    wait_milliseconds_min: int | None = 120000, # 120 seconds
    wait_milliseconds_max: int | None = 240000, # 240 seconds
) -> None:
    try:
        clear_unavailable_transcripts = bool(clear_unavailable_transcripts)
    except (TypeError, ValueError):
        clear_unavailable_transcripts = False

    try:
        wait_for_input_to_continue = bool(wait_for_input_to_continue)
    except (TypeError, ValueError):
        wait_for_input_to_continue = False

    cc_status: int | None
    youtube_caption: str | None

    length: int = len(records)
    i: int
    record: Record # Reference/pointer to a mutable Record.
    for i, record in enumerate(records):
        if wait_for_input_to_continue:
            if not input_to_continue():
                print("exit")

                return None

            print("continue")

        print(f"({i+1}/{length}) ", end="", flush=True)

        cc_status = int_or_none(record.cc_status)

        # CC status None means that new closed captions availability has not
        # been determined for this YouTube video ID.
        # Type None is the default type when no integer value has been set.
        if cc_status is None:
            print("Fetching transcript ...", end="", flush=True)
            print("skipped (cc_status is None, a CC status value must be set).")

            continue

        # CC status 0 means that someone (wathne) has determined that new closed
        # captions are permanently unavailable for this YouTube video ID.
        # Value 0 should only have been set manually.
        # CC status 0 is appropriate for deleted videos or videos where closed
        # captions have been disabled.
        if cc_status == 0:
            print("Fetching transcript ...", end="", flush=True)
            print("skipped (cc_status = 0, closed captions are unavailable).")

            if clear_unavailable_transcripts:
                # Mutates the Record.
                record.youtube_caption = None

            continue

        # TODO(wathne): Include block 204?
        if (
            # CC status 1 means that someone (wathne) has tentatively determined
            # that new closed captions are available for this YouTube video ID.
            # Value 1 should only have been set manually.
            # CC status 1 is appropriate for videos that seem fine and have
            # closed captions.
            cc_status == 1 or
            # CC status 404 basically means what you think it means.
            # Value 404 should have been set automatically.
            cc_status == 404
        ):
            try:
                youtube_caption = fetch_transcript_string(
                    youtube_video_id=record.video_id,
                    wait_milliseconds_min=wait_milliseconds_min,
                    wait_milliseconds_max=wait_milliseconds_max,
                )
            # Known youtube_transcript_api exceptions:
            except (
                IpBlocked,
                YouTubeRequestFailed,
                FailedToCreateConsentCookie,
                YouTubeDataUnparsable,
                RequestBlocked,
                AgeRestricted,
                InvalidVideoId,
                VideoUnavailable,
                VideoUnplayable,
                #TranscriptsDisabled,
                #NoTranscriptFound,
                PoTokenRequired,
            ) as instance:
                print(instance)

                # Returns to allow saving fetched transcripts instead of
                # crashing and losing all unsaved progress.
                # Getting IP banned (IpBlocked) does happen, at least when the
                # randomized_wait() function is not set to wait long enough.
                return None

            if youtube_caption is None:
                # Mutates the Record.
                record.cc_status = 404

                if clear_unavailable_transcripts:
                    # Mutates the Record.
                    record.youtube_caption = None

                continue

            if youtube_caption == "":
                # Mutates the Record.
                record.cc_status = 204

                if clear_unavailable_transcripts:
                    # Mutates the Record.
                    record.youtube_caption = None

                continue

            # Mutates the Record.
            record.cc_status = 200
            record.youtube_caption = youtube_caption

            continue

        # CC status 200 means that new closed captions have already been saved
        # for this YouTube video ID.
        # Value 200 should have been set automatically.
        if cc_status == 200:
            print("Fetching transcript ...", end="", flush=True)
            print(
                "skipped (cc_status = 200, new closed captions have already"
                " been saved)."
            )

            continue

        # TODO(wathne): Combine block 204 with blocks 1 and 404?
        # CC status 204 means that empty new closed captions have already been
        # fetched but not saved for this YouTube video ID.
        # Value 204 should have been set automatically.
        if cc_status == 204:
            print("Fetching transcript ...", end="", flush=True)
            print(
                "skipped (cc_status = 204, empty new closed captions have"
                " already been fetched but not saved)."
            )

            # TODO(wathne): Keep this here?
            if clear_unavailable_transcripts:
                # Mutates the Record.
                record.youtube_caption = None

            continue

        print("Fetching transcript ...", end="", flush=True)
        print("skipped (cc_status is set, but unknown).")

    return None


# TODO(wathne): Test. Remove this.
def test() -> None:
    records: RecordList = RecordList(tsv_file_path=None)

    record_1: Record = Record(
        video_id="e8KRPFOD1RE",
        cc_status=1,
    )
    records.append(value=record_1)

    record_2: Record = Record(
        video_id="ElTTOsj3y-Q",
        cc_status=1,
    )
    records.append(value=record_2)

    #records: RecordList = RecordList(tsv_file_path="./test.tsv")

    fetch_transcripts_for_records(
        records=records,
        clear_unavailable_transcripts=True,
        wait_for_input_to_continue=False,
        wait_milliseconds_min=120000,
        wait_milliseconds_max=240000,
    )

    records.save(tsv_file_path="./test.tsv")


# TODO(wathne): Test. Remove this.
if __name__ == "__main__":
    test()
