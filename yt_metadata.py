# YouTube video metadata module.

# Module requirements (included in requirements.txt):
# google-api-python-client # TODO(wathne): Remove from requirements?
# google-auth-oauthlib # TODO(wathne): Remove from requirements?
# google-auth-httplib2 # TODO(wathne): Remove from requirements?
# python-youtube


from captions_with_evaluation_results_wrapper import Record
from captions_with_evaluation_results_wrapper import RecordList
from helpers import int_or_none
from helpers import randomized_wait
from helpers import str_or_empty_if_none
from helpers import str_or_none_if_empty
from json import JSONDecoder
from json import JSONEncoder
from os import getenv
from pyyoutube import Client
from pyyoutube import Comment
from pyyoutube import CommentSnippet
from pyyoutube import CommentThread
from pyyoutube import CommentThreadListResponse
from pyyoutube import CommentThreadSnippet
from pyyoutube import Video
from pyyoutube import VideoListResponse
from pyyoutube import VideoSnippet
from pyyoutube.error import PyYouTubeException


def _get_api_key() -> str | None:
    api_key: str | None
    try:
        from private_api_keys import GOOGLE_YOUTUBE_API_KEY
        api_key = str_or_none_if_empty(GOOGLE_YOUTUBE_API_KEY)
    except (ImportError, NameError) as instance:
        print(instance)
        api_key = getenv(
            key="GOOGLE_YOUTUBE_API_KEY",
            default=None,
        )

    return api_key


def fetch_metadata_json(
    youtube_video_id: str | None,
    api_key: str | None = None,
) -> str | None:
    api_key = str_or_none_if_empty(api_key)
    if api_key is None:
        api_key = _get_api_key()
    if api_key is None:
        print("Error: GOOGLE_YOUTUBE_API_KEY does not exist.")
        return None

    youtube_video_id = str_or_none_if_empty(youtube_video_id)
    if youtube_video_id is None:
        print("Fetching metadata ...", end="", flush=True)
        print("skipped (YouTube video ID must be a non-empty string).")

        return None

    # Initializes an instance of Python YouTube Data API v3 wrapper
    # (python-youtube).
    client: Client = Client(
        client_id=None,
        client_secret=None,
        access_token=None, # Access token is not required.
        refresh_token=None,
        api_key=api_key,
        client_secret_path=None,
        timeout=None,
        proxies=None,
        headers=None,
    )

    print(f"{youtube_video_id} Fetching metadata ...", end="", flush=True)

    # Title and description.
    title: str | None = None
    description: str | None = None

    video_list_response: dict | VideoListResponse | None
    try:
        video_list_response = client.videos.list(
            parts=[
                #"kind", # Not allowed by python-youtube.
                #"etag", # Not allowed by python-youtube.
                "id",
                "snippet",
                #"contentDetails",
                #"status",
                #"statistics",
                #"paidProductPlacementDetails", # Not supported.
                #"player",
                #"topicDetails",
                #"recordingDetails",
                #"fileDetails", # Not supported.
                #"processingDetails", # Not supported.
                #"suggestions", # Not supported.
                #"liveStreamingDetails",
                #"localizations", # Not supported.
            ],
            chart=None,
            video_id=youtube_video_id,
            my_rating=None,
            hl=None,
            max_height=None,
            max_results=None,
            max_width=None,
            on_behalf_of_content_owner=None,
            page_token=None,
            region_code=None,
            video_category_id=None,
            return_json=False,
        )
    except PyYouTubeException as instance:
        video_list_response = None
        print(instance)

    # if return_json is True.
    if isinstance(video_list_response, dict):
        # TODO(wathne): Handle JSON dict response (not necessary).
        print("TODO(wathne): Handle JSON dict response.")
        pass

    # if return_json is False.
    if isinstance(video_list_response, VideoListResponse):
        videos: list[Video] | None = video_list_response.items
        if videos is not None:
            video: Video | None
            try:
                video = videos[0]
            except IndexError:
                video = None
            if video is not None:
                video_id: str | None = str_or_none_if_empty(video.id)
                if video_id is None or video_id != youtube_video_id:
                    # TODO(wathne): Should an exception be raised or logged?
                    pass

                video_snippet: VideoSnippet | None = video.snippet
                if video_snippet is not None:
                    title = str_or_none_if_empty(video_snippet.title)
                    description = str_or_none_if_empty(
                        video_snippet.description
                    )

    # At this point the title and description are either strings or None.

    # Comments.
    comments: list[Comment] = [] # Top 20 comments by relevance.

    comment_thread_list_response: dict | CommentThreadListResponse | None
    try:
        comment_thread_list_response = client.commentThreads.list(
            parts=[
                #"kind", # Not allowed by python-youtube.
                #"etag", # Not allowed by python-youtube.
                #"id",
                "snippet",
                #"replies",
            ],
            all_threads_related_to_channel_id=None,
            channel_id=None,
            thread_id=None,
            video_id=youtube_video_id,
            max_results=20,
            moderation_status=None,
            order="relevance",
            page_token=None,
            search_terms=None,
            text_format="plainText",
            return_json=False,
        )
    except PyYouTubeException as instance:
        comment_thread_list_response = None
        print(instance)

    # if return_json is True.
    if isinstance(comment_thread_list_response, dict):
        # TODO(wathne): Handle JSON dict response (not necessary).
        print("TODO(wathne): Handle JSON dict response.")
        pass

    # if return_json is False.
    if isinstance(comment_thread_list_response, CommentThreadListResponse):
        comment_threads: list[CommentThread] | None
        comment_threads = comment_thread_list_response.items
        if comment_threads is not None:
            top_lvl_comment: Comment | None
            comment_thread_snippet: CommentThreadSnippet | None
            comment_thread: CommentThread | None
            for comment_thread in comment_threads:
                if comment_thread is not None:
                    comment_thread_snippet = comment_thread.snippet
                    if comment_thread_snippet is not None:
                        top_lvl_comment = comment_thread_snippet.topLevelComment
                        if top_lvl_comment is not None:
                            comments.append(top_lvl_comment)

    # At this point the list of comments is either populated with comments or
    # empty.

    # Metadata: video title, description, and top comments (relevance).
    metadata: dict[str, str | list[dict[str, str]]] = {}

    # Structural example:
    #{
    #    "title": "Alien eating chocolate for the first time",
    #    "description": "",
    #    "top_comments": [
    #        {
    #            "comment": "no way",
    #        },
    #        {
    #            "comment": "I hope the alien is ok.",
    #        },
    #        {
    #            "comment": "auto-translate doesn't work, what did it say?",
    #        },
    #    ]
    #}

    metadata["title"] = str_or_empty_if_none(title)
    metadata["description"] = str_or_empty_if_none(description)
    metadata["top_comments"] = []

    comment_text: str | None
    comment_snippet: CommentSnippet | None
    comment: Comment
    for comment in comments:
        comment_snippet = comment.snippet
        if comment_snippet is None:
            continue

        comment_text = str_or_none_if_empty(comment_snippet.textDisplay)
        if comment_text is None:
            continue

        metadata["top_comments"].append(
            {
                "comment": comment_text,
            }
        )

    # Initializes an instance of JSONEncoder.
    json_encoder: JSONEncoder = JSONEncoder(
        skipkeys=False,
        ensure_ascii=True,
        check_circular=True,
        allow_nan=True,
        sort_keys=False,
        indent=None,
        separators=(", ", ": "),
        default=None,
    )

    metadata_json: str = json_encoder.encode(metadata)

    print("done.")

    return metadata_json


def fetch_metadata_for_records(
    records: RecordList,
    wait_milliseconds_min: int | None = 12000, # 12 seconds
    wait_milliseconds_max: int | None = 24000, # 24 seconds
) -> None:
    api_key: str | None = _get_api_key()
    if api_key is None:
        print("Error: GOOGLE_YOUTUBE_API_KEY does not exist.")
        return None

    cc_status: int | None
    metadata_json: str | None

    length: int = len(records)
    i: int
    record: Record # Reference/pointer to a mutable Record.
    for i, record in enumerate(records):
        print(f"({i+1}/{length}) ", end="", flush=True)

        cc_status = int_or_none(record.cc_status)

        # The following CC status checks have been copied from yt_transcripts.py
        # and modified to act here as a more general filter for good versus bad
        # videos.

        # CC status None means that new closed captions availability has not
        # been determined for this YouTube video ID.
        # Type None is the default type when no integer value has been set.
        if cc_status is None:
            print("Fetching metadata ...", end="", flush=True)
            print("skipped (cc_status is None, a CC status value must be set).")

            continue

        # CC status 0 means that someone (wathne) has determined that new closed
        # captions are permanently unavailable for this YouTube video ID.
        # Value 0 should only have been set manually.
        # CC status 0 is appropriate for deleted videos or videos where closed
        # captions have been disabled.
        if cc_status == 0:
            print("Fetching metadata ...", end="", flush=True)
            print("skipped (cc_status = 0, closed captions are unavailable).")

            continue

        if (
            # CC status 1 means that someone (wathne) has tentatively determined
            # that new closed captions are available for this YouTube video ID.
            # Value 1 should only have been set manually.
            # CC status 1 is appropriate for videos that seem fine and have
            # closed captions.
            cc_status == 1 or
            # CC status 200 means that new closed captions have already been
            # saved for this YouTube video ID.
            # Value 200 should have been set automatically.
            cc_status == 200 or
            # CC status 204 means that empty new closed captions have already
            # been fetched but not saved for this YouTube video ID.
            # Value 204 should have been set automatically.
            cc_status == 204 or
            # CC status 404 basically means what you think it means.
            # Value 404 should have been set automatically.
            cc_status == 404
        ):
            try:
                metadata_json = fetch_metadata_json(
                    youtube_video_id=record.video_id,
                    api_key=api_key,
                )

                # Mutates the Record.
                record.metadata = metadata_json

            except PyYouTubeException as instance:
                print(instance)

            # Waits and hopefully avoids getting banned by YouTube.
            print(f"    ", end="", flush=True)
            randomized_wait(
                wait_milliseconds_min=wait_milliseconds_min,
                wait_milliseconds_max=wait_milliseconds_max,
                verbose=True,
            )

            continue

        print("Fetching metadata ...", end="", flush=True)
        print("skipped (cc_status is set, but unknown).")

    return None


# TODO(wathne): Test. Remove this.
def test() -> None:
    #records: RecordList = RecordList(tsv_file_path=None)

    #record_1: Record = Record(
    #    video_id="e8KRPFOD1RE",
    #    cc_status=1,
    #)
    #records.append(value=record_1)

    #record_2: Record = Record(
    #    video_id="ElTTOsj3y-Q",
    #    cc_status=1,
    #)
    #records.append(value=record_2)

    records: RecordList = RecordList(tsv_file_path="./test.tsv")

    fetch_metadata_for_records(
        records=records,
        wait_milliseconds_min=12000,
        wait_milliseconds_max=24000,
    )

    records.save(tsv_file_path="./test.tsv")

    #records.load(tsv_file_path="./test.tsv", clear=True)

    # Initializes an instance of JSONDecoder.
    json_decoder: JSONDecoder = JSONDecoder(
        object_hook=None,
        parse_float=None,
        parse_int=None,
        parse_constant=None,
        strict=True,
        object_pairs_hook=None,
    )

    comment_dict: dict[str, str]
    comment_dict_list: str | list[dict[str, str]] # list[dict[str, str]]
    metadata_dict: dict[str, str | list[dict[str, str]]]
    metadata_json: str | None
    record: Record
    for record in records:
        metadata_json = record.metadata
        if metadata_json is None:
            continue

        metadata_dict = json_decoder.decode(metadata_json)

        print(f"title: {metadata_dict["title"]}")
        print(f"description: {metadata_dict["description"]}")

        comment_dict_list = metadata_dict["top_comments"]
        if isinstance(comment_dict_list, list):
            for comment_dict in comment_dict_list:
                print(f"comment: {comment_dict["comment"]}")


# TODO(wathne): Test. Remove this.
if __name__ == "__main__":
    test()
