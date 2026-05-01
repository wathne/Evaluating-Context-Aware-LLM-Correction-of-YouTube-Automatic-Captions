# Evaluating Context-Aware LLM Correction of YouTube Automatic Captions
Running the full main() function in "./evaluation.py" will cost you about $1 in
total credit with the OpenAI API for GPT 3.5 and GPT 5.4.
The log file at "./logs/log_evaluation_2026_april_9.txt" shows what the terminal
output looks like.
The current main() function takes about an hour to finish since it does not have
to refetch all YouTube captions. It takes hours more if it is told to regenerate everything. Most of the time is then spent on waiting. Getting IP banned
(IpBlocked) does happen, at least when the randomized_wait() function is not set
to wait long enough. The current main() function avoids these problems by
instead loading a local copy of new YouTube captions that were fetched earlier.

See the main() function at the end of the evaluation module (evaluation.py). The main() function is meant be modified.

All captions and evaluation results are stored in the
"./captions_with_evaluation_results/" directory.

## How to run the evaluation:
### Step 1
```
git clone https://github.com/wathne/Evaluating-Context-Aware-LLM-Correction-of-YouTube-Automatic-Captions.git
```
### Step 2
```
cd ./Evaluating-Context-Aware-LLM-Correction-of-YouTube-Automatic-Captions
```
### Step 3
```
python -m venv venv
```
### Step 4
```
source venv/bin/activate
```
### Step 5
```
python -m pip install -r requirements.txt
```
### Step 6 (optional)
This step is optional and you can skip it. The fetch_metadata_for_records()
function is currently being skipped. If you still want to refetch all new video
metadata, then you need to uncomment the fetch_metadata_for_records() function
within the main() function of the evaluation module (evaluation.py), currently
lines 848 to 852.

The YouTube video metadata module requires a Google API key for YouTube video
metadata.
```
https://developers.google.com/youtube/v3/getting-started
https://console.cloud.google.com/apis/credentials
```
Set your GOOGLE_YOUTUBE_API_KEY environment variable or populate
"./private_api_keys.py" with your own private API key if you want to use the
YouTube video metadata module.
### Step 7
You can not skip this step unless you disable most of the jobs within the main()
function of the evaluation module (evaluation.py). Without an API key you will
see lots of "Error: OPENAI_GPT_API_KEY does not exist.".

The OpenAI GPT LLM module requires an OpenAI API key for GPT LLM.
```
https://developers.openai.com/api/docs/quickstart
https://platform.openai.com/api-keys
```
Set your OPENAI_GPT_API_KEY environment variable or populate
"./private_api_keys.py" with your own private API key if you want to use the
OpenAI GPT LLM module.
### Step 8 (final)
```
python evaluation.py
```
This final step will run all jobs as specified in the main() function of the
evaluation module (evaluation.py).

TODO(wathne): More details.
