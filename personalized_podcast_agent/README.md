# Personalized Podcast Agent — Offline Candidate Pool

An offline-first personalized podcast prototype that generates reusable podcast candidates from a listener profile and personal content history.

The project is designed to prepare content before playback:

1. Build a user preference profile from optional Instagram and YouTube history.
2. Retrieve and embed personal feed items.
3. Generate topic-related content trunks.
4. Predict personalized questions for each trunk.
5. Generate short spoken podcast segments for the selected candidate.

The generated candidate pool can later be used by an interactive player without requiring every response to be generated from scratch.

## Project structure

```text
personalized_podcast_agent/
├── src/
│   ├── offline/
│   │   ├── generate_main_narrative.py  # Split each selected transcript into 5 ordered parts
│   │   ├── generate_trunks.py          # Rewrite parts into podcast trunks and generate Q&A
│   │   ├── predict_user_actions.py     # Predict likely listener questions
│   │   ├── link_questions.py           # Find next trunks
│   │   └── question_index.py           # Index candidate questions
│   ├── user_profile/
│   │   ├── import_youtube_homepage_manual.py  # Import homepage videos and transcripts
│   │   ├── build_personal_feed.py             # Build the unified personal feed
│   │   ├── build_feed_embeddings.py           # Create feed embeddings
│   │   └── select_feed_seeds.py               # MMR-select daily video candidates
│   ├── api_app.py                      # API entry point
│   └── utils.py                        # Shared utilities and local LLM calls
├── data/
│   ├── input/                          # Local input data; do not commit
│   └── output/                         # Generated files; do not commit
├── web/                                # Front-end files
├── requirements.txt
└── README.md
```

## Installation

From the project folder:

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment variables

Create a local `.env` file:

```env
YOUTUBE_API_KEY=your_key_here
LOCAL_LLM_HOST=your_local_host
LOCAL_LLM_MODEL=qwen2.5:1.5b
PROMPT_CASCADE_MOCK=0
```

The local LLM must be running before generating trunks. Translation settings
may also be required when the collected YouTube metadata or transcripts are
not already in English.

## Optional profile exports

If Instagram or YouTube Takeout exports are available, place them under the
local `data/` directory and run:

```bash
python src/00_parse_instagram_export.py
python src/00_parse_youtube_history.py
python src/00_build_user_preference.py
```

This creates the user profile used when predicting possible listener questions.

## Daily Workflow

### 1. Collect Daily Source Data

The browser collection scripts are stored in:

```text
src/fetch_code/
├── fetch_youtube.js
├── fetch_instagram.js
└── fetch_google_search.js
```

Open the corresponding file and run it in the browser Console.

Save the collected files to:

```text
data/daily/
├── youtube_homepage_manual.json
├── instagram_feed_manual_json
└── google_search_manual.json
```

The scripts collect:

- 100 YouTube homepage videos
- 100 Instagram posts or Reels
- The user's daily Google Search records

### 2. Organize Source Content

Run:

```bash
python -m src.user_profile.import_youtube_homepage_manual
python -m src.user_profile.organize_instagram_feed
python -m src.user_profile.organize_google_search
```

Generated files:

```text
data/daily/
├── youtube_daily_items_raw.json
├── youtube_daily_items.json
├── instagram_profile_raw.json
├── instagram_profile.json
├── google_search_profile_raw.json
└── google_search_profile.json
```

### 3. Build the Personal Feed

Run:

```bash
python -m src.user_profile.build_personal_feed
```

Output:

```text
data/output/personal_feed_items.json
```

The unified feed contains:

```text
feed_id
source_type
text
url
creator
```

### 4. Build Multilingual Embeddings

Run:

```bash
python -m src.user_profile.build_feed_embeddings
```

Outputs:

```text
data/output/
├── personal_feed_embeddings.npy
└── personal_feed_embedding_ids.json
```

The embeddings are used to compare YouTube, Instagram, and Google Search feeds in the same semantic space.

### 5. Select Topics and Generate Narratives

Run:

```bash
python -m src.offline.select_feed_seeds
python -m src.offline.generate_main_narrative
```

This step groups semantically related feeds from all sources and generates English narratives for each topic.

Outputs:

```text
data/output/
├── topic_feed_seeds.json
└── main_narratives.json
```

### 6. Generate Trunks and Questions

Run:

```bash
python -m src.offline.generate_trunks
```

This step:

- Uses flexible sections based on the content
- Splits long-form content into multiple trunks
- Usually keeps short posts, Reels, and Shorts as one trunk
- Generates three predicted questions for each trunk

Output:

```text
data/output/candidate_trunks.json
```

### 7. Link Questions to Other Trunks

Run:

```bash
python -m src.offline.link_questions
```

For each question, the system:

1. Creates an embedding for the question.
2. Finds similar trunks, excluding the current trunk.
3. Checks whether another trunk can answer the question.
4. Links the question to the answer trunk through `next_trunks`.

Output:

```text
data/output/candidate_trunks_with_next.json
```


## Output Files

```text
data/output/
├── personal_feed_items.json
├── personal_feed_embeddings.npy
├── personal_feed_embedding_ids.json
├── topic_feed_seeds.json
├── main_narratives.json
├── candidate_trunks.json
└── candidate_trunks_with_next.json
```
