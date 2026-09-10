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

## Daily YouTube podcast workflow

### 1. Copy 50 non-Shorts videos from the YouTube homepage

Sign in to YouTube, open the homepage, scroll until at least 50 long-form
videos are loaded, then open the browser Console with `F12` and run:

```javascript
const rows = [...document.querySelectorAll('a[href^="/watch"]')]
  .map((a) => {
    const card = a.closest(
      'ytd-rich-grid-media, ytd-rich-item-renderer, ytd-video-renderer, ytd-lockup-view-model'
    );

    return {
      title: (
        a.querySelector('#video-title')?.textContent ||
        a.textContent ||
        a.getAttribute('aria-label') ||
        ''
      ).trim(),
      channel: (
        card?.querySelector('#channel-name a, #channel-name')?.textContent ||
        ''
      ).trim(),
      url: a.href
    };
  })
  .filter((video) => video.title && video.url);

const videos = [...new Map(rows.map((video) => [video.url, video])).values()]
  .slice(0, 50);

copy(JSON.stringify(videos, null, 2));
console.log(`Copied ${videos.length} non-Shorts homepage videos.`);
```

Paste the copied JSON into:

```text
data/output/youtube_homepage_manual.json
```

The selector only collects `/watch` URLs, so it excludes YouTube Shorts.

### 2. Import metadata and transcripts

```bash
python -m src.user_profile.import_youtube_homepage_manual
```

This collects public metadata and available transcripts, skips videos without
usable transcripts, and filters out clearly unsuitable sources such as songs,
music performances, fancams, and sports highlights. It writes:

```text
data/output/youtube_daily_items_raw.json
data/output/youtube_daily_items.json
```

### 3. Build the personal feed and embeddings

```bash
python -m src.user_profile.build_personal_feed
python -m src.user_profile.build_feed_embeddings
```

`personal_feed_items.json` combines the user's available local sources with
the filtered daily YouTube items. The embedding step creates multilingual,
normalized embeddings for MMR selection.

### 4. Select stories and split transcripts

```bash
python -m src.offline.generate_main_narrative
```

This step runs MMR over the daily YouTube videos, selects up to 10 different
videos with saved transcripts, and splits every selected transcript into five
ordered sections. The result is saved to:

```text
data/output/topic_feed_seeds.json
data/output/main_narratives.json
```

### 5. Generate podcast trunks and questions

```bash
python -m src.offline.generate_trunks
```

For every transcript section, Qwen produces one third-person English podcast
script. The system then predicts three likely listener questions for each
trunk, creates transcript-supported answers, generates audio, and stores
embeddings for later retrieval.

Generated files include:

```text
data/output/candidate_trunks.json
data/output/candidate_trunk_embeddings.npy
data/output/candidate_question_embeddings.npy
data/output/candidate_embedding_ids.json
data/output/audio/
```

## Privacy and repository policy

Do not commit personal raw data, transcripts, generated profiles, embeddings,
audio files, or credentials. Keep at least the following out of Git:

```text
.env
data/instagram_export/
data/youtube_history/
data/output/
.venv/
__pycache__/
*.npy
*.wav
```
__pycache__/
.venv/
Generated embeddings and user-specific profiles should be recreated locally from each user's own data.
