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
│   │   ├── generate_trunks.py          # Generate topic/content trunks
│   │   ├── generate_tree.py            # Generate offline question tree
│   │   ├── generate_main_narrative.py  # Generate short narrative segments
│   │   ├── predict_user_actions.py     # Predict likely listener actions/questions
│   │   └── question_index.py           # Search and index candidate questions
│   ├── online/                         # Optional online interaction components
│   ├── 00_parse_instagram_export.py    # Parse Instagram export
│   ├── 00_parse_youtube_history.py     # Parse YouTube history
│   ├── 00_build_user_preference.py     # Build preference profile
│   ├── api_app.py                      # API entry point
│   └── utils.py                        # Shared utilities
├── data/
│   ├── input/                          # Local input data; do not commit private data
│   └── output/                         # Generated profiles, embeddings, and candidates
├── web/                                # Front-end files
├── requirements.txt
└── README.md
```

## Installation

From the `personalized_podcast_agent` folder:

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

Create a `.env` file locally. Do not commit it to GitHub.

```env
OPENAI_API_KEY=your_key_here
```

Depending on the retrieval configuration, additional API keys may be required.

## Offline workflow

### 1. Prepare a user profile

Place optional source exports in the local `data/` directory, then run the relevant parsing and profile-building scripts:

```bash
python src/00_parse_instagram_export.py
python src/00_parse_youtube_history.py
python src/00_build_user_preference.py
```

This creates a preference profile used to personalize content selection.

### 2. Build personal-feed embeddings

Create multilingual embeddings for the personal feed items:

```bash
python src/offline/build_feed_embeddings.py
```

The embeddings support cosine-similarity retrieval of content related to a selected topic.

### 3. Generate offline trunks and questions

Generate topic-specific content trunks and personalized question candidates:

```bash
python src/offline/generate_trunks.py
```

Generated files are saved under `data/output/`, for example:

```text
candidate_trunks.json
candidate_trunk_embeddings.npy
personal_feed_embeddings.npy
personal_feed_embedding_ids.json
```

### 4. Generate podcast segments

The system generates short spoken-style segments based on the selected trunk and predicted listener question. Each segment focuses on one main idea and can be used as the next podcast segment.

## Privacy and repository policy

Do not commit personal raw data or credentials, including:

```text
.env
data/instagram_export/
data/youtube_history/
data/output/user_profile.json
data/output/personal_feed_items.json
*.npy
__pycache__/
.venv/
```

Generated embeddings and user-specific profiles should be recreated locally from each user's own data.
