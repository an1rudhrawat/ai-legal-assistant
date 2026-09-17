# Authoritative legal source input

Place only authoritative material here: official legislation, gazette notifications, court material, or official procedural guidance. Do not add blogs or unsourced summaries.

You may upload a PDF, TXT, Markdown, or self-contained JSON document by itself. The importer creates a stable document ID and display name from the filename, and marks its provenance as unverified. Name files clearly, for example `bharatiya-nyaya-sanhita-2023.pdf`.

For reliable user-facing citations, you can optionally add a companion metadata file named `<source>.metadata.json` (or `<source>.meta.json`) with `document_id`, `document_name`, `act_name`, `document_type`, `source_url`, and `issuing_authority`. A JSON source can include those fields and `text` in one file. Every explicit `document_id` must be unique.

The API synchronizes this folder automatically at startup and before normal chat requests. It hashes every source plus its metadata, preserves chunks for unchanged documents in `data/processed/documents`, and chunks only new or modified documents. You can also sync manually with `python -m app.retrieval.sync` from `backend`.
