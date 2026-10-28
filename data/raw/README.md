# Add legal sources to the corpus

Place source documents in this folder. Use official legislation, gazette notifications, government procedure, or authoritative court material. Do not add blogs or unsourced summaries.

Supported files: PDF, TXT, Markdown, and JSON. For PDF, TXT, or Markdown, you may add a metadata sidecar next to the source. For a source named `example-act.pdf`, name the sidecar `example-act.metadata.json` or `example-act.meta.json` (the source extension is replaced).

Example sidecar:

```json
{
  "document_id": "example-act-2024",
  "document_name": "Example Act, 2024",
  "act_name": "Example Act",
  "document_type": "legislation",
  "source_url": "https://official.example.gov/example-act.pdf",
  "issuing_authority": "Issuing authority",
  "document_version": "2024 text",
  "effective_date": null,
  "publication_date": null
}
```

The importer requires `document_id`, `document_name`, `act_name`, `document_type`, `source_url`, and `issuing_authority`. Every `document_id` must be unique. Optional version/date fields help explain what text was ingested. Do not label a source current or authoritative unless the source itself supports that description.

A self-contained JSON source can place its text in a `text` property and include the same metadata fields in that JSON file.

If metadata is missing, the importer generates fallback values and labels the provenance unverified. A filename is not proof of source authenticity or legal currency.

## Sync

The API synchronizes this folder at startup and before normal chat processing. To build the index manually, run from `backend`:

```powershell
python -m app.retrieval.sync
```

The sync process stores generated chunks and a manifest in `data/processed`, then rebuilds `data/vector_store/legal_index.json` when source content changes. Keep original source files and their metadata together so the index can be reproduced.
