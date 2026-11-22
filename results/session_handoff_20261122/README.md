# Session history and handoff

Start with `../../HANDOFF_20261122.md`, then `REVISION_LOG.md` and `../../results_report/REVIEW_NOTES.md`.

The **local-only** visible conversation and execution archive preserve earlier records across the session's 12 recorded compactions. `manifest.json` identifies the source snapshot, counts, exclusions and hashes. `CONVERSATION.md` is readable chronologically. `visible_conversation.jsonl.gz` is the structured equivalent. `execution_records.jsonl.gz` preserves command/tool records. `REVISION_INDEX.csv` indexes candidate patch/write/Git/scheduler actions by timestamp and source record ordinal; these are actions to inspect, not a claim that every command succeeded.

Private reasoning, system/developer instructions and image binary content are excluded. Obvious credentials are redacted. These conversation and execution files are deliberately ignored by Git; do not publish raw chat histories. The concise handoff, revision log, export code and manifest can be shared with the next agent.

Refresh with `python3 code/export_session_handoff.py` from the repository root. The export takes a bounded snapshot of the append-only source while the chat can keep running. It cannot recover records absent from that source. For historical exact changes, inspect indexed tool records and the Git commits named in `REVISION_LOG.md`; current workspace files and the publishing branch provide reproducible snapshots.

The resumed Claude sessions are also preserved in local `CLAUDE_CONVERSATION.md`, `claude_visible_conversation.jsonl.gz` and `claude_execution_records.jsonl.gz`; hashes/counts are in `claude_manifest.json`. Refresh them with `python3 code/export_claude_handoff.py`. The active completed handoff is `../../HANDOFF_CURRENT.md`.
