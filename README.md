# qwen-commit

`qwen-commit` is a fine-tuned SLM ([Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)) to generate concise Git commit subjects from staged diffs.

The pipeline collects a user's local Git history, trains SFT and DPO adapters and serves it locally for a fill-then-review Git hook.

## History scan

The scan reports repositories, shallow clones, total commits, and distinct author-email counts; it
does not store commit messages, diffs, or author identities.

```console
uv run qwen-commit history scan
```

It writes `history-scan.json` in the current directory. Use `--json NAME` to choose a
different report path.

## Candidate dataset

Build the candidate dataset from the configured included repositories:

```console
uv run qwen-commit data build-candidates
```

This writes `candidates.parquet` for training and `provenance.parquet` for private source traceability. The builder excludes merge, fixup, bot, binary,
generated-only, and empty commits. It assumes the configured local history is suitable for collection. It never puts repository paths, remotes, commit SHAs, or author identities in `candidates.parquet` - which is considered the training dataset.

## Output contract

The model must return exactly one Git commit subject:

- at most 72 characters;
- no preamble or explanation;
- no Markdown or surrounding quotation marks;
- no required Conventional Commit prefix.

## Development

The project targets Python 3.12 and uses [`uv`](https://docs.astral.sh/uv/) for environments, locking, and builds.

```console
uv sync --all-groups
uv run qwen-commit --help
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv build
```

The package can also be run as a module:

```console
uv run python -m qwen_commit --version
```

## License

Project source code is available under the [Apache License 2.0](LICENSE). Model weights retain
their own upstream license and provenance.
