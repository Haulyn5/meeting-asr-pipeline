# Security Policy

Meeting ASR Pipeline is designed for local processing of private recordings. Treat raw audio, generated transcripts, model caches, and `.env` files as sensitive data.

## Before Opening Issues

Do not attach private recordings, full transcripts, API keys, internal hostnames, or screenshots containing confidential content. If a bug requires an example, use a synthetic or publicly shareable audio sample.

## Reporting Security Issues

If you publish this project on GitHub, configure a private security reporting channel in the repository settings and update this file with the preferred contact method.

## Data Handling

The repository `.gitignore` excludes common sensitive paths by default:

- `data/raw_input/`
- `outputs/`
- `models/`
- `PretrainedModels/`
- `.env`
- local virtual environments and caches

Run `scripts/audit_open_source.sh` before pushing changes.
