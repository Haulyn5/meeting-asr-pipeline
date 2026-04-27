# Open Source Release Checklist

Use this checklist before pushing the repository to GitHub.

## Must Not Be Committed

- Raw meeting recordings under `data/raw_input/`.
- Generated transcripts, chunks, embeddings, manifests, and normalized audio under `outputs/`.
- Local model weights under `models/`, `PretrainedModels/`, or any external model cache.
- Local Python environments and package caches such as `.venv/` and `.uv-cache/`.
- Private `.env` files, tokens, API keys, internal hostnames, or personal paths.

The repository `.gitignore` excludes these paths by default. Still run the audit commands below before the first push.

## Audit Commands

```bash
git status --ignored
find . -type f -size +10M -not -path './.git/*'
scripts/audit_open_source.sh
```

If the repository already has commits containing sensitive files, `.gitignore` is not enough. Rewrite history or create a clean repository from the sanitized working tree.

## Recommended GitHub Repository Name

Use a URL-safe slug:

```text
meeting-asr-pipeline
```

The display title can remain:

```text
Meeting ASR Pipeline
```

GitHub repository names can contain spaces in the UI in some contexts, but a lowercase hyphenated slug is easier to clone, document, package, and reference.

## First Publish Flow

```bash
git init
git add README.md .gitignore .env.example envs docs scripts data/raw_input/.gitkeep outputs/.gitkeep
git status
git commit -m "Initial open-source release"
git branch -M main
git remote add origin git@github.com:<owner>/meeting-asr-pipeline.git
git push -u origin main
```

Do not run `git add .` until `git status --ignored` confirms private artifacts are excluded.
