# Project Maintenance Workflow

This guide describes a practical workflow for maintaining, organizing, pushing, and validating GeoMetricLab.

## 1. Daily maintenance

- Pull the latest branch state before making changes
- Check `git status --short` before and after edits
- Keep local-only artifacts under ignored directories such as `cache/`, `weights/`, `results/`, `vis/`, and `wandb/`
- Avoid mixing code cleanup, experiments, and documentation changes in one oversized commit when possible

## 2. Repository organization

- Keep public-facing entrypoints small and explicit
- Preserve provenance notes for third-party methods
- Store heavyweight data, checkpoints, and generated figures outside tracked source paths
- Update local module `README.md` files when a module’s public purpose changes

## 3. Editing and review cycle

1. Modify the target files
2. Run focused checks for the files or scripts you changed
3. Inspect diffs with `git --no-pager diff` or `git --no-pager diff --cached`
4. Confirm ignored outputs are not being tracked with `git check-ignore -v <path>`

## 4. Staging changes

- Stage all intended changes with `git add -A`
- If a file should be ignored but was tracked before, remove it from the index with `git rm --cached <path>` or `git rm --cached -r <dir>`
- Re-run `git status --short` to confirm the exact staged set

## 5. Committing

- Use a concise, descriptive message focused on the user-visible change
- Prefer one commit per logical change set

Example:

```bash
git add -A
git commit -m "Polish README and project module docs"
```

## 6. Pushing

- Confirm the remote uses a working authentication method
- Push with `git push`
- If HTTPS credentials are problematic, switch the remote to SSH and retry

## 7. Validation checklist

Before pushing, verify:

- `git status -sb` is clean or only contains intended changes
- docs do not mention ignored or private-only components that should stay hidden
- local artifact directories are ignored and untracked
- README links, banner assets, and badges render correctly
- any changed training or evaluation entrypoint has been sanity-checked when feasible

## 8. Cleanup checklist

- Remove stale tracked artifacts from version control
- Keep `.gitignore` aligned with the actual research workflow
- Revisit module `README.md` files after major refactors
- Keep top-level `README.md` focused on public, maintained capabilities