# Contributing

## Branching strategy

The modernisation work (see
[docs/napari-clemreg-modernisation-plan.md](docs/napari-clemreg-modernisation-plan.md))
began on the fork `martlj/napari-clemreg`, on its `modernisation` branch.
Since 2026-11-25 it continues in this repository, `krentzd/napari-clemreg`:

- **`main`** — the published version and the default branch. This work
  doesn't go into it until it's ready to release.
- **`version-2`** — the integration branch. It started from the fork's
  `modernisation` branch. Everything lands here via per-issue branches
  (below), not by committing to it directly.
- **One branch per issue or theme**, branched off `version-2`, named
  `<issue-number>-<short-slug>` (e.g. `6-mobie-export`, `12-em-mask-sample-data`).
  Open a PR into `version-2` when the work is ready — a real PR gives a
  diff boundary, a description, CI results, and a revert point that a growing
  pile of commits on one branch doesn't.

Most existing issues are on the fork. In PRs here, refer to them as
`martlj/napari-clemreg#N`: a bare `#N` means this repository's issue. Merging
into `version-2` doesn't close issues, because it isn't the default branch, so
close them by hand. Still only write a closing keyword ("Fixes ...") before an
issue the PR completely resolves: it fires once `version-2` merges into `main`
(see [CLAUDE.md § Closing issues](CLAUDE.md#closing-issues)).

**When Claude Code is doing the work**: it opens the branch, commits, pushes,
and opens the PR, but does **not** merge it unless asked — merging is left to
a human review (2026-09-22).

### Workflow

1. Pick up a GitHub issue.
2. `git checkout version-2 && git pull upstream version-2`
3. `git checkout -b <issue-number>-<short-slug>`
4. Do the work, commit (see commit message convention below), push to `upstream`.
5. Open a PR into `version-2`.
6. A human reviews and merges once CI is green.
7. Delete the branch after merging.

When `version-2` is ready to release, it becomes a single PR into `main` — see
[docs/napari-clemreg-modernisation-plan.md §7](docs/napari-clemreg-modernisation-plan.md#7-repository--release-strategy)
for the full repository/release plan.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/) —
`type(scope): short imperative description`, types: `feat`, `fix`, `docs`,
`style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.

## Before merging a branch into `version-2`

- Run the test suite (`pixi run -e test test`, `tox`, or `pytest` against an
  active env — see [CLAUDE.md](CLAUDE.md) for which local env to use and why).
- Check CI is green.
- Update [CHANGELOG.md](CHANGELOG.md) (an entry under `[Unreleased]`) and
  [ROADMAP.md](ROADMAP.md) (remove what's finished, add follow-ups) in the
  PR itself. See [CLAUDE.md § CHANGELOG and ROADMAP](CLAUDE.md#changelog-and-roadmap).
- Update the relevant GitHub issue (and
  [docs/napari-clemreg-modernisation-plan.md](docs/napari-clemreg-modernisation-plan.md)
  if the change affects the plan) rather than letting them drift out of sync
  with what actually landed.
