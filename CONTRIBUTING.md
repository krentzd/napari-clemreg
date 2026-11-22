# Contributing

## Branching strategy

This fork (`martlj/napari-clemreg`) is doing substantial modernisation work (see
[docs/napari-clemreg-modernisation-plan.md](docs/napari-clemreg-modernisation-plan.md))
while the upstream owner is away. To keep that work reviewable and `main` safe
to sync from upstream later, branches follow a simple structure:

- **`main`** — a clean mirror of upstream (`krentzd/napari-clemreg`). Nothing
  fork-specific is merged here. Keeping it untouched means `git fetch upstream
  && git merge upstream/main` stays a clean fast-forward whenever upstream
  moves, and it's what an eventual PR back to upstream would be based on.
  Don't merge `modernisation` into it until the upstream PR has been accepted
  or the upstream owner agrees otherwise (plan §7). At that point `main` can
  become the default branch again.
- **`modernisation`** — the integration branch for this fork's own work, and
  the fork's **default branch** (since 2026-09-23). Everything from the plan
  lands here first, via per-issue branches (below), not by committing to it
  directly once an issue has its own branch. Because it's the default, a plain
  `git clone` gets it, new PRs target it, scheduled and manual CI workflows
  run from it, and "Fixes #N" in a merged PR closes the issue automatically.
  (Before the switch, issues fixed by merged PRs had to be closed by hand.)
  GitHub matches those keywords anywhere in the description, including
  "doesn't fix #N", so only use them for issues the PR completely resolves
  (see [CLAUDE.md § Closing issues](CLAUDE.md#closing-issues)).
- **One branch per issue or theme**, branched off `modernisation`, named
  `<issue-number>-<short-slug>` (e.g. `6-mobie-export`, `12-em-mask-sample-data`).
  Open a PR into `modernisation` when the work is ready — a real PR gives a
  diff boundary, a description, CI results, and a revert point that a growing
  pile of commits on one branch doesn't.

**When Claude Code is doing the work**: it opens the branch, commits, pushes,
and opens the PR, but does **not** merge it — merging is left to a human
review, on request (2026-09-22).

Rationale for the split: `modernisation` started as a single branch named for
one specific task (splitting out the `clemreg` core, §6) and accumulated
unrelated work instead — testing, packaging, CI, bug fixes, docs. The name no
longer matched the content, and there was no PR boundary to review any of it
in reviewable chunks. Renamed and restructured 2026-09-22.

### Workflow

1. Pick up a GitHub issue.
2. `git checkout modernisation && git pull`
3. `git checkout -b <issue-number>-<short-slug>`
4. Do the work, commit (see commit message convention below), push.
5. Open a PR into `modernisation`.
6. A human reviews and merges once CI is green (Claude Code opens PRs but does not merge them — see above).
7. Delete the branch after merging.

When `modernisation` itself is ready to go back upstream (or once the
upstream owner is back to review), it becomes a single PR from
`modernisation` into `krentzd/napari-clemreg`'s `main` — see
[docs/napari-clemreg-modernisation-plan.md §7](docs/napari-clemreg-modernisation-plan.md#7-repository--release-strategy)
for the full repository/release plan.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/) —
`type(scope): short imperative description`, types: `feat`, `fix`, `docs`,
`style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.

## Before merging a branch into `modernisation`

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
