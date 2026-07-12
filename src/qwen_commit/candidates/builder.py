"""Extract private, training-safe commit-message candidates from Git history."""

from __future__ import annotations

import logging
import subprocess
from collections import Counter
from pathlib import Path

from qwen_commit.candidates.errors import CandidateBuildError
from qwen_commit.candidates.filters import (
    is_fixup,
    is_generated_only,
    normalise_patch_text,
    normalise_subject,
)
from qwen_commit.candidates.models import (
    Candidate,
    CandidateBuildReport,
    CandidateRejectionReason,
    Provenance,
)
from qwen_commit.candidates.storage import write_parquet
from qwen_commit.candidates.utils import as_utc, opaque_id
from qwen_commit.history import (
    HistoryScanError,
    HistoryScanReport,
    RepositoryScanStatus,
)
from qwen_commit.history.utils import git_raw_text, git_text

logger = logging.getLogger(__name__)


def build_candidates(
    scan_report: HistoryScanReport,
    candidates_path: Path,
    provenance_path: Path,
    author_emails: tuple[str, ...],
) -> CandidateBuildReport:
    """Extract accepted commits from included repositories into private Parquet files."""
    configured_author_emails = frozenset(
        email.strip().casefold() for email in author_emails if email.strip()
    )
    if not configured_author_emails:
        raise CandidateBuildError("Configure at least one candidates.author_emails entry.")

    candidates: list[Candidate] = []
    provenance: list[Provenance] = []
    rejection_counts: Counter[CandidateRejectionReason] = Counter()
    inspected_count = 0

    logger.info(
        "Building candidates from %d commits across %d repositories",
        scan_report.commit_count,
        scan_report.included_repository_count,
    )

    for repository in scan_report.repositories:
        if repository.status is not RepositoryScanStatus.INCLUDED:
            continue
        if repository.shallow:
            raise CandidateBuildError(
                f"{repository.path}: shallow repository history cannot build candidates."
            )
        logger.info("Scanning repository: %s", repository.path)
        repository_group_id = opaque_id("repository", str(repository.path))
        remote_urls = _remote_urls(repository.path)
        for commit_sha in _commit_shas(repository.path):
            inspected_count += 1
            metadata = _commit_metadata(repository.path, commit_sha)
            candidate, rejected_as = _extract_candidate(
                repository.path,
                repository_group_id,
                commit_sha,
                metadata,
                configured_author_emails,
            )
            if rejected_as:
                rejection_counts[rejected_as] += 1
            else:
                if candidate is None:
                    raise CandidateBuildError(
                        f"{repository.path}: candidate extraction produced no result."
                    )
                candidates.append(candidate)
                provenance.append(
                    Provenance(
                        id=candidate.id,
                        repository_group_id=repository_group_id,
                        repository_path=str(repository.path),
                        remote_urls=remote_urls,
                        commit_sha=commit_sha,
                        author_name=metadata[2],
                        author_email=metadata[3],
                    )
                )
            if inspected_count % 100 == 0:
                logger.info(
                    "Processed %d/%d commits (%d accepted, %d rejected)",
                    inspected_count,
                    scan_report.commit_count,
                    len(candidates),
                    sum(rejection_counts.values()),
                )

        logger.info("Finished scanning repository: %s", repository.path)

    candidates.sort(key=lambda candidate: candidate.id)
    provenance.sort(key=lambda entry: entry.id)
    write_parquet(candidates, provenance, candidates_path, provenance_path)
    logger.info(
        "Candidate build complete: %d accepted, %d rejected",
        len(candidates),
        sum(rejection_counts.values()),
    )
    return CandidateBuildReport(
        scan_report=scan_report,
        candidates_path=candidates_path,
        provenance_path=provenance_path,
        accepted_count=len(candidates),
        rejection_counts=rejection_counts,
    )


def _extract_candidate(
    repository: Path,
    repository_group_id: str,
    commit_sha: str,
    metadata: tuple[tuple[str, ...], str, str, str, str],
    author_emails: frozenset[str],
) -> tuple[Candidate | None, CandidateRejectionReason | None]:
    parents, subject, _author_name, author_email, committed_at = metadata
    subject = normalise_subject(subject)
    if author_email.strip().casefold() not in author_emails:
        return None, CandidateRejectionReason.AUTHOR_NOT_MATCHED
    if len(parents) > 1:
        return None, CandidateRejectionReason.MERGE
    if not subject:
        return None, CandidateRejectionReason.EMPTY_SUBJECT
    if is_fixup(subject):
        return None, CandidateRejectionReason.FIXUP

    paths, has_binary = _change_stats(repository, commit_sha)
    if not paths:
        return None, CandidateRejectionReason.EMPTY_CHANGE
    if has_binary:
        return None, CandidateRejectionReason.BINARY
    if is_generated_only(paths):
        return None, CandidateRejectionReason.GENERATED_ONLY

    raw_diff = _commit_diff(repository, commit_sha)
    diff = normalise_patch_text(raw_diff)
    candidate_id = opaque_id("candidate", f"{repository_group_id}\0{commit_sha}")
    return (
        Candidate(
            id=candidate_id,
            repository_group_id=repository_group_id,
            subject=subject,
            diff=diff,
            committed_at_utc=as_utc(committed_at),
            # Use the raw diff so the patch_id matches `git patch-id` output exactly.
            patch_id=_patch_id(repository, raw_diff),
        ),
        None,
    )


def _commit_shas(repository: Path) -> tuple[str, ...]:
    return tuple(git_text(repository, "rev-list", "--all").splitlines())


def _commit_metadata(
    repository: Path, commit_sha: str
) -> tuple[tuple[str, ...], str, str, str, str]:
    fields = git_text(
        repository,
        "show",
        "--no-patch",
        "--format=%P%x00%s%x00%an%x00%ae%x00%cI",
        commit_sha,
    ).split("\0")
    if len(fields) != 5:
        raise CandidateBuildError(f"{repository}: could not read commit metadata for {commit_sha}.")
    parents, subject, author_name, author_email, committed_at = fields
    return (
        tuple(parents.split()),
        subject,
        author_name,
        author_email,
        committed_at.rstrip("\n"),
    )


def _change_stats(repository: Path, commit_sha: str) -> tuple[tuple[str, ...], bool]:
    output = git_text(
        repository,
        "diff-tree",
        "--root",
        "--no-commit-id",
        "--no-ext-diff",
        "--no-textconv",
        "--no-renames",
        "--numstat",
        "-r",
        "-z",
        commit_sha,
    )
    paths: list[str] = []
    has_binary = False
    for entry in output.split("\0"):
        if not entry:
            continue
        added, removed, path = entry.split("\t", maxsplit=2)
        paths.append(path)
        if added == "-" or removed == "-":
            has_binary = True
            continue
    return tuple(paths), has_binary


def _commit_diff(repository: Path, commit_sha: str) -> str:
    return git_raw_text(
        repository,
        "diff-tree",
        "--root",
        "--no-commit-id",
        "--no-ext-diff",
        "--no-textconv",
        "--no-renames",
        "--patch",
        "-r",
        "--format=",
        commit_sha,
    )


def _patch_id(repository: Path, diff: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repository), "patch-id", "--stable"),
        check=False,
        input=diff,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        text=True,
    )
    if completed.returncode != 0 or not completed.stdout:
        message = completed.stderr.strip() or "Git could not calculate a patch ID."
        raise CandidateBuildError(f"{repository}: {message}")
    return completed.stdout.split(maxsplit=1)[0]


def _remote_urls(repository: Path) -> tuple[str, ...]:
    remotes = git_text(repository, "remote").splitlines()
    urls: set[str] = set()
    for remote in remotes:
        try:
            urls.update(
                url
                for url in git_text(repository, "remote", "get-url", "--all", remote).splitlines()
                if url
            )
        except HistoryScanError:
            continue
    return tuple(sorted(urls))
