"""Failure modes for the DAG knowledgebase core.

Every one of these is a *fail-closed* signal: the caller must stop, never
silently proceed (paper: Algorithm 1 lines 2, 6).
"""

from __future__ import annotations


class DagKBError(Exception):
    """Base class for all knowledgebase errors."""


class NotADag(DagKBError):
    """A mutation would introduce a cycle into the lineage graph."""


class LineageIncomplete(DagKBError):
    """A parent record, dependency boundary, or owner mapping is missing.

    Corresponds to Algorithm 1, line 2 ("a derived parent, dependency
    boundary, or owner mapping is missing -> return blocked").
    """


class HeadConflict(DagKBError):
    """A head update was rejected.

    Raised when the writer identity does not match the declared owner, or the
    owner sequence number does not strictly increase (paper Table 3 /
    Appendix A: "an owner accepts a head update only when the writer identity
    matches the declared owner and the sequence increases").
    """


class ImmutableViolation(DagKBError):
    """An attempt to overwrite an existing immutable record with new content."""


class MalformedDraft(DagKBError):
    """An incoming draft is missing required fields or references."""


class RevertRejected(DagKBError):
    """A revert failed a precondition (state, retention, or authorisation)."""
