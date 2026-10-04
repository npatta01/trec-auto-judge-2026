"""Citation-format normalization contracts, independent of legacy scoring."""

import json

import pytest

from tests.report_fixtures import report


def test_preserves_priority_order_ties_and_uncited_units():
    from judges.generic.reports import normalize_report

    got = normalize_report(report())
    assert got["sentences"][0] == {
        "sentence_id": 0,
        "text": "The sample is blue.",
        "citations": ["d1", "d2"],
        "top_citations": ["d1", "d2"],
    }
    assert len(got["sentences"]) == 3
    assert got["sentences"][1]["citations"] == []
    assert "hidden-team" not in json.dumps(got)
    assert "run_id" not in json.dumps(got)


def test_missing_documents_do_not_remove_citation_links():
    from judges.generic.reports import normalize_report

    got = normalize_report(report(documents={}))
    assert got["documents"] == {}
    assert got["sentences"][0]["citations"] == ["d1", "d2"]


@pytest.mark.parametrize("index", [-1, 99])
def test_invalid_reference_index_is_not_silently_dropped(index):
    from judges.generic.reports import normalize_report
    from judges.generic.models import JudgeError

    with pytest.raises(JudgeError, match="index"):
        normalize_report(report([{"text": "Claim.", "citations": [index]}]))


def test_index_citations_resolve_to_supplied_documents():
    from judges.generic.reports import normalize_report

    item = report([{"text": "Claim.", "citations": [1, 0]}])
    item.references = ["d1", "d2"]
    got = normalize_report(item)
    assert got["sentences"][0]["citations"] == ["d2", "d1"]
    assert set(got["documents"]) == {"d1", "d2"}


def test_nonfinite_citation_priority_is_rejected():
    from judges.generic.reports import normalize_report
    from judges.generic.models import JudgeError

    with pytest.raises(JudgeError, match="Nonfinite"):
        normalize_report(
            report([{"text": "Claim.", "citations": {"d1": float("inf")}}])
        )
