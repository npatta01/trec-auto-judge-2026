"""Synthetic reports shared by citation tests; no evaluation-run content."""

from autojudge_base import Report


def report(sentences=None, documents=None):
    return Report.model_validate(
        {
            "metadata": {
                "run_id": "synthetic",
                "team_id": "hidden-team",
                "topic_id": "t",
            },
            "responses": sentences
            if sentences is not None
            else [
                {
                    "text": "The sample is blue.",
                    "citations": {"d1": 100.0, "d2": 100.0},
                },
                {"text": "A second factual claim.", "citations": {}},
                {"text": "Conclusion.", "citations": {}},
            ],
            "documents": documents
            if documents is not None
            else {
                "d1": {"id": "d1", "text": "The sample is blue."},
                "d2": {"id": "d2", "text": "The sample is red."},
            },
        }
    )
