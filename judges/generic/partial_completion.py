"""One exact isolated support contract; evidence choices remain model decisions."""

from enum import Enum
from typing import Literal, Union

from pydantic import Field, create_model

from .coverage_completion import KeyedOutput
from .models import JudgeError


FORMAT = """\nOutput format only: return one object under link, not a links list.
Do not output case, item or mentioned; code supplies these from existing coverage.
For supported or partially_supported, select one primary_pair_id from the
permitted pairs and state a nonblank supported_portion. That primary pair is
always selected. additional_contributions is an object with every permitted
pair ID as a required boolean key; true adds that pair, false does not add it.
Each selected pair must contribute to the stated portion. Supported requires
empty missing_or_unsupported. If evidence cannot establish a portion, use
not_established or uncertain, with no primary_pair_id. Keep all prior semantic
rules; do not invent support to satisfy the output contract."""


def keyed_partial(payload):
    items = payload["items"]
    if len(items) != 1:
        raise JudgeError("Keyed partial correction requires one isolated item.")
    pair_ids = items[0]["pair_ids"]
    if (not pair_ids or len(pair_ids) != len(set(pair_ids))
            or not set(pair_ids) <= {p["id"] for p in payload["answer"]["pairs"]}):
        raise JudgeError("Invalid permitted isolated evidence selection.")
    primary = Enum("PartialPrimaryPair", {pid: pid for pid in pair_ids}, type=str)
    selection = create_model("PartialAdditionalContributions", __base__=KeyedOutput,
        **{pid: (bool, ...) for pid in pair_ids})
    fields = dict(primary_pair_id=(primary, ...),
        additional_contributions=(selection, ...),
        supported_portion=(str, Field(..., min_length=1, pattern=r"\S")))
    supported = create_model("KeyedSupportedPartial", __base__=KeyedOutput,
        status=(Literal["supported"], ...),
        missing_or_unsupported=(str, Field(..., max_length=0)), **fields)
    partial = create_model("KeyedPartialSupport", __base__=KeyedOutput,
        status=(Literal["partially_supported"], ...),
        missing_or_unsupported=(str, ...), **fields)
    negative = create_model("KeyedUnestablishedPartial", __base__=KeyedOutput,
        status=(Literal["not_established", "uncertain"], ...),
        additional_contributions=(selection, ...),
        supported_portion=(str, ...), missing_or_unsupported=(str, ...))
    return create_model("KeyedPartialOutput", __base__=KeyedOutput,
        link=(Union[supported, partial, negative], ...))


def restore_partial(result, *, case, item, mentioned):
    value = result.model_dump(mode="json")["link"]
    primary = value.pop("primary_pair_id", None)
    selections = value.pop("additional_contributions")
    # Keys select a set. A primary choice is explicit, never inferred evidence.
    pair_ids = [pid for pid, selected in selections.items() if selected or pid == primary]
    return dict(case=case, item=item, mentioned=mentioned, pair_ids=pair_ids, **value)
