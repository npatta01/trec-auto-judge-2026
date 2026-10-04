"""Strongest-support reduction for ONE original claim / document pair.

Callers must group by topic, answer, claim and document identity before calling.
Labels describe observed support; acceptance additionally requires complete
assessment and explicit contradiction screening. No partial-credit weights.
"""

LABELS = ('unsupported', 'partially_supported', 'supported')


def aggregate_chunks(rows, *, expected_chunk_ids):
    rows = list(rows)
    expected = list(expected_chunk_ids)
    if not expected or any(type(x) is not str or not x for x in expected):
        raise ValueError('Expected nonempty chunk IDs.')
    if len(set(expected)) != len(expected):
        raise ValueError('Duplicate expected chunk IDs.')
    seen = set()
    for row in rows:
        cid = row.get('chunk_id')
        if type(cid) is not str or cid not in expected or cid in seen:
            raise ValueError('Unexpected or duplicate chunk ID.')
        if row.get('label') not in LABELS or type(row.get('uncertain')) is not bool:
            raise ValueError('Invalid support judgment.')
        if 'contradiction' in row and type(row['contradiction']) is not bool:
            raise ValueError('Invalid contradiction flag.')
        seen.add(cid)
    complete = seen == set(expected)
    best = max((r['label'] for r in rows), key=LABELS.index, default=None)
    winners = [r for r in rows if r['label'] == best]
    # For negative conclusions every chunk matters; for positive evidence a
    # confident winner suffices, unless another chunk explicitly conflicts.
    uncertain = (any(r['uncertain'] for r in rows) if best == 'unsupported'
                 else not any(not r['uncertain'] for r in winners))
    conflict = any(r.get('contradiction', False) for r in rows)
    checked = complete and all('contradiction' in r for r in rows)
    return dict(label=best if complete else None, observed_best=best,
                complete=complete, missing_chunk_ids=sorted(set(expected)-seen),
                uncertain=uncertain or conflict or not complete or not checked,
                conflict=conflict, contradiction_checked=checked,
                accepted=complete and checked and best == 'supported'
                         and not uncertain and not conflict,
                winning_chunk_ids=sorted(r['chunk_id'] for r in winners))
