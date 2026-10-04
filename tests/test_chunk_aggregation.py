import pytest


def row(chunk, label, uncertain=False, contradiction=False):
    return dict(chunk_id=chunk, label=label, uncertain=uncertain,
                contradiction=contradiction)


@pytest.mark.parametrize('labels,want', [
    (['unsupported', 'supported', 'unsupported'], 'supported'),
    (['partially_supported', 'partially_supported'], 'partially_supported'),
    (['unsupported', 'unsupported'], 'unsupported'),
])
def test_strongest_support_not_average(labels, want):
    from judges.generic.chunk_aggregation import aggregate_chunks
    result = aggregate_chunks([row(str(i), x) for i, x in enumerate(labels)],
                              expected_chunk_ids=[str(i) for i in range(len(labels))])
    assert result['label'] == want
    assert result['accepted'] == (want == 'supported')
    assert result['complete']


def test_uncertain_positive_is_not_accepted():
    from judges.generic.chunk_aggregation import aggregate_chunks
    result = aggregate_chunks([row('a', 'supported', True)], expected_chunk_ids=['a'])
    assert result['label'] == 'supported' and result['uncertain']
    assert not result['accepted']


def test_confident_positive_survives_uncertain_irrelevant_chunk():
    from judges.generic.chunk_aggregation import aggregate_chunks
    result = aggregate_chunks([row('a', 'supported'), row('b', 'unsupported', True)],
                              expected_chunk_ids=['a', 'b'])
    assert result['accepted'] and not result['uncertain']


def test_contradiction_blocks_acceptance_without_fourth_class():
    from judges.generic.chunk_aggregation import aggregate_chunks
    result = aggregate_chunks([row('a', 'supported'), row('b', 'unsupported', contradiction=True)],
                              expected_chunk_ids=['a', 'b'])
    assert result['label'] == 'supported' and result['conflict']
    assert result['uncertain'] and not result['accepted']


def test_missing_chunks_cannot_produce_final_label():
    from judges.generic.chunk_aggregation import aggregate_chunks
    result = aggregate_chunks([row('a', 'supported')], expected_chunk_ids=['a', 'b'])
    assert result['label'] is None and not result['complete']
    assert result['observed_best'] == 'supported' and not result['accepted']


def test_legacy_missing_contradiction_flag_is_not_silently_false():
    from judges.generic.chunk_aggregation import aggregate_chunks
    result = aggregate_chunks([dict(chunk_id='a', label='supported', uncertain=False)],
                              expected_chunk_ids=['a'])
    assert not result['contradiction_checked'] and not result['accepted']


@pytest.mark.parametrize('rows,expected', [
    ([row('a', 'supported'), row('a', 'unsupported')], ['a']),
    ([row('b', 'supported')], ['a']),
    ([row('a', 'bad')], ['a']),
    ([row('a', 'supported', uncertain='false')], ['a']),
    ([], []),
    ([], ['a', 'a']),
])
def test_invalid_provenance_and_labels_rejected(rows, expected):
    from judges.generic.chunk_aggregation import aggregate_chunks
    with pytest.raises(ValueError):
        aggregate_chunks(rows, expected_chunk_ids=expected)
