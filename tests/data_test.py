from babygrad.data import Sample, collate_seq


def test_collate_seq_row_already_at_max_length():
    """A row needing no padding keeps its features and is fully unmasked."""
    row = Sample(features=[5, 3, 9, 1], target=["a"])

    collated, mask = collate_seq(row, max_length=4)

    assert collated == [5, 3, 9, 1]
    assert mask == [1, 1, 1, 1]
    assert len(mask) == len(collated)


def test_collate_seq_row_shorter_than_max_length():
    """A short row is padded up to max_length, and the mask marks the pad cells 0."""
    row = Sample(features=[5, 3, 9], target=["a"])

    collated, mask = collate_seq(row, max_length=6)

    assert collated == [5, 3, 9, 0, 0, 0]
    assert mask == [1, 1, 1, 0, 0, 0]
    assert len(mask) == len(collated)
    assert sum(mask) == len(row.features)


def test_collate_seq_honours_custom_pad_id():
    """The pad token is configurable; the mask is unaffected by which id is used."""
    row = Sample(features=[5, 3, 9], target=["a"])

    collated, mask = collate_seq(row, max_length=5, pad_id=-1)

    assert collated == [5, 3, 9, -1, -1]
    assert mask == [1, 1, 1, 0, 0]
