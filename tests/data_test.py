from babygrad.data import split_train_val_test, split_feature_target, Dataset


def test_split_train_val_test():
    df = Dataset(rows=[[1, 2, 3] for _ in range(10)])
    train, val, test = split_train_val_test(df, 0.8, 0.1, 0.1)
    assert train.nrow == 8
    assert val.nrow == 1
    assert test.nrow == 1


def test_one_hot_mapping(monkeypatch):
    # Force random.sample(...) to preserve row order.
    monkeypatch.setattr("babygrad.data.random.sample", lambda rows, _: rows)

    data = Dataset(
        rows=[
            # --- train
            [1, 2, 3],
            [1, 2, "cat"],
            [1, 2, "dog"],
            [1, 2, 3],
            [1, 2, "cat"],
            [1, 2, "dog"],
            [1, 2, 3],
            [1, 2, "cat"],
            # --- val
            [1, 2, "dog"],
            # --- test
            [1, 2, "cat"],
        ],
        target_col_idx=2,
        one_hot=True,
    )
    train, val, test = split_train_val_test(data)

    assert train.one_hot_mapping is not None
    assert train.one_hot_mapping[3] == [1, 0, 0]
    assert train.one_hot_mapping["cat"] == [0, 1, 0]
    assert train.one_hot_mapping["dog"] == [0, 0, 1]

    assert train.one_hot_mapping == val.one_hot_mapping == test.one_hot_mapping


def test_one_hot_mapping_uses_training_labels_only(monkeypatch):
    # Force random.sample(...) to preserve row order.
    monkeypatch.setattr("babygrad.data.random.sample", lambda rows, _: rows)

    data = Dataset(
        rows=[
            # --- train
            [1, 2, 3],
            [1, 2, "cat"],
            [1, 2, "dog"],
            [1, 2, 3],
            [1, 2, "cat"],
            [1, 2, "dog"],
            [1, 2, 3],
            [1, 2, "cat"],
            # --- val
            [1, 2, "dog"],
            # --- test
            [1, 2, "unseen"],
        ],
        target_col_idx=2,
        one_hot=True,
    )

    train, _, _ = split_train_val_test(data)

    assert train.one_hot_mapping
    assert "unseen" not in train.one_hot_mapping


def test_one_hot_mapping_target_vectors(monkeypatch):
    # Force random.sample(...) to preserve row order.
    monkeypatch.setattr("babygrad.data.random.sample", lambda rows, _: rows)

    data = Dataset(
        rows=[
            # --- train
            [1, 2, 3],
            [1, 2, "cat"],
            [1, 2, "dog"],
            [1, 2, 3],
            [1, 2, "cat"],
            [1, 2, "dog"],
            [1, 2, 3],
            [1, 2, "cat"],
            # --- val
            [1, 2, "dog"],
            # --- test
            [1, 2, "unseen"],
        ],
        target_col_idx=2,
        one_hot=True,
    )

    train, _, _ = split_train_val_test(data)
    x_train, y_train = split_feature_target(train)

    assert train.one_hot_mapping
    assert x_train.rows == [[1, 2]] * 8
    assert y_train.nrow == 8
    assert y_train.rows == [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ]
