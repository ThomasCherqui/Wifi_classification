def split_dataset(x, y, test_proportion, random_generator=default_rng()):

    shuffled_indices = random_generator.permutation(len(x))
    n_test = round(len(x) * test_proportion)
    n_train = len(x) - n_test
    x_train = x[shuffled_indices[:n_train]]
    y_train = y[shuffled_indices[:n_train]]
    x_test = x[shuffled_indices[n_train:]]
    y_test = y[shuffled_indices[n_train:]]
    train_dataset = np.column_stack((x_train, y_train))
    test_dataset = np.column_stack((x_test, y_test))

    return train_dataset, test_dataset

