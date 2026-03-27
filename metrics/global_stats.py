def fix_dim(x):
    if x.ndim < 3:
        return x[:, :, np.newaxis]
    return x


def get_stats(X_data, X_sbts, col=None):
    """
    Plot 1% and 99% percentiles, mean, and standard deviation for two input arrays.
    :params X_data: original data; [np.array]
    :params X_sbts: generated data; [np.array]
    :params col: name of each features; [list]
    """
    X_data, X_sbts = fix_dim(X_data), fix_dim(X_sbts)

    # calculate 1% and 99% percentiles for both arrays
    percentiles1 = np.percentile(X_data, [1, 99], axis=(0, 1))  # shape (2, D)
    lower_percentile1 = percentiles1[0, :]  # shape (D,)
    upper_percentile1 = percentiles1[1, :]  # shape (D,)

    percentiles2 = np.percentile(X_sbts, [1, 99], axis=(0, 1))  # shape (2, D)
    lower_percentile2 = percentiles2[0, :]  # shape (D,)
    upper_percentile2 = percentiles2[1, :]  # shape (D,)

    # calculate mean and standard deviation for both arrays
    mean1 = np.mean(X_data, axis=(0, 1))  # shape (D,)
    std1 = np.std(X_data, axis=(0, 1))  # shape (D,)

    mean2 = np.mean(X_sbts, axis=(0, 1))  # shape (D,)
    std2 = np.std(X_sbts, axis=(0, 1))  # shape (D,)
    min_data = X_data.min(axis=(0, 1))
    min_sbts = X_sbts.min(axis=(0, 1))

    max_data = X_data.max(axis=(0, 1))
    max_sbts = X_sbts.max(axis=(0, 1))

    if col is None:
        col = range(len(lower_percentile1))

    df = pd.DataFrame({
        'Feature': col,
        '1% Sim': lower_percentile1,
        '1% Gen': lower_percentile2,
        '99% Sim': upper_percentile1,
        '99% Gen': upper_percentile2,
        'Mean Sim': mean1,
        'Mean Gen': mean2,
        'Std Sim': std1,
        'Std Gen': std2,
        'Min Sim': min_data,
        'Min Gen': min_sbts,
        'Max Sim': max_data,
        'Max Gen': max_sbts
    })

    df.set_index('Feature', inplace=True)
    return df.round(3)