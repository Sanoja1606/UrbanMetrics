import numpy as np
import pandas as pd


def outlier_detect_arbitrary(data, col, upper_fence, lower_fence):
    """
    Identify outliers based on manually defined upper and lower boundaries.
    """
    outlier_index = (data[col] > upper_fence) | (data[col] < lower_fence)
    count = outlier_index.sum()
    print(f'Num of outliers detected: {count}')
    print(f'Proportion of outliers detected: {count / len(data):.4f}')
    return outlier_index, (upper_fence, lower_fence)


def outlier_detect_IQR(data, col, threshold=1.5):
    """
    Identify outliers using Interquartile Range (IQR) method.
    """
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_fence = Q1 - threshold * IQR
    upper_fence = Q3 + threshold * IQR
    outlier_index = (data[col] < lower_fence) | (data[col] > upper_fence)
    count = outlier_index.sum()
    print(f'Num of outliers detected: {count}')
    print(f'Proportion of outliers detected: {count / len(data):.4f}')
    return outlier_index, (upper_fence, lower_fence)


def outlier_detect_mean_std(data, col, threshold=3):
    """
    Identify outliers using mean and standard deviation method.
    """
    mean = data[col].mean()
    std = data[col].std()
    upper_fence = mean + threshold * std
    lower_fence = mean - threshold * std
    outlier_index = (data[col] > upper_fence) | (data[col] < lower_fence)
    count = outlier_index.sum()
    print(f'Num of outliers detected: {count}')
    print(f'Proportion of outliers detected: {count / len(data):.4f}')
    return outlier_index, (upper_fence, lower_fence)


def outlier_detect_MAD(data, col, threshold=3.5):
    """
    Identify outliers using Median Absolute Deviation (MAD) method.
    """
    median = data[col].median()
    mad = np.median(np.abs(data[col] - median))
    if mad == 0:
        print("MAD is zero. Cannot compute modified z-scores.")
        return pd.Series([False] * len(data)), None
    modified_z_scores = 0.6745 * (data[col] - median) / mad
    outlier_index = np.abs(modified_z_scores) > threshold
    count = outlier_index.sum()
    print(f'Num of outliers detected: {count}')
    print(f'Proportion of outliers detected: {count / len(data):.4f}')
    return outlier_index


def impute_outlier_with_arbitrary(data, outlier_index, value, cols):
    """
    Impute outliers with a custom value.
    """
    data_copy = data.copy()
    for col in cols:
        data_copy.loc[outlier_index, col] = value
    return data_copy


def windsorization(data, col, fences, strategy='both'):
    """
    Cap extreme values at the given thresholds.
    """
    data_copy = data.copy()
    upper, lower = fences
    if strategy in ['both', 'top']:
        data_copy.loc[data_copy[col] > upper, col] = upper
    if strategy in ['both', 'bottom']:
        data_copy.loc[data_copy[col] < lower, col] = lower
    return data_copy


def drop_outlier(data, outlier_index):
    """
    Remove rows where outlier_index is True.
    """
    return data[~outlier_index].copy()


def impute_outlier_with_avg(data, col, outlier_index, strategy='mean'):
    """
    Replace outliers with the column's mean, median, or mode.
    """
    data_copy = data.copy()
    if strategy == 'mean':
        value = data[col].mean()
    elif strategy == 'median':
        value = data[col].median()
    elif strategy == 'mode':
        value = data[col].mode()[0]
    else:
        raise ValueError("Strategy must be one of: 'mean', 'median', 'mode'")
    
    data_copy.loc[outlier_index, col] = value
    return data_copy
