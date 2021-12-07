# Angus Dempster, Daniel F Schmidt, Geoffrey I Webb

# MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series
# Classification

# https://arxiv.org/abs/2012.08791

from numba import njit, prange, vectorize
from itertools import combinations
import numpy as np
import math

def _fit_biases(X, dilations, num_features_per_dilation, quantiles, kernel_length, alpha_num):

    num_examples, input_length = X.shape

    indices = np.array([_ for _ in combinations(np.arange(kernel_length), alpha_num)], dtype = np.int32)

    num_kernels = len(indices)
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)

    biases = np.zeros(num_features, dtype = np.float32)

    feature_index_start = 0

    for dilation_index in range(num_dilations):

        dilation = dilations[dilation_index]
        padding = ((kernel_length - 1) * dilation) // 2

        num_features_this_dilation = num_features_per_dilation[dilation_index]

        for kernel_index in range(num_kernels):

            feature_index_end = feature_index_start + num_features_this_dilation

            _X = X[np.random.randint(num_examples)]

            A = -_X          # A = alpha * X = -X
            G = _X + _X + _X # G = gamma * X = 3X

            C_alpha = np.zeros(input_length, dtype = np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((kernel_length, input_length), dtype = np.float32)
            C_gamma[kernel_length // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(kernel_length // 2):

                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]

                end += dilation

            for gamma_index in range(kernel_length // 2 + 1, kernel_length):

                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]

                start += dilation

            index_0, index_1, index_2 = indices[kernel_index]

            C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

            biases[feature_index_start:feature_index_end] = np.quantile(C, quantiles[feature_index_start:feature_index_end])

            feature_index_start = feature_index_end

    return biases

def _fit_dilations(input_length, num_features, max_dilations_per_kernel, kernel_length, alpha_num):

    num_kernels = Cnn(kernel_length, alpha_num)

    # 一个kernel能有几个特征
    num_features_per_kernel = num_features // num_kernels

    # 一个kernel真正的dilation数量，看一个kernel有多少特征，每个dilation都可以计算一个特征，但最多32
    true_max_dilations_per_kernel = min(num_features_per_kernel, max_dilations_per_kernel)

    # TODO 如果需要的特征超过32，这里有个multiplier可能会有用
    multiplier = num_features_per_kernel / true_max_dilations_per_kernel

    # 
    max_exponent = np.log2((input_length - 1) / (kernel_length - 1))
    dilations, num_features_per_dilation = \
    np.unique(np.logspace(0, max_exponent, true_max_dilations_per_kernel, base = 2).astype(np.int32), return_counts = True)
    num_features_per_dilation = (num_features_per_dilation * multiplier).astype(np.int32) # this is a vector

    remainder = num_features_per_kernel - np.sum(num_features_per_dilation)
    i = 0
    while remainder > 0:
        num_features_per_dilation[i] += 1
        remainder -= 1
        i = (i + 1) % len(num_features_per_dilation)

    return dilations, num_features_per_dilation

# low-discrepancy sequence to assign quantiles to kernel/dilation combinations
# 有多少kernel有多少quantile，分位数，模1得到
def _quantiles(n):
    return np.array([(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)], dtype = np.float32)

def fit(X, num_features = 10_000, max_dilations_per_kernel = 32, kernel_length = 9, alpha_num = 3):

    _, input_length = X.shape

    num_kernels = Cnn(kernel_length, alpha_num)

    dilations, num_features_per_dilation = _fit_dilations(input_length, num_features, max_dilations_per_kernel, kernel_length, alpha_num)

    num_features_per_kernel = np.sum(num_features_per_dilation)

    quantiles = _quantiles(num_kernels * num_features_per_kernel)

    biases = _fit_biases(X, dilations, num_features_per_dilation, quantiles, kernel_length, alpha_num)

    return dilations, num_features_per_dilation, biases

# _PPV(C, b).mean() returns PPV for vector C (convolution output) and scalar b (bias)
def _PPV(a, b):
    if a > b:
        return 1
    else:
        return 0

def Cnn(n, m):
    return math.factorial(n)//(math.factorial(m)*math.factorial(n-m))

def transform(X, parameters, kernel_length = 9, alpha_num = 3):

    num_examples, input_length = X.shape

    dilations, num_features_per_dilation, biases = parameters

    indices = np.array([_ for _ in combinations(np.arange(kernel_length), alpha_num)], dtype = np.int32)

    num_kernels = len(indices)
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)

    features = np.zeros((num_examples, num_features), dtype = np.float32)

    for example_index in prange(num_examples):

        _X = X[example_index]

        A = -_X          # A = alpha * X = -X
        G = _X + _X + _X # G = gamma * X = 3X

        feature_index_start = 0

        for dilation_index in range(num_dilations):

            _padding0 = dilation_index % 2

            dilation = dilations[dilation_index]
            padding = ((kernel_length - 1) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation[dilation_index]

            C_alpha = np.zeros(input_length, dtype = np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((kernel_length, input_length), dtype = np.float32)
            C_gamma[kernel_length // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(kernel_length // 2):

                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]

                end += dilation

            for gamma_index in range(kernel_length // 2 + 1, kernel_length):

                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]

                start += dilation

            for kernel_index in range(num_kernels):

                feature_index_end = feature_index_start + num_features_this_dilation

                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

                if _padding1 == 0:
                    for feature_count in range(num_features_this_dilation):
                        features[example_index, feature_index_start + feature_count] = _PPV(C, biases[feature_index_start + feature_count]).mean()
                else:
                    for feature_count in range(num_features_this_dilation):
                        features[example_index, feature_index_start + feature_count] = _PPV(C[padding:-padding], biases[feature_index_start + feature_count]).mean()

                feature_index_start = feature_index_end

    return features
