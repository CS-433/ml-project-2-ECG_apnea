from scipy import signal
import numpy as np


# filters an xn signal with lower and upper band frequencies
def bandpass_signal(xn, lower, upper):
    length = len(xn)
    t = np.linspace(-1, 1, length)

    b, a = signal.butter(3, [lower, upper], btype="bandpass", fs=100)

    zi = signal.lfilter_zi(b, a)

    z, _ = signal.lfilter(b, a, xn, zi=zi * xn[0])
    z2, _ = signal.lfilter(b, a, z, zi=zi * z[0])
    y = signal.filtfilt(b, a, xn)

    return y


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x


# function to handle nans in feature
def handle_nans(array):
    """shifts the nan values to the mean of the other defined values"""
    # We compute the mean of all values except nan
    # nan values are values that are not equal to themselves
    temp = [[1, x] if x == x else [0, 0] for x in array]

    sum_columns = np.sum(temp, axis=0)
    mean_of_array_without_nan = sum_columns[1] / sum_columns[0]
    for i in range(len(array)):
        if array[i] != array[i]:
            array[i] = mean_of_array_without_nan



def confusion(prediction, truth):
    """Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float("inf")).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives