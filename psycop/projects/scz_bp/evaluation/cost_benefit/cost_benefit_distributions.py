import numpy as np
from scipy.stats import truncnorm


def sample_float_from_truncated_log_normal(
    mean: float, lower_bound: float, upper_bound: float, n: int = 10000
) -> np.ndarray:  # type: ignore
    # Log-normal distribution parameters
    # Calculate the parameters of the underlying normal distribution
    # Mean (mu) and standard deviation (sigma) of the underlying normal distribution
    sigma = 0.5  # You might need to adjust this to get the exact mean and skewness
    mu = np.log(mean) - 0.5 * sigma**2

    # Define the truncation bounds in terms of the standard normal distribution
    a, b = (np.log(lower_bound) - mu) / sigma, (np.log(upper_bound) - mu) / sigma

    # Sample from the truncated normal distribution
    truncated_normal = truncnorm(a, b, loc=mu, scale=sigma)

    return np.exp(truncated_normal.rvs(size=n))


def sample_int_from_truncated_normal(
    mean: int, std: int, lower_bound: int, n: int = 10000
) -> np.ndarray:  # type: ignore
    # Calculate the truncation bounds in terms of the standard normal distribution
    a, b = (lower_bound - mean) / std, np.inf

    # Create the truncated normal distribution
    truncated_normal = truncnorm(a, b, loc=mean, scale=std)

    return truncated_normal.rvs(size=n).astype(int)
