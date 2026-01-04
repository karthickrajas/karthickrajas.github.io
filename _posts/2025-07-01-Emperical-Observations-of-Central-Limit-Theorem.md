---
title: Empericall observations of Central Limit Theorem
date: 2025-07-01
categories: [statistics]
tags: [CLT, theorem, normal]     # TAG names should always be lowercase
image: https://karthickrajas.github.io/assets/imgs/CLT.jpg
math : true
---

### Introduction

The Central Limit Theorem (CLT) is arguably the most important pillar of modern statistics. It provides the theoretical justification for why we can use the Normal distribution to make inferences about population means, even when the population itself is anything but normal.

This holds true regardless of the shape of the source population distribution, provided the variance is finite. Mathematically, as the sample size $n$ approaches infinity ($n \to \infty$):

$$\bar{X}_n \sim N\left(\mu, \frac{\sigma^2}{n}\right)$$

In simpler terms, as you increase the size of your samples, the "map" of those sample means will always start to look like a bell curve.

This has huge applications in the field of Data science and AI. It enables us to bridge the chaotic world of raw data to structured, predictable world of guassian statistics. Some of the applications of CLT include
1. Hypothesis testing , AB testing
2. Normality assumption in Regression models, Log power transformations
3. Weight initializations, Batch normalization in Deep learning
4. Monte carlo simulations

In this blog, we will try to take different distributions and see if they follow CLT and see if there any distribution which doesn't adhere to CLT.
---

### Experimental Setup: Commonly used Distributions and their Parameters

To empirically test the CLT, we analyze several probability distributions. Each distribution represents a different "starting shape" for our population. By varying the parameters and increasing the sample size , we can observe how quickly (or if) the sample means converge to normality.

| Distribution | Parameters Used | Description |
| --- | --- | --- |
| **Binomial** | $n=10, p=0.5$ | Discrete distribution representing successes in independent trials. |
| **Poisson** | $\lambda = 4$ | Expresses the probability of a given number of events occurring in a fixed interval. |
| **Uniform** | $a=0, b=1$ | All outcomes are equally likely within a defined range. |
| **Exponential** | $\beta = 1.0$ | Describes the time between events in a Poisson process; highly skewed. |
| **Gamma** | $k=2, \theta=2$ | A flexible, skewed distribution used for waiting times. |
| **Beta** | $\alpha=2, \beta=5$ | Defined on the interval [0, 1]; can be symmetric or skewed. |
| **Cauchy** | $x_0=0, \gamma=1$ | A "pathological" distribution with heavy tails and no defined mean/variance. |

---
### How to measure normality: Histogram, Boxplot, QQ Plot and statistical tests

Whether the distribution of a sample statistic is Normal or not, depends primarily on three things :
1. The nature of the original population distribution,  like Poisson in the given example
2. The nature of the sample statistic, like mean and IQR in the given example below
3. The sample size n, which is 12 in the given example. Larger the nn better one gets the idea about the distribution of the statistic, but it has nothing to do with Normal approximation per say. Here we start from smaller value of nn and go larger till we achieve normal distribution

To see whether these sample statistics have Normal distribution, we are employing three simple graphical techniques namely the histogram, box and whiskers plot and Normal Probability Plot (NPP). For Normal distribution the histogram should be bell shaped, the box and whiskers plot should be symmetric about the central line within the box, and the NPP should be a straight line. Thus we see that though the distribution of the sample mean of Poisson(25) is approximately Normal, that of the sample IQR is not, for n=25.

Example of poisson distribution: ![Poisson](https://karthickrajas.github.io/assets/imgs/poisson_example.png)


#### **Histogram**

The histogram provides a visual representation of the data's density.

* **For a Normal Distribution:** It should appear symmetric and "bell-shaped." As  increases, the spread (standard error) should narrow around the population mean.

#### **Boxplot**

The boxplot summarizes the distribution through quartiles.

* **For a Normal Distribution:** The median should be exactly in the center of the box, and the "whiskers" should be of roughly equal length with very few, if any, outliers.

#### **Quantile-Quantile (QQ) Plot**

The QQ plot compares the quantiles of our sample data against the theoretical quantiles of a standard normal distribution.

* **For a Normal Distribution:** The data points should fall along a straight 45-degree diagonal line. Deviations at the ends of the line indicate "heavy tails" or skewness that the CLT hasn't yet smoothed out.

Python code to generate the above graphs
```python
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def plot_distributions_combined(data_array, title_prefix=''):
    """
    Draws a histogram, boxplot, and QQ norm plot for the input array
    in a single figure with three subplots.

    Args:
        data_array (numpy.ndarray or list): The input data array.
        title_prefix (str): Prefix for the subplot titles.
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    # Histogram
    sns.histplot(data_array, kde=True, bins='auto', ax=axes[0])
    axes[0].set_title(f'{title_prefix} Histogram')
    axes[0].grid(axis='y', alpha=0.75)

    # Boxplot
    sns.boxplot(y=data_array, ax=axes[1])
    axes[1].set_title(f'{title_prefix} Boxplot')
    axes[1].set_ylabel('Value')
    axes[1].grid(axis='y', alpha=0.75)

    # QQ Norm Plot
    stats.probplot(data_array, dist="norm", plot=axes[2])
    axes[2].set_title(f'{title_prefix} QQ Norm Plot')

    plt.tight_layout()
    plt.show()

    return fig
```

#### Statistical tests for nomality : 

Rather than relying soley on visual inspection of histograms or QQ plots, these tests quantify the likelihood that your sample means are drawn from a guassian distribution.
The Shapiro-Wilk test is highly sensitive and typically preferred for smaller sample sizes, whereas D'Agostino's K-squared test focuses on the "shape" of the distribution by measuring its skewness and kurtosis. Using these tests in tandem provides a comprehensive statistical verification.

The Null Hypothesis ($H_0$)In the context of normality testing (for these 2 tests), the Null Hypothesis ($H_0$) is defined as:
- $H_0$: The data is sampled from a population that follows a normal distribution.

Interpreting the Results
- If $p > 0.05$: We fail to reject the null hypothesis. There is no significant evidence to suggest the data is non-normal (i.e., we assume it is normally distributed)
- If $p < 0.05$: We reject the null hypothesis. There is statistically significant evidence that the data does not follow a normal distribution


Python code to check for normality
```python
from scipy import stats
import numpy as np

def check_normality(data):
    """
    Performs Shapiro-Wilk, D'Agostino's K-squared, and Anderson-Darling tests for normality.

    Args:
        data (list or array): The numerical data to test.

    Returns:
        dict: A dictionary containing test names as keys and tuples of (statistic, p-value or critical values) as values.
    """
    results = {}

    # Shapiro-Wilk Test
    shapiro_stat, shapiro_p = stats.shapiro(data)
    results['Shapiro-Wilk'] = (shapiro_stat, shapiro_p)

    # D'Agostino's K-squared Test
    dagostino_stat, dagostino_p = stats.normaltest(data)
    results['D_Agostinos_K_squared'] = (dagostino_stat, dagostino_p)

    comment = ""
    if (shapiro_p < 0.05) or (dagostino_p < 0.05):
        comment = "The data is not normally distributed."
    else:
        comment = "The data is normally distributed."

    return results, comment
```
---

### How to run the simulation ? 

We can make use of the existing distribution samplers available in Numpy. These functions draws random samples from a several statistical distribution which models different processes.

Below is the python function for using different numpy functions with several parameters.

```python
import numpy as np


def generate_distribution(distribution, params):
    """
    Samples m values from a binomial distribution.

    Args:
        distribution and params
    """

    samples = distribution(**params)
    return samples


def run_trails(distribution, params, n_trails):
  ls_mean = []
  ls_std = []
  for i in range(n_trails):
    samples = generate_distribution(distribution, params)
    mean = np.mean(samples)
    std = np.std(samples)
    ls_mean.append(mean)
    ls_std.append(std)
  return ls_mean, ls_std
  
tests_ = [
    {'binomial': [np.random.binomial, {'n' :12, 'p':0.1}, 50],},
	{'poisson': [np.random.poisson, {'lam':0.001,}, 20],},
	{'uniform': [np.random.uniform, {'low':0, 'high':1,} ,2]},
	{'exponential': [np.random.exponential, {'scale':0.001,}, 10],},
	{'beta': [np.random.beta, {'a':2, 'b':2}, 40],},
	{'gamma': [np.random.gamma, {'shape':0.1, 'scale':1, }, 50],},
	{'cauchy': [np.random.standard_cauchy, {'size':10}, 1000],},
]

ls_figs = []
ls_mean =[]
ls_std = []
for test in tests_:
  for distribution, params in test.items():
    ls_mean, ls_std = run_trails(*params)
    fig = plot_distributions_combined(ls_mean, f"{distribution} || Params : {params[1]} || Num trails {params[2]} ||")
    ls_figs.append(fig)
    test_result , comment = check_normality(ls_mean)
    print(comment)
```
---
### Results and Discussion

<details open>
<summary>Binomial distribution</summary>

![Binomial](https://karthickrajas.github.io/assets/imgs/Binomial_experiment.png)

</details>

<details>
<summary>Poisson distribution</summary>

![Poisson](https://karthickrajas.github.io/assets/imgs/Poisson_experiment.png)

</details>

<details>
<summary>Un distribution</summary>

![Uniform](https://karthickrajas.github.io/assets/imgs/Uniform_experiment.png)

</details>

<details>
<summary>Exponential distribution</summary>

![Exponential](https://karthickrajas.github.io/assets/imgs/Exponential_experiment.png)

</details>

<details>
<summary>Beta distribution</summary>

![Beta](https://karthickrajas.github.io/assets/imgs/Beta_experiment.png)

</details>

<details>
<summary>Gamma distribution</summary>

<img src="https://karthickrajas.github.io/assets/imgs/Gamma_experiment.png" alt="Gamma distribution">

</details>

<details>
<summary>Cauchy distribution</summary>

<img src="https://karthickrajas.github.io/assets/imgs/Cauchy_experiment.png" alt="Cauchy distribution">
</details>

Further Qualitative observations:
<embed src="https://karthickrajas.github.io/assets/pdfs/distributional_observations.pdf" type="application/pdf" width="100%" height="800px" />

---

### The Outlier: Why the Cauchy Distribution Fails

During the analysis, you likely noticed that the **Cauchy distribution** refused to conform to the bell curve, no matter how large the sample size  became.

This happens because the CLT has a strict prerequisite: the population must have a **finite mean and finite variance**. The Cauchy distribution is "pathological"—its tails are so heavy that the integral used to calculate the mean does not converge. In mathematical terms, its variance is undefined ().

Because the "average" of a Cauchy sample is dominated by extreme outliers that do not diminish with larger , the sample mean remains as volatile as a single observation. It follows a Cauchy distribution regardless of , effectively "breaking" the Central Limit Theorem.

---

### Conclusion and Scope for Further Exploration

Our empirical analysis confirms that for most distributions, the CLT is remarkably robust. Even highly skewed distributions like the Exponential or Gamma eventually yield a beautiful bell curve for the sample mean once .

**Further Scope for Exploration:**

* **Rate of Convergence:** Investigate how the "Skewness" of the initial distribution affects the  required to reach normality (e.g., does a highly skewed Beta distribution require  vs ?).
* **The Lindeberg-Feller Condition:** Explore cases where the random variables being averaged are not identically distributed (Non-IID).
* **Multivariate CLT:** Extend this analysis to vectors and higher-dimensional data spaces.
