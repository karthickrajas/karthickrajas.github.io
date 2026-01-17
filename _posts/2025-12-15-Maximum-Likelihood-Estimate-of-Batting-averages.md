---
title: Maximum Likelihood Estimation of Batting averages
date: 2025-12-15
categories: [ML]
tags: [classical-ml, ml, mle, statistics, survival, hazzard, reliability ]     # TAG names should always be lowercase
image: https://karthickrajas.github.io/assets/imgs/batting_average.png
math : true
---

Statistics play a huge role, behind the scenes, in any modern sport. For those who enjoy movies, there is no better way to explain it than watching `money ball` staring brad pitt. 

This is true for cricket as well & one of the metrics that is important for cricket is player's batting average. Batting average is common statistical measure to evaluate a player's performance across several innings (especially during recent times). 

### How are batting averages calculated ?

Batting average, in cricket, is calculated based on the below formula

$$
Batting Average = (Total Runs Scored)/(Number of Times out)
$$

The key point here is to note that, the innings where the batter remains `not out` do not count towards the denominator. A higher batting average generally means consistent performance and a reliable player.

Conceptually it makes sense, to have only `completed innings` as part of the denominator. If the player is not dismissed by the opponent before reaching a particular target, even if the game is over, the innings is considered `incomplete`. Dividing total runs by total innings (including not outs) would artificially lower the average by treating those incomplete innings as fully finished innings.

Also, practically it can inflate averages as well. For example, MS Dhoni and Michael Bevan have unusually high averages because of frequent not-out innings, even though those innings sometimes involve shorter batting opportunities.

**But what is the statistical validatity to do so ?**

### Statistical estimation of batting average (parameter $\theta$) using MLE

The conventional knowledge suggests to calculate parameter estimates for a given data, we would have to follow the below steps

1. Define a probability distribution
2. Formulate the Likelihood function
3. Determination of Log-Likelihood function
4. Maximize the Log-likelihood function
5. Solve the expression for maximum likelihood estimate of the parameter

#### Probability distribution

Here we will be considering exponential distrubution for modelling the runs untill dismissal for practical reasons as listed below

1. Memoryless property : meaning the probability of dismissal in the next run is independent of how many runs have been scored so far.
2. Mathematical simplicity : It has only one parameter (rate $\lambda$)
3. Serves as a baseline model for survival and reliability analysis, which can handle censorded data (incomplete innings).

However, it is important to note that exponential distrubution is helping us in simplifying the proccess. It would be much wiser to assume **Gamma, Weibull** distributions, but can be difficult with many parameters.

#### Formulating the likelihood function

The exponential distribution models a random variable $X$ that represents *time until an event* (runs until dismissal):

$$
f(x; \lambda) = \lambda e^{-\lambda x}, \quad x > 0
$$

where $\lambda > 0$ is the **rate** parameter (mean = $1/\lambda$).

Suppose you observe $n$ innings:

- Some innings ended with the player **out** → event observed.
- Others ended **not out** → event not observed (right-censored).

Denote:

- $x_i$: the runs scored in the $i$-th inning
- $\delta_i = 1$: if out (event observed)
- $\delta_i = 0$: if not out (right-censored)

For each observed event:

- **If out**: we know exactly $x_i$, so the density $f(x_i; \lambda) = \lambda e^{-\lambda x_i}$
- **If not out**: we only know the player survived (was not dismissed) up to $x_i$, so we use the **survival function**:

$$
S(x_i; \lambda) = P(X > x_i) = e^{-\lambda x_i}
$$

The combined likelihood across all $n$ innings is the product:

$$
L(\lambda) = \prod_{i=1}^n [f(x_i; \lambda)]^{\delta_i} [S(x_i; \lambda)]^{1 - \delta_i}
$$

Substitute the two expressions:

$$
L(\lambda) = \prod_{i=1}^n (\lambda e^{-\lambda x_i})^{\delta_i} (e^{-\lambda x_i})^{1 - \delta_i}
$$

#### Determination of Log-likelihood function

Simplify:

$$
L(\lambda) = \lambda^{\sum \delta_i} e^{-\lambda \sum x_i}
$$

Take the log to simplify multiplication:

$$
\ell(\lambda) = (\sum \delta_i) \ln(\lambda) - \lambda \sum x_i
$$

#### Maximize the Log-likelihood function

Set the derivative of $\ell(\lambda)$ with respect to $\lambda$ equal to zero:

$$
\frac{d\ell}{d\lambda} = \frac{\sum \delta_i}{\lambda} - \sum x_i = 0
$$

Solve for $\lambda$:

$$
\hat{\lambda} = \frac{\sum \delta_i}{\sum x_i}
$$

#### Solve the expression for maximum likelihood estimate of the parameter

The MLE for the **mean runs between dismissals** is the reciprocal:

$$
\hat{\theta} = \frac{1}{\hat{\lambda}} = \frac{\sum x_i}{\sum \delta_i}
$$

That’s **total runs divided by the number of dismissals** — exactly the traditional batting average formula!

### Intuitive Understanding based on survival analysis

- When a player is *not out*, you only know they’ve scored *at least* that many runs; the innings is censored.
- The exponential model assumes constant hazard (player's probability of getting out (dismissed) at any given run scored is constant). This is a **reasonable assumption**.
- The MLE adapts naturally by using only observed dismissals in the denominator.

In short: the **MLE derivation** justifies the classical cricket batting average formula in a formal probabilistic framework. It proves that excluding *not outs* from the denominator is statistically correct because those are censored observations and contribute only to the likelihood’s exponential term.