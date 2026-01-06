---
title: Maximum likelihood
date: 2025-07-01
categories: [Linear algebra]
tags: [positive definite, deep learning, matrices, gradient descent]     # TAG names should always be lowercase
image: https://karthickrajas.github.io/assets/imgs/PDM.png
math : true
---

### Introduction

The default configuration that one would use for training a regression model is **MSE - Mean squared error** as loss function and **$R^2$** or **MAE or MAPE** as metrics. The metrics are intutive, they tell us, "how far off our predictions are on average ?". But a novice data scientists and students might question, "why do we almost never train our models to minimize/maximize them directly ?".

MSE is defined as $$J(w) = \frac{1}{n} \|Xw - y\|^2$$

On the surface, Squared errors disproportionately punishes large outliers that moves us away from the "real-world" units of our target variable. This roughly translate that if your predictions are wrong by large margin, then your loss would be disproportionately higher and your algorithm would be penalized more for this than a prediction which is closer to target. However, the preference for MSE isn't just a mathematical quirk; it’s a fundamental bridge between matrix calculus, statistics, and the way machines actually "learn."

In this blog, We will try to discuss the below sections which ties up many subjects.

1. MSE is a result of maximum likelihood estimate of linear regression equation
2. Importance of Positive Definite matrix in a loss function
3. What happens if we go non linear [deep learning], will the effect remain the same ?

---

### Maximum likelihood estimate

MLE is a powerful statistical method to find the parameters of a model that best fit observed data by maximizing the probability (likelihood) of seeing that data. We can do this by following the below generic steps 

1. Define model assumptions
2. Construct the likelihood function
3. Formulate the log-likelihood
4. Find the partial derivates wrt parameters [ discussed in sec 2]
5. Solve the partial derivates and arrive at parameters [ discussed in sec 2]


#### The Probabilistic Assumption

In linear regression, we assume the relationship between inputs $X$ and targets $y$ is:

$$y = Xw + \epsilon$$

The key assumption for MLE is that the noise $\epsilon$ follows a Gaussian (Normal) distribution with mean $0$ and some constant variance $\sigma^2$:

$$\epsilon \sim \mathcal{N}(0, \sigma^2)$$

This implies that for a given $X$ and $w$, the target $y$ is also Gaussian:

$$p(y|X; w, \sigma^2) = \mathcal{N}(Xw, \sigma^2)$$

#### The likelihood function

For $n$ independent and identically distributed (i.i.d.) samples, the Likelihood $L(w)$ is the joint probability of all observed targets $y$ given the weights $w$:

$$L(w) = \prod_{i=1}^{n} p(y_i|x_i; w, \sigma^2)$$

Plugging in the Gaussian probability density function:

$$L(w) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(y_i - x_i^T w)^2}{2\sigma^2} \right)$$

#### The log likelihood function

Maximizing a product is difficult, so we take the natural logarithm to turn it into a sum. Since $\ln$ is a monotonic function, the $w$ that maximizes the likelihood will also maximize the Log-Likelihood ($\ell$):

$$\ell(w) = \ln L(w) = \sum_{i=1}^{n} \ln \left[ \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(y_i - x_i^T w)^2}{2\sigma^2} \right) \right]$$

Using log rules ($\ln(ab) = \ln a + \ln b$ and $\ln(e^x) = x$):

$$\ell(w) = \sum_{i=1}^{n} \left[ -\ln(\sqrt{2\pi\sigma^2}) - \frac{(y_i - x_i^T w)^2}{2\sigma^2} \right]$$

$$\ell(w) = -n \ln(\sqrt{2\pi\sigma^2}) - \frac{1}{2\sigma^2} \sum_{i=1}^{n} (y_i - x_i^T w)^2$$

#### From MLE to MSE

To find the optimal $w$, we maximize $\ell(w)$. In this equation:

1. The term $-n \ln(\sqrt{2\pi\sigma^2})$ is a constant with respect to $w$.

2. The term $\frac{1}{2\sigma^2}$ is a constant multiplier.

Therefore, maximizing $\ell(w)$ is equivalent to minimizing the negative of the summation part:

$$\text{maximize } \ell(w) \iff \text{minimize } \sum_{i=1}^{n} (y_i - x_i^T w)^2$$

In matrix form, the sum of squared differences is $\|Xw - y\|^2$. To turn this into the mean squared error, we simply divide by $n$ (another constant that doesn't change where the minimum is):

$$J(w) = \frac{1}{n} \|Xw - y\|^2$$

**Minimizing MSE is equivalent to finding the Maximum Likelihood Estimate under the assumption of Gaussian noise.**

### Importance of Positive Definite matrix in a loss function

Continuing from where we left in previous section ., Let $X$ be an $n \times d$ design matrix (where $n$ is the number of samples and $d$ is the number of features), $y$ be an $n \times 1$ target vector, and $w$ be the $d \times 1$ weight vector.

The Mean Squared Error (MSE) objective function $J(w)$ is defined as:

$$J(w) = \frac{1}{n} \|Xw - y\|^2$$

Expanding this into matrix products:

$$J(w) = \frac{1}{n} (Xw - y)^T (Xw - y)$$

$$J(w) = \frac{1}{n} (w^T X^T X w - w^T X^T y - y^T X w + y^T y)$$

Since $w^T X^T y$ is a scalar, it is equal to its transpose $(y^T X w)$. We can combine them:

$$J(w) = \frac{1}{n} (w^T X^T X w - 2w^T X^T y + y^T y)$$

#### Differentiating to Solve for $w$

To find the minimum, we take the gradient with respect to $w$ and set it to zero ($\nabla_w J(w) = 0$).

Using the rules of matrix calculus:

1. $\frac{\partial}{\partial w} (w^T A w) = 2Aw$ (if $A$ is symmetric)
2. $\frac{\partial}{\partial w} (w^T b) = b$

$$\nabla_w J(w) = \frac{1}{n} (2X^T X w - 2X^T y) = 0$$

Dividing by the constants and rearranging:

$$X^T X w = X^T y$$

This yields the Normal Equation:

$$w = (X^T X)^{-1} X^T y$$

To guarantee that the solution $w$ is a minimum (and not a maximum or saddle point), we look at the second derivative, or the Hessian matrix:

$$\mathbf{H} = \nabla^2_w J(w) = \frac{2}{n} X^T X$$

**A critical point is a global minimum if the Hessian is Positive Definite.**

- A matrix $A$ is positive definite if $v^T A v > 0$ for all non-zero vectors $v$.

- In our case, $v^T (X^T X) v = (Xv)^T (Xv) = \|Xv\|^2$.

- Since the squared norm is always $\ge 0$, the matrix $X^T X$ is at least Positive Semi-Definite.

If $X$ has full column rank (meaning features are not perfectly redundant), $X^T X$ is strictly Positive Definite. This ensures the loss surface is a convex "bowl" with a single unique global minimum.

This is also the reason why we need feature which are less correlated for the **stability** of the model training. In the presence of correlated feature, the H matrix becomes non invertable, leading to lack of unique solution.

We got a closed form solution for our parameters. However, In the matrix solution $w = (X^T X)^{-1} X^T y$, we solve the entire system at once. This is computationally expensive ($O(d^3)$) and only works for linear systems. And if the problem is not linearly separable or/and require non linear method., what do we do then ?

### What happens if we go non linear [deep learning], will the effect remain the same ?

Almost all the items mentioned above still matters however we will be no longer able to find a global solution for the entire network.

1. Non-Linearity: Deep networks use activation functions (like ReLU or Sigmoid). The prediction becomes $y = \sigma(W_2 \sigma(W_1 X))$. You can no longer pull $W$ out into a simple linear form like $Xw$

2. Non-Convexity: Because of these layers and non-linearities, the loss surface is no longer a simple bowl. It is highly non-convex with many local minima, plateaus, and saddle points

3. No Closed-Form Solution: You cannot algebraically isolate the weight matrices $W$ in a deep network to solve for them in one step.

Due to above reason, we must use iterative methods like **Gradient Descent (GD)**.

### Next steps

Gradient Descent is different because:

1. Scalability: It calculates gradients using small batches of data, making it possible to train on millions of images.

2. Flexibility: It doesn't care if the function is non-linear (like a Deep Neural Network). As long as we can calculate the derivative (the "slope"), we can update the weights.

3. Local vs. Global: While the matrix solution finds the global minimum of a convex function, Gradient Descent in deep learning navigates a "mountainous" landscape to find a good enough local minimum.

We will see it in detail in a different technical note.

