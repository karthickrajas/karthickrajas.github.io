---
layout: post
excerpt: First step to reinforcement learning
images:
  - url: /assets/cmab.jpg
---

The IPL season 2019 is coming up in few days, with that in mind this is a small discussion that happened in my classroom. we were discussing Survival function in our class, particularly right censoring. The question that made us think was **why batting averages are caculated the way they are calculated?**

First of all **how they are calculated?**
Batting average is calculated as follows: Runs divided by (number of times out)

Lets take a simple example of a batsman's scoring history as 2,10,5*,8,9*, 6,4. He is not out for 2 innings and out for 5 innings. Hence 44/5.

**why they are calculated like we did above?**

Now starting with our survival analysis part of the theory. 

**Assumption:** Consider the chance of getting out in the next ball is independent of the number of runs he has already scored. Basically a constant hazard function. 

on the contrary one can also argue saying the more he scored, the more probable he gets out, but that would result in an increasing hazard function.

Exponential distribution, a special case of Weibull distribution, can be used to model the scores of a batsman which will give a constant Hazard rate and Survival rate as below.

* PDF = f(x) = λ* exp ( - λx )
* Survival Function = s(x) = exp ( - λx )
* Hazard function = f(x)/s(x) = λ

Writing the Likelihood function for the provided data.

λexp(-2λ) + λexp(-10λ) + λexp(-8λ) + λexp(-6λ) + λexp(-4λ)........ for Innings that he got out + exp(-2λ) + exp(-10λ) ........ for Innings that he didn't get out ( f(x) replaced by s(x) )

Taking the Log on both sides, Log Likelihood = 5log λ - 44 λ

Differentiating on both sides for λ and equating it to zero. we find λ = 5/44.

Also, Mean of the exponential distribution is 1/λ = 44/5.

Hence the batting average is calculated with the number of innings he got out, after right censoring the batsmen score when he is not out. The survival function of the batsmen based on the data can be given as.. exp(-5x/44)

Found it interesting.!

#### References

* [Prof.Abhinanda Sarkar ](http://mgmt.iisc.ac.in/newwordpress/abhinanda-sarkar/)
