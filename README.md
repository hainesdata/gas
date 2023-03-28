# Modeling California Retail Gas Prices Using Linear Regression

### Jake Haines

*March 2023*

---

# Introduction

## Prologue

Gas prices in the United States have consistently been an area of interest due to heavy reliance on automotive transportation$^{[1]}$. Despite observable increase in EV usage in the US, California especially, gas expenses remain an issue for those who commute by internal combustion vehicles. The goal of this study was to make a small contribution to optimizing currently used modes of transportation. In doing so, I could explore processes of building a regression model.

In general, I was curious to see if price at a gas station was predictable given a set of parameters at a *fixed* time. This would serve as a precursor to predicting the price of a gas station given parameters at an **********indeterminate********** time. 

## Methodology

To establish a baseline for predicting gas prices intrinsically given fixed parameters, I aimed to gather data containing several parameters irrespective of time. 

To optimize feature selection, I examined properties, distributions, and associations with target variable of each potential feature. Features were dropped if they met any of the following criteria:

- Noisy:
    - Large number of possible categorical values that could not be grouped easily
- Feature $x$ has insignificant association with target variable $y$:
    - $f(x)=\hat y=ax+b$ where slope $a$ is significantly low
    - $f(x)=\hat y=ax+b$ where fit $R^2$ is significantly low
- Only had one possible value
- Aren’t practically involved
    - There is no physical attribution to target variable

This study covers development and assessment of a linear regression model. Other models should be considered beyond a linear regression, as natural occurrences are generally non-linear.

**To view the full paper, visit the Notion page or view in PDF.
**
