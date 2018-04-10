# machine-learning

Equations: [Codecogs equation editor](https://www.codecogs.com/latex/eqneditor.php)

## Linear regression
### Hypothesis function
![h_theta](https://latex.codecogs.com/svg.latex?h_{\\theta}(x)=\\theta_{0}&plus;\\theta_{1}x)

#### Multi-feature
![h_theta_multi](https://latex.codecogs.com/svg.latex?h_\\theta(x)=\\theta^Tx=\\theta_0x_0&plus;\\theta_1x_1&plus;...&plus;\\theta_nx_n)

Note: element-wise multiplication.

#####octave

```octave example
theta=[1;2]
X=[2 3;4 5]
h_theta = theta'.*X
```

### Cost function
![j_theta](https://latex.codecogs.com/svg.latex?J(\\theta)=\\frac{1}{2m}\\sum_{i=1}^{m}(h_{\\theta}(x^{(i)})-y^{(i)})^{2})

### Gradient descent
![grad_desc](https://latex.codecogs.com/svg.latex?repeat\\,\\{\\theta_j\\leftarrow\\theta_j-\\alpha\\frac{\\delta}{\\delta\\theta_j}J(\\theta)\\}\\,until\\,convergence)

_j_: index in _&theta;_ vector

![derivative](https://latex.codecogs.com/svg.latex?\\frac{\\delta}{\\delta\\theta_j}\\frac{1}{2m}\\sum_{i=1}^{m}J(\\theta_0,\\theta_1)=\\\\\\frac{\\delta}{\\delta\\theta_j}\\frac{1}{2m}\\sum_{i=1}^{m}(h_\\theta(x^{(i))}-y^{(i)})=\\\\\\frac{\\delta}{\\delta\\theta_j}\\frac{1}{2m}\\sum_{i=1}^{m}(\\theta_0&plus;\\theta_1x^{(i)}-y^{(i)}))

![partial_deriv](https://latex.codecogs.com/svg.latex?\\frac{\\delta}{\\delta\\theta_j}): the partial derivative with respect to _&theta;<sub>j_. For example if we make the partial derivative with respect to _&theta;<sub>1_, for the summing part (_h<sub>0 -_ ) cost function above we treat the other terms (_&theta;<sub>0_ and _y_) as constants, which results \\in ![](https://latex.codecogs.com/svg.latex?\\frac{\\delta}{\\delta\\theta_1}(\\theta_0&plus;\\theta_1x-y)^2=2x(\\theta_0&plus;\\theta_1x-y))

**steps**:

![](https://www4f.wolframalpha.com/Calculate/MSP/MSP2382105ia0ee4i5hh15g00001969ggcecf13e8df?MSPStoreType=image/png&s=43)

## Classification: Logistic regression
### Hypothesis function
![](https://latex.codecogs.com/svg.latex?h_0(x)=g(\\theta^{T}x))

![](https://latex.codecogs.com/svg.latex?z=\\theta^{T}x)

![](https://latex.codecogs.com/svg.latex?g(z)=\\frac{1}{1+e^{-z}})

![](https://latex.codecogs.com/svg.latex?h_0(x)>=0.5\\rightarrow1)

![](https://latex.codecogs.com/svg.latex?h_0(x)<0.5\\rightarrow0)

### Cost function 
General cost function for linear regression
![](https://latex.codecogs.com/svg.latex?Cost(h_0(x),y)=\\frac{1}{2}(h_0(x)-y)^2)

General cost function for logistic regression

![](https://latex.codecogs.com/svg.latex?Cost(h_0(x^{(i)}),y^{(i)})=\\left\\{\\begin{matrix}-log(h_0(x))y=1\\\\-log(1-h_0(x)):y=0\\end{matrix}\\right.)

<=>

![](https://latex.codecogs.com/svg.latex?Cost(h_0(x),y)=-y(log(h_0(x)))-(1-y)(log(1-h_0(x))))

=>

![](https://latex.codecogs.com/svg.latex?J(\\theta)=-\\frac{1}{m}[\\sum_{i=1}^{m}y^{(i)}log(h_0(x^{(i)}))&plus;(1-y^{(i)})log(1-h_0(x^{(i)}))])

### Gradient descent algo

repeat
![](https://latex.codecogs.com/svg.latex?\\theta_{new}\\leftarrow\\theta-\\alpha\\frac{1}{m}\\sum_{i=1}^{m}[(h_0(x^{(i)})-y^{(i)})\\cdot&space;x^{(i)}])