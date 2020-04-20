class: center, middle

## Artificial Intelligence

# Neural Networks &

# Deep Learning

<br>

Gerard Escudero, 2019

<br>

![:scale 50%](figures/neuron.png)

.footnote[[Source](http://www.gabormelli.com/RKB/Artificial_Neuron)]

---
class: left, middle, inverse

# Outline

* .cyan[Neural Networks]

* Deep Learning Arquitectures

* Reinforcement Learning

* References

---

# Artificial Neuron Model

.center[![:scale 55%](figures/neuron.png)]

.center[![:scale 60%](figures/neuron_model.png)]

.footnote[Source: [Artificial Neuron](http://www.gabormelli.com/RKB/Artificial_Neuron)]

---

# Perceptron

.cols5050[
.col1[

- Classification and regression

- Linear model

- Classification:

$$h(x)=f(\sum_{i=1}^n w_i x_i + b)$$

$$f(x)=step\ function$$

- Learning rule:

$$w_i'=w_i+\eta(h(x)-y)$$

]
.col2[

![:scale 90%](figures/hyperplane.png)

![:scale 60%](figures/step.png)

.tiny[.red[*] Source: [wikipedia](https://en.wikipedia.org/wiki/Heaviside_step_function)]

]]


---

# Perceptron in sklearn

```python3
from sklearn.linear_model import Perceptron

clf = Perceptron()
clf.fit(x_train, y_train)
predictions = clf.predict(x_test)
```

**Main parameters:**

- .blue[max_iter]: default=1000

#### User guide: <br>
.tiny[[https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html)]

---

# Multi-layer Perceptron

.col5050[
.col1[
- Hidden layers

- Non-linear model

- Classification & regression

- Forward propagation of perceptrons

- Backpropagation as training algorithm <br>
Gradient descent (optimization)

$$W^{t+1}=W^t-\eta\frac{\partial loss}{\partial W}$$

$$W=\{w,b\}$$

$$\eta=learning\ rate$$

$$loss=training\ error$$
]
.col2[
![:scale 110%](figures/mlp.png)
]]


.footnote[Source: [wikipedia](https://en.wikipedia.org/wiki/Artificial_neural_network)]

---

# Components

#### Loss functions

- Regression: minimum squared error or root minimum squared error

- Binary classification: binary cross entropy

- Multiclass classification: categorical cross entropy

#### Activation functions

- Hidden units: ReLU $f(x)=max(0,x)$

- Output
  - Regression: linear $f(x) = x$

  - Binary classification: sigmoid $f(x) = 1 / (1 + exp(-x))$

  - Multiclass classification: softmax (one unit per class) <br>
maximum value after normalizing as distribution

---

# Deep Learning

- Neural network with several hidden layers

![](figures/chart-1.png)

![](figures/chart-2.png)

.footnote[Source: [The Neural Network Zoo](https://www.asimovinstitute.org/neural-network-zoo/)]

---

# Deep Learning

**Software packages:**

- .blue[Tensorflow]: efficient, low-level math stuff, ugly learning curve

- .blue[Keras] ($\subset$ Tensorflow): high-level, nice learning curve

**Example:**

- Keras on MNIST
  - [view](codes/keras-mlp.html)
  - [download](codes/keras-mlp.ipynb)
  - [Source](https://github.com/keras-team/keras)

---
class: left, middle, inverse

# Outline

* .brown[Neural Networks]

* .cyan[Deep Learning Arquitectures]

* Reinforcement Learning

* References

---

# Convolutional Neural Networks

**from Computer Vision**

to process image & video

![:scale 90%](figures/cnn2.png)

.footnote[Source: [A Comprehensive Guide to Convolutional Neural Networks](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)]

---

# Convolutional Neural Networks II

.cols5050[
.col1[
- Convolution: extract the high-level features such as edges

- Pooling: reduce dimensionality for 
  - computational cost
  - extracting dominant features which are rotational and positional invariant

- Example:
  - Keras on MNIST
  - [view](codes/keras-cnn.html) / [download](codes/keras-cnn.ipynb)
  - [Source](https://github.com/keras-team/keras)
]
.col2[
![:scale 70%](figures/convolutional.gif)

![:scale 80%](figures/pooling.gif)
]]

.footnote[Source: [A Comprehensive Guide to Convolutional Neural Networks](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)]

---

# The Neural Network Zoo

![:scale 90%](figures/zoo-1.png)

.footnote[Source: [The Neural Network Zoo](https://www.asimovinstitute.org/neural-network-zoo/)]

---

# The Neural Network Zoo II

![:scale 90%](figures/zoo-2.png)

.footnote[Source: [The Neural Network Zoo](https://www.asimovinstitute.org/neural-network-zoo/)]

---

# The Neural Network Zoo III

![:scale 90%](figures/zoo-3.png)

.footnote[Source: [The Neural Network Zoo](https://www.asimovinstitute.org/neural-network-zoo/)]

---
class: left, middle, inverse

# Outline

* .brown[Neural Networks]

* .brown[Deep Learning Arquitectures]

* .cyan[Reinforcement Learning]

* References

---

# Reinforcement Learning

![:scale 65%](figures/rl.png)

.cols5050[
.col1[
- an _agent_
- a set of states $S$
- a set of actions $A$

]
.col2[
Learning a reward function $Q: S \times A \to \mathbb{R}$ for maximizing the total future reward.

]]

- _Q-Learning_: method for learning an aproximation of $Q$.

.footnote[Source: [My Journey Into Deep Q-Learning with Keras and Gym](https://medium.com/@gtnjuvin/my-journey-into-deep-q-learning-with-keras-and-gym-3e779cc12762)]

---

# Reinforcement Learning II

Training example:

.center[![:scale 55%](figures/q-matrix.png)]

.footnote[Source: [Reinforcement Q-Learning from Scratch in Python with OpenAI Gym](https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/)]

---

# Reinforcement Learning Example

- Simple example of Q-learning & Q-Table.

.center[[![:scale 65%](figures/ooops.png)](figures/ooops.mp4)]

---

# Deep Reinforcement Learning

- Convolutional Neural Network for learning $Q$ <br>

.center[[![:scale 65%](figures/breakout.png)](https://www.youtube.com/watch?v=TmPfTpjtdgg&feature=youtu.be)]

.footnote[Source: [Deep Reinforcement Learning](https://deepmind.com/blog/article/deep-reinforcement-learning)]


---
class: left, middle, inverse

# Outline

* .brown[Neural Networks]

* .brown[Deep Learning Arquitectures]

* .brown[Reinforcement Learning]

* .cyan[References]

---

# References

- Aurélien Géron. _Hands-On Machine Learning with Scikit-Learn, Keras, and Tensorflow_. O’Reilly, 2019.

- [Tensorflow](https://www.tensorflow.org)

- [Keras](https://keras.io/)

- [A Comprehensive Guide to Convolutional Neural Networks](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)

- [The Neural Network Zoo](https://www.asimovinstitute.org/neural-network-zoo/)

- [Deep Reinforcement Learning](https://deepmind.com/blog/article/deep-reinforcement-learning)

- [Unity MLAgents](https://github.com/Unity-Technologies/ml-agents)

- [OpenAI Gym](https://gym.openai.com/)

