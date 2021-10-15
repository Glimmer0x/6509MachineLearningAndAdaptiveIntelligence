# 6509 Machine Learning and Adaptive Intelligence

- **Lecture and Lab Notes of  MLAI class, author: Glimmer**
- Lecturer → Dr. Mauricio A. Álvarez + Dr. Haiping Liu
- Reading List:
    - [ ]  Rogers and Girolami, A First Course in Machine Learning, Chapman and Hall/CRC Press, 2nd Edition, 2016
    - [ ]  Bishop, Pattern Recognition and Machine Learning, Springer-Verlag, 2006.
    - [ ]  Murphy, Machine Learning: A Probabilistic Perspective, MIT Press, 2012.
    - [ ]  Géron, Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow, O’Reilly, 2nd Edition, 2019.

---

# Part1 - Introduction to Machine Learning

- Basic definitions: **Training set, Estimation or training phase, Generalisation**
- Types of learning:
    - Supervised learning →classification (discrete target) and regression (continuous target)
    - Unsupervised learning → clustering (groups), density estimation (probability function), dimensionality reduction and visualisation (lower dimensionality)
    - Other types of learning: reinforcement learning, semi-supervised learning
- Main challenges:
    - Insufficient quantity of training data
    - Non-representative training data
    - Poor-quality data
    - Irrelevant features
    - Overfitting and underfitting the training data
- Review of probability:
    - Random variables → discrete RV and continuous RV,
    **Notation** → capital letters X, Y, Z denote RV and lowercase letters x, y, z denote value
    - Discrete RV → **joint probability mass function (pmf)** $P(X=x_i,Y=y_j)$ 
    →  $i=1,...,n$ and $j=1,...,m$
        - Properties: $P(X = x_{i}, Y = y_{j}) \geq 0$  , and $\sum_{i=1}^{n}\sum_{j=1}^{m} P(X=x_{i}, Y=y_{i}) =1$
        - Marginal pmf: $P(X=x_{i}) = \sum_{j=1}^{m} P(X=x_{i}, Y=y_{i})$ → the sum rule of probability
        - Conditional pmf: $P(X=x_{i}| Y=y_{j}) = \frac{P(X=x_{i}, Y=y_{j})}{P(Y=y_{j})}$, when $P(Y=y_{j}) \neq 0$
        or $P(X=x_{i} \mid Y=y_{j}) \times P(Y=y_{j}) = P(X=x_{i}, Y=y_{j})$  → product rule of probability
        - Sampling → $P(X=x_{i}, Y=y_{j}) = \lim_{N \rightarrow \infty} \frac{n_{X=x_{i}, Y=y_{j}}}{N}$
        
         → $P(X=x_{i}\mid Y=y_{j}) = \lim_{N \rightarrow \infty} \frac{n_{X=x_{i}, Y=y_{j}}}{N} \times \frac{N}{n_{Y=y_{j}}} \approx \frac{n_{X=x_{i}, Y=y_{j}}}{n_{Y=y_{j}}}$
        - Expected value and statistical moments (the mean and the variance)
        → $\mathbb{E}(g(X)) = \sum_{i=1}^{n} g(x_{i})P(X=x_{i})$
        → $\mu_{X} = \mathbb{E}(X)=\sum_{i=1}^{n}X_{i}P(X=x_{i})$, that is $g(x_{i})=x_{i}$
        
        → $\begin{aligned}
        \sigma_{X}^{2} &= var(X) 
        = \mathbb{E}\{(X - \mu_{X})^2\} \\
        &= \sum_{i}^{n} (x_{i} - \mu_{X})^2P(X=x_{i}) \\
        &= \mathbb{E}(X^2)-\mu_{X}^2
        \end{aligned}$ → $\sigma_X$ is the standard deviation
    - Continuous RV → **joint probability density function (pdf)** $p_{X,Y}(x,y)$
        - Properties: $p_{X,Y}(x,y) \geq 0$, $\int_{-\infty}^\infty \int_{-\infty}^\infty p_{X,Y}(x,y) \mathrm{d}x \mathrm{d}y= 0$
        → $P(a \leq X \leq b, c \leq Y \leq d) = \int_{a}^b \int_{c}^d p_{X,Y}(x,y) \mathrm{d} x \mathrm{d} y$
        - Marginal pdf: $p_X(x) = \int_{-\infty}^\infty p_{X,Y}(x,y) \mathrm{d}y= 0$
        - Conditional pdf: $p_{X \mid Y}(x \mid y) = \frac{p_{X,Y}(x,y)}{p_{Y}(y)}$
        - Expected value and statistical moments (the mean and the variance)
        →  $\mathbb{E}(g(X)) = \int_{-\infty }^{\infty} g(x)p_{X}(x) \mathrm{d}x$
        → $\mu_{X} = \mathbb{E}(X)=\int_{-\infty }^{\infty} x \cdot p_{X}(x) \mathrm{d}x$
        
        → $\begin{aligned}
        \sigma_{X}^{2} &= var(X) 
        = \mathbb{E}\{(X - \mu_{X})^2\} \\
        &= \int_{-\infty}^{\infty} (x - \mu_{X})^2p_{X}(x) \\
        &= \mathbb{E}(X^2)-\mu_{X}^2
        \end{aligned}$ → $\sigma_X$ is the standard deviation
    - Statistical independence → $P(X=x, Y=y)=P(X=x)P(Y=y)$
    - Bayes theorem → $P(X=x \mid Y=y) = P(Y=y \mid X=x)\times \frac{P(X=x)}{P(Y=y)}$
    - Estimators
    → $\hat{\mu}_X = \frac{1}{N} \sum_{k=1}^{N}X_k$
    → $\hat{\sigma^{2}}_X = \frac{1}{N-1} \sum_{k=1}^{N} (X_k - \hat{\mu}_X)^2$

---

# Part2 - An end-to-end ML project

- Metrics in regression
    - The root mean squared error $\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N [y_i-f(x_i, w)}]^2$
    - The mean absolute error $\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} \mid y_i - f(x_i,w)\mid$
- Metrics in classification
    - Confussion matrix: **TP** → true positive, **TN** → true negative, **FP** and **FN**
    → $\text{precision} = \frac{TP}{TP+FP}$ → correct positive predictions in all positive **predictions**
    → $\text{recall} = \frac{TP}{TP+FN}$ → correct positive predictions in all positive **examples**
    - Accuracy → $\text{accuracy} = \frac{TP+TN}{TP+TN+FP+FN}$
    → Acc is useful when all classes are equally important
    → But it can be misleading in imbalance classification problems
- Get the data: UC Irvine Machine Learning Repository, Kaggle, Amazon’s AWS datasets, etc.
→ legal obligations regarding the data
→ since 2018 the UK abides to the General Data Protection Regulation (GDPR) under EU law
    - Three sets: training, validation and test sets
    → for small datasets, use cross-validation → randomly split a validation set (size k) from a training set (size N) → **leave-one-out (LOO)** when k=1
- Explore the data
    - Visualise the data using histograms, scatterplots, maps, etc.
    - Study correlations between attributes
    → covariance $\rho_{X,Y} = \frac{\mathbb{E}\{(X-\mu_X)(Y-\mu_Y)\}}{\rho_X \rho_Y} = \frac{\rho_{X,Y}}{\rho_X \rho_Y}$, and $-1 < \rho_{X,Y} < 1$
    - Identify what transformations you might be able to apply
- Prepare the data
    - Data cleaning
    → Remove the **outliers**, optional
    → Handle the missing values → drop feature or instance, filling using the mean or others
    → Non-numerical features ⇒ one hot encoding
    - Feature selection and feature engineering → remove independent features relative to the target
    → Feature scaling ⇒ $log(x), \sqrt{x}$, etc. 
    → **normalization** (or min-max scaling) and **standardization** (or z-score normalization)
    → standardisation $\hat{X}_j = \frac{X_j - \mu_{j}}{\sigma_j}$, for feature $X_j$
- Shortlist models and then fine-tune them
→ Try different models from several categories (e.g. linear, non-linear, probabilistic, non-probabilistic)
→ shortlist two to three models that look promising
→ Fine-tune the hyperparameters of your model using cross-validation
→ Finally, use the test data to assess the generalisation error of your ML system.

---

# Part3 Decision trees and ensemble methods

## Decision trees

- Definitions →a root node + interior nodes + leaf nodes
⇒ non-leaf nodes → contain query’s descriptive features
⇒ leaf nodes → specifies a predicted classification
    - Informative features → a feature make homogeneous sets
- **Shannon's Entropy** → always set $s=2$
    - $H(t,\mathcal{D}) = -\sum_{i\in \text{levels}(t)}(P(t=l)\times \mathrm{log}_{s}(P(t=l)))$
- **Information Gain** → find a feature to split instances while maximize IG
    - $IG(d,\mathcal{D})=H(t,\mathcal{D})+IV(d,\mathcal{D})$
    - Intrinsic value $IV(d, \mathcal{D})=\sum_{l \in \text { levels }(d)} \underbrace{\frac{\left|\mathcal{D}_{d=l}\right|}{|\mathcal{D}|}}_{\text {weighting }} \times \underbrace{H\left(t, \mathcal{D}_{d=l}\right)}_{{\text { entropy of }} \text { partition } \mathcal{D}_{d=1}}$
- Handling continuous features
    - Thresholds and partitions
    → sort instances and then calculate the medium value between instances
- The ID3 Algorithm (Iterative Dichotomizer 3) → Greedy
→ aim to create the shallowest tree
→ in a recursive, depth-first manner to maximize IG
- The CART Algorithm (Classification and Regression Trees)→ Greedy
→ each node to find the feature and threshold to maximize IG
- **Regression trees**
→ reduce the variance in the set of training examples at each of the leaf nodes
    - $var(t,\mathcal{D}) = \frac{\sum_{i=1}^{n}(t_i-\bar{t})^2}{n-1}$
    - minimizes $\mathbf{d}[\text { best }]=\underset{d \in \mathbf{d}}{\operatorname{argmin}} \sum_{l \in \text { levels }(d)} \frac{\left|\mathcal{D}_{d=l}\right|}{|\mathcal{D}|} \times \operatorname{var}\left(t, \mathcal{D}_{d=l}\right)$
- Gini index → Another commonly used measure of impurity → misclassify measurement
    - $\operatorname{Gini}(t, \mathcal{D})=1-\sum_{l \in \text { levels }(t)} P(t=l)^{2}$
- Information gain ratio → addressing problem that IG prioritizes features with many values
    - $G R(d, \mathcal{D})=\frac{IG(d, \mathcal{D})}{-\sum_{l \in \text { levels }(d)}\left(P(d=l) \times \log _{2}(P(d=l))\right)}=\frac{IG(d, \mathcal{D})}{IV(d, \mathcal{D})}$
- Overfitting and Tree Pruning
    - Pre-pruning → stop the recursive partitioning early
    - Post-pruning → prune the tree of the branches that cause over-fitting after growing
        - evaluate using validation set

## **Model Ensembles**

- subspace sampling ⇒ randomly selected subset of the descriptive features
- **boosting** 
→ iteratively creating models and adding them to the ensemble with a predefined amount
    
    → new model pays more attention to instances that previous models miss-classified
    
    → by incrementally adapting the dataset used to train the models, using weighted dataset
    
    - Weighted Dataset ⇒ **sampling without replacement**
    → In the beginning, weight of each instances is $\frac{1}{n}$
    → increasing the weight of incorrect instances (such as increasing 50%)
    - Prediction ⇒ Weighted majority vote
- **bagging** (or bootstrap aggregating)
→ trained on a random sample of the dataset known as **bootstrap samples**
→ each random sample is the same size as the dataset, using **sampling with replacement**
    - **Random forest**
    → the combination of bagging, subspace sampling, and decision trees
    - Prediction ⇒ Weighted average
- boosting VS. bagging: [https://analyticsindiamag.com/primer-ensemble-learning-bagging-boosting/](https://analyticsindiamag.com/primer-ensemble-learning-bagging-boosting/)