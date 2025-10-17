# Overview and Relevance
T-SNE is a dimension reduction algorithm that is intended to preserve characteristics of higher dimensional data when it is projected into lower dimensions.

Many phenomena can be described using many variables in high dimension, for instance breast cancer tumors that reside in 30 dimensional data. 

This particular method, TSNE was very relevant for its time of publication in 2002 Our goal is to convert a dataset, $X=\{x_{1},x_{2},\cdots,x_{n} \}$ where each $x_{i} \in \mathbb{R}^d$ to a dataset $\mathcal{Y}=\{y_{1},y_{2},\cdots,y_{n}\}$ where each $y_{i}$ is in either $\mathbb{R},\mathbb{R}^2, \mathbb{R}^3$. 

### Drawbacks of Other Methods 
PCA works great but it is linear, and has inherent drawbacks due to the structure projected onto. PCA and Classical MDS (Multidimensional Scaling) prefer to keep data far apart. However, for data points that are on a nonlinear surface near to each other, we tend to want to keep these points together rather than spread them apart. 

T-SNE converts $X$ into a dataset of pairwise similarities (a new matrix which consists of the same entries that tells us how "similar" the entries are).

T-SNE is intended to really preserve local data very well, but also preserve some aspects of higher dimensional data like how they tend to group/cluster 

# Section 2.) SNE (Stochastic Neighbor Embedding)
Stochastic Neighbor Embedding converts Euclidean distance into conditional probabilities that correlate to similarities. We want to encode a function that makes big distances between two points in $X$, $x_{i},x_{j} \in X$, get mapped to large values, and hence a large similarity, if they are close, and get mapped to small values if they tend to be very far from one another. 

Say we want to compute the similarity between $j$ and $i$ in that order. Then we would use the following formula:
$$\mathbb{P}[i|j]=\frac{\exp(-\|x_{i}-x_{j}\|^2/2\sigma_{i}^2)}{\sum_{k \neq i} \exp(-\|x_{i}-x_{k}\|^2/2\sigma_{i}^2)}$$
$\sigma_{i}$ is the variance of the Gaussian centered at $i$. We only care about pairwise similarities, so if an entry is pairwise identical for some reason we set $\mathbb{P}[i|i]=0$. 

I think for this part, it's saying, okay we have $x_{i}$, and we make some kind of Gaussian distribution in a higher dimension around it, and are trying to look for neighbors around it. So we just assume points are distributed from $x_{i}$ in a Gaussian manner, and $x_{j}$ and $x_{i}$ have some distance, and the distance should correspond to being some number of standard deviations from the mean of our function that should be centered at $x_{i}$. 

It seems like our incentive to do this instead of a normal distance function is because as the dimension of the datapoints in our dataset increases, distance becomes increasingly meaningless, so we need another way to measure it. 

We can also compute pairwise similarity for the low dimensional counterparts of $x_{i},x_{j}$ which are given by $y_{i},y_{j}$, this is given by $q_{j|i}$. For some reason here we let the variance $\sigma$
$$q_{j|i}=\frac{\exp(-\|y_{i}-y_{j} \|^2)}{\sum_{k \neq 0} \exp(-\|y_{i}-y_{j} \|^2)}$$
Like for $p$, we set $q_{i|i}=0$. If we are able to project things from $X$ down to $Y$, then we should have the conditional probabilities preserved: 
$$q_{j|i}=p_{j|i}$$
This is the key idea behind Stochastic Neighbor Embedding, we want to preserve this quantity across two dimensions. 

We can turn this into an optimization problem then. How do we find $Y$ such that each $y_{i},y_{j} \in Y$ minimizes the mismatch between pairwise similarity between it and $x_{i},x_{j} \in X$:

One way we can go about optimizing this quantity in how SNE goes about it, by using the "Kullback-Leibler divergence" (KLD). We can use Gradient Descent to minimize the KLD over all datapoints in $X$:
$$C=\sum_{i} KL(P_{i}||Q_{i})=\sum_{i}\sum_{j}p_{j|i}\log\left( \frac{p_{j|i}}{q_{j|i}} \right)$$
This is the cost function, $C$, we are trying to minimize over. The main idea is that close points incur barely any cost, but points that are very far apart have a high cost. 

We need to compute each $\sigma_{i}$ as well. When we find a $\sigma_{i}$, we impose a PDF with a particular variance, $P_{i}$. Data has weird spread generally, so we cannot find a $\sigma$ that optimizes it over all points due to this variance. 

One of the parameters for SNE that is user defined is the entropy, and we perform a binary search looking for an appropriate $\sigma_{i}$ that yields the corresponding perplexity. We produce a distribution that has a fixed perplexity, $Perp(P_{i})=2^{H(P_{i})}$, where $H(P_{i})=-\sum_{j}p_{j|i} \log_{2}(p_{i|j})$ is the entropy equation. 

From Wiki:
"The larger the perplexity, the less likely it is that an observer can guess the value which will be drawn from the distribution."

Ok, now we know $\sigma_{i}$ and $P_{i}$, so we have enough info to concretely define it, so we can do a gradient descent over it:
$$\frac{\partial C}{\partial y_{i}}=2\sum_{j}(p_{j|i}-q_{j|i}+p_{i|j}-q_{i|j})(y_{i}-y_{j})$$
The gradient is n
# Section 3.) SNE vs t-SNE 
# Section 5.) How t-SNE can visualize dataset of 10,000+ datapoints 

# Drawbacks of TSNE
For sets that are Linear, TSNE is considerably slower compared to PCA which is linear over the number of columns: $O(nd^2+d^3)$. 