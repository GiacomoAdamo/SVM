# SVM

In this project are implemented different optimization methods for training Support Vector
Machines (supervised learning) in order to solve a classification problem that distinguishes between images of handwritten digits from the MNIST database.

## Description

For the SVMs, three different optimization techniques are tested:
 - Full Minimization
 - Decomposition method
 - Most Violating Pair Decomposition

For each method a detailed explanation is provided in the sections below.

In this project is also implemented a multiclass SVM using the "One Against One” procedure (OAO).

### Target problem

Shallow Feedforward Neural Network (FNN) (one only hidden layer) is built, both
 a MLP and a RBF network, that provides the model f(x) which approximates the true function
 F. We denote by π the hyper-parameters of the network to be settled by means of an heuristic
 procedure and ω the parameters to be settled by minimizing the regularized training error:
 
$ E(ω;π) = 1/2P
 P
 ∑
 p=1
 f (xp)−yp 2+ ρ
 2
 $
 where the hyper parameter ρ stays in the range [10−5÷10−3].
 ∥ω∥2

 For the MPL case the network structure consists in the following formula:

 aaa 


 For the RBF network instead the network structure is:

 bbb

 
### Full Minimization 

The target problem is a highly non-linear and non-convex, therefore the goal of the optimization procedure is not to find the global minimizer of the regularized training error, but to find one of the local minimizers. In this project a batch method has been choosen rather than an online method.

In addition, to speed up the performance of the algorithm, we implemented the forward propagation and the back propagation in a vectorized form, using the python broadcasting and the native numpy function for handling matrices operations, without loops. 

The python routine used to solve the optimization problem is scipy.optimize.minimize employing as unconstraint first order derivative methods the BFGS one.

To enhance the performances of the method, it employs the callable function of the gradient so as not to have it estimated by the algorithm. The accuracy of our computation of the gradient is checked using the function check grad leading to a difference between the gradient computed by us and the one evaluated with finite differences of the order of magnitude of $10^{−6}$.

### Two blocks method

Once 𝑊 (the first layer parameter) is fixed, the problem of minimizing the regularized training error is a quadratic convex problem with a unique global
minimizer. The problem can be reformulated in the form of 1
2𝑃 ‖𝐴𝑉 − 𝐵‖2
, where 𝑉 is the 𝑁 × 1 vector of the output weights,
while 𝐴 = ((𝑔(𝑊𝑋 + 𝑏)𝑇
√𝜌𝑃𝐼𝑁
) 𝑎𝑛𝑑 𝐵 = ( 𝑌
0𝑁
).
To solve this problem, we first constructed the matrices 𝐴 and 𝐵. Then we used the solver lsq_linear from scipy.optimize. This
method requires as mandatory parameters 𝐴, 𝐵. Concerning the other parameters, we used the default values. This solver finds
the global minimizer of the linear least square problem with 0 iterations.

#### Random initialization of W
Since we had noticed that performing the optimization process just one time starting from a random initialization of 𝑊
guaranteed poor performances in term of training error, we decided to iterate the optimization procedure 100000 times starting
from different initial points, obtained by varying the seed for each iteration. At the very end we decided to
choose, as initial point W, the one that guarantees the lowest regularized training error.

#### RBF case

The
optimization problem is now the following:
𝑚𝑖𝑛
𝑣
1
2𝑃 ‖( 𝛷
√𝜌1𝑃𝐼𝑁
) 𝑣 − ( 𝑌
0𝑁
)‖
2
So, we implemented two functions that allow us to concatenate first the matrix 𝛷 with √𝜌1𝑃𝐼𝑁, then the column vector 𝑌 with a
vector of 𝑁 zeros. As in question 2.1 the optimization problem is a linear least square problem (LLSQ). So, the optimization routine
used is the function lsq_linear imported from scipy.optimize. Before the code completion, we checked possible errors in the
gradient evaluation using the function scipy.optimize.check_grad and obtaining an error of about 10-6.

For the unsupervised selection of the centers, we tried two different methods:
1) Random selection of patterns in the training set
2) Custering-based approach to select the 𝑁 most representative vectors (even if the centroids hardly coincide with
patterns in the dataset).
We compared different runs of the first method with the results of the clustering-based approach. This last procedure was the
best and it was carried out by calling the Kmeans function from scikit-learn.clustering, which consists of a Kmeans ++ to which we
set the number of initializations to 20 (equal to 10 by default). Briefly Kmeans ++, unlike traditional K-means, selects not all the
centroids together but it starts from a single random point.

### Decomposition Method

The target problem involves the alternation of a linear least square problem and a highly non-linear and non-convex problem. In
every outer iteration (what we mean by outer iteration is the couple of iterations with respect to both blocks), we solved the
convex block in an exact way using the lsq_linear solver; concerning the non-convex block, we decided to use the BFGS solver

## Getting Started

### Dependencies and Executing program

 - Download the repo with all the subdirectiories of a subset of interest
 - Check if Python 3.10 is installed on your machine 
 - Run the following command to install al the dependencies required for this project
```
pip install -r requirements.txt
```
- To run a specific network of optimization technique, launch the command
```
python main.py
```

## Authors

Author name and contact info

- [Giacomo Mattia Adamo](www.linkedin.com/in/giacomo-mattia-adamo-b36a831ba)
- Simone Foà

## License

This project is licensed - see the LICENSE.md file for details
