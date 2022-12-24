# Machine-learning-Algorithms
## Linear Regression
* The output of regression problem is continuos values. It is used for predictive analysis. It tries to learn relationship between dependent and independent variable(x&y)
* In simple linear regression, It tries to find the line which best fits the training data.
* Y = W0 + W1*X {X = x1,x2,...xm} (simple linear regression)
* Residual error: y_pred - y_i = e. Total error is the sum of residual errors across all the data.
* Assumptions: 
    * Linear relationship: There exists a linear relationship between the independent variable, x, and the dependent variable, y.
    * Independence: The residuals are independent. In particular, there is no correlation between consecutive residuals in time series data.
    * Homoscedasticity: The residuals have constant variance at every level of x.
    * Normality: The residuals of the model are normally distributed.
    
If one or more of these assumptions are violated, then the results of our linear regression may be unreliable or even misleading.
More details here: https://www.statology.org/linear-regression-assumptions/
* Gradient Descent: optimization algorithm to optimize the cost functions to reach the optimal minimal solutions. This is done by updating the values B0 and B1 iteratively until we get the optimal solution.
* `Learning rate: In the gradient descent algorithm, the number of steps you’re taking can be considered as the learning rate, and this decides how fast the algorithm converges to the minima.

## Logistic Regression: 
* Logistic regression is used to describe data and to explain the relationship between one dependent binary variable and one or more nominal, ordinal, interval or ratio-level independent variables.
* It is used when our dependent variable is dichotomous or binary. It just means a variable that has only 2 outputs, for example, A person will survive this accident or not, The student will pass this exam or not. The outcome can either be yes or no (2 outputs). This regression technique is similar to linear regression and can be used to predict the Probabilities for classification problems.
* More details here: 
    * https://www.analyticsvidhya.com/blog/2021/08/conceptual-understanding-of-logistic-regression-for-data-science-beginners/
    * https://towardsdatascience.com/logit-of-logistic-regression-understanding-the-fundamentals-f384152a33d1
## Naive Bayes
* Naive Bayes is a simple technique for constructing classifiers: models that assign class labels to problem instances, represented as vectors of feature values, where the class labels are drawn from some finite set. There is not a single algorithm for training such classifiers, but a family of algorithms based on a common principle: all naive Bayes classifiers assume that the value of a particular feature is independent of the value of any other feature, given the class variable. For example, a fruit may be considered to be an apple if it is red, round, and about 10 cm in diameter. A naive Bayes classifier considers each of these features to contribute independently to the probability that this fruit is an apple, regardless of any possible correlations between the color, roundness, and diameter features.
* Secondly, each feature is given the same weight(or importance). For example, knowing only temperature and humidity alone can’t predict the outcome accurately. None of the attributes is irrelevant and assumed to be contributing equally to the outcome.
* https://towardsdatascience.com/naive-bayes-classifier-81d512f50a7c

## SVM
* A support vector machine (SVM) is a supervised machine learning model that uses classification algorithms for two-group classification problems.
* It tries to find maximum margin hyperplan that separates two classes in best possible way.
* The C parameter tells the SVM optimization how much you want to avoid misclassifying each training example. For large values of C, the optimization will choose a smaller-margin hyperplane if that hyperplane does a better job of getting all the training points classified correctly. Conversely, a very small value of C will cause the optimizer to look for a larger-margin separating hyperplane, even if that hyperplane misclassifies more points. For very tiny values of C, you should get misclassified examples, often even if your training data is linearly separable.
* Penalized svm: you can set weights for the class in the training. Typically use it to handle unbalanced classes problem.
* A small gamma means a Gaussian with a large variance so the influence of x_j is more, i.e. if x_j is a support vector, a small gamma implies the class of this support vector will have influence on deciding the class of the vector x_i even if the distance between them is large. If gamma is large, then variance is small implying the support vector does not have wide-spread influence. Technically speaking, large gamma leads to high bias and low variance models, and vice-versa.
* More details:
    * https://www.quora.com/Whats-the-difference-between-LibSVM-and-LibLinear
    * https://www.quora.com/What-are-C-and-gamma-with-regards-to-a-support-vector-machine
    * https://www.quora.com/Intuitively-how-do-Lagrange-multipliers-work-in-SVMs/answer/Prasoon-Goyal
    * https://www.quora.com/Can-an-SVM-use-gradient-descent-to-minimize-its-margin-instead-of-using-a-lagrange
    * http://www.svms.org/srm/
    * https://monkeylearn.com/blog/introduction-to-support-vector-machines-svm/

## K-nearest neighbor
* K Nearest Neighbor algorithm falls under the Supervised Learning category and is used for classification (most commonly) and regression. It is a versatile algorithm also used for imputing missing values and resampling datasets. As the name (K Nearest Neighbor) suggests it considers K Nearest Neighbors (Data points) to predict the class or continuous value for the new Datapoint.
* The algorithm’s learning is:
    * Instance-based learning: Here we do not learn weights from training data to predict output (as in model-based algorithms) but use entire training instances to predict output for unseen data.
    *  Lazy Learning: Model is not learned using training data prior and the learning process is postponed to a time when prediction is requested on the new instance.
    *  Non -Parametric: In KNN, there is no predefined form of the mapping function.
* More details:
    * https://www.analyticsvidhya.com/blog/2021/04/simple-understanding-and-implementation-of-knn-algorithm/
## Tree based algorithms
* Tree based algorithms are generally robust to imbalanced datasets.
    ### Decision Tree
    * A decision tree is a model composed of a collection of "questions" organized hierarchically in the shape of a tree. The questions are usually called a condition, a split, or a test. Each non-leaf node contains a condition, and each leaf node contains a prediction.
    * Inference of a decision tree model is computed by routing an example from the root (at the top) to one of the leaf nodes (at the bottom) according to the conditions. The value of the reached leaf is the decision tree's prediction. The set of visited nodes is called the inference path.
    *  A decision tree or a classification tree is a tree in which each internal (non-leaf) node is labeled with an input feature. The arcs coming from a node labeled with an input feature are labeled with each of the possible values of the target feature or the arc leads to a subordinate decision node on a different input feature. Each leaf of the tree is labeled with a class or a probability distribution over the classes, signifying that the data set has been classified by the tree into either a specific class, or into a particular probability distribution (which, if the decision tree is well-constructed, is skewed towards certain subsets of classes).
    * More details: https://www.analyticsvidhya.com/blog/2022/01/decision-tree-machine-learning-algorithm/
    ### Bagging
    * Bagging stands for Bootstrap Aggregation 
    * Takes original data set D with N training examples 
    * Creates m copies D_1.....D_m
        * Each D_i is generated from D by sampling with replacement 
        * Each data set D_i has the same number of examples as in data set D 
        * These data sets are reasonably different from each other (since only about 63% of the original examples appear in any of these data sets) 
    * Train models h_1,...,h_m using  D_11,..., D_m, respectively 
    * Use an averaged model h = sum(h_i)/m (over i=1 to m) as the final model 
    * Useful for models with high variance and noisy data
    ### Boosting
    * The basic idea 
    * Take a weak learning algorithm 
        * Only requirement: Should be slightly better than random 
    * Turn it into an awesome one by making it focus on difficult cases 
    * Most boosting algoithms follow these steps: 
        1. Train a weak model on some training data 
        2. Compute the error of the model on each training example 
        3. Give higher importance to examples on which the model made mistake
        4. Re-train the model using “importance weighted” training examples 
        5. Go back to step 2

    ### Random Forest
    * An ensemble of decision tree (DT) classifiers 
    * Uses bagging on features (each DT will use a random set of features)
    * Given a total of D features, each DT uses √ D randomly chosen feature
    * Randomly chosen features make the different trees uncorrelated 
    * All DTs usually have the same depth 
    * Each DT will split the training data differently at the leaves 
    * Prediction for a test example votes on/averages predictions from all the DTs
    * Random forest use bagging on the features. Each decision tree use different set of features.
    * The random forest algorithm works well when you have both categorical and numerical features. It also worlk well with the missing values.
    * Details: https://www.knowledgehut.com/blog/data-science/bagging-and-random-forest-in-machine-learning
    ### GBDT
    1. Fit a simple linear regression or decision tree on the data.
    2. Calcluate the error residuals.[e_1 = y-y_pred1]
    3. Fit a new model on error residuals as target variable with same input
    variable [call it e1_pred]
    4. Add the predicted residuals to previous predictions.
    [y_pred2 = y_pred1 + e1_pred]
    5. Fit another model on residuals that is still left i.e.[e2 = y-y_pred2].
    Repeat steps 2 to 5 untill it starts overfitting or sum of residuals becomes constant.
    y' = f_0(x)+ delta_1(x) + delta_2(x).....delta_m(x)
       = f_0(x)+ sum(delta_i(x)(m=1toM) = F_m(x)
       F_m(x) = F_m-1(x)+ k*delta_m(x) ( k is learning parameter)
       
    GBM leverage the patterns of residuals around 0 to fit a model. GBM repeatidly leverage the patterns in residual and strengthen a model with weak predictions and make it better. Once we reach a stage that residuals do not have any pattern that could be modelled, we can stop modelling residuals.
    GBDT has a few other advantages, like working well with feature collinearity, handling features with different ranges and missing feature values, etc
    ### XGBOOST
    * Benefits
        * open source & efficient implementation of GBDT.
        * Parallelization of tree construction using all CPU cores
        * Provides distributed computing for training very large models using cluster of machines.
        * Cache awareness and out-of-core computing
        * Regularization for avoiding overfitting
        * Efficient handling of missing data
        * In-built cross-validation capability
        * Automatic tree pruning
        * Handle outliers
        * Need to handle categorical features separately.
    More details: https://www.kdnuggets.com/2018/08/unveiling-mathematics-behind-xgboost.html
    
    ## LightGBM
    * 20 times faster than GBDT and 10 times faster than XGBoost.
    * Leafwise growth of trees
    * It can handle categorical feature automatically(no need to encode)
    * It uses two types of techniques which are gradient Based on side sampling or GOSS and Exclusive Feature bundling or EFB.
    * GOSS will actually exclude the significant portion of the data part which have small gradients and only use the remaining data to estimate the overall information gain. The data instances which have large gradients actually play a greater role for computation on information gain. GOSS can get accurate results with a significant information gain despite using a smaller dataset than other models.
    * With the EFB, It puts the mutually exclusive features along with nothing but it will rarely take any non-zero value at the same time to reduce the number of features. This impacts the overall result for an effective feature elimination without compromising the accuracy of the split point.
    * By combining the two changes, it will fasten up the training time of any algorithm by 20 times. So LGBM can be thought of as gradient boosting trees with the combination for EFB and GOSS. 
    * More details here:
        * https://www.geeksforgeeks.org/lightgbm-light-gradient-boosting-machine/
        * https://www.analyticsvidhya.com/blog/2021/08/complete-guide-on-how-to-use-lightgbm-in-python/
        * https://www.youtube.com/watch?v=9uxWzeLglr0
    ## CatBoost
    * We can use text and categorical features(as it is). In the paper, they reported that it performs well if you just pass categorical features with out any encoding. It Handles Categorical features automatically.
    * It make sure that the tree is symmetric.
    * It uses MVS and target based encoding(to encode categorical features.)
    * It uses "Pool" internal data structure to store traninig and test data.
    * More details
        * https://neptune.ai/blog/when-to-choose-catboost-over-xgboost-or-lightgbm
        * https://www.youtube.com/watch?v=V5158Oug4W8
    ## Extra links
    * http://zhanpengfang.github.io/418home.html
    * https://github.com/rpmcruz/machine-learning/blob/master/ensemble/boosting/gboost.py
    * https://zpz.github.io/blog/gradient-boosting-tree-for-binary-classification/
    * https://towardsdatascience.com/gradient-boosting-is-one-of-the-most-effective-ml-techniques-out-there-af6bfd0df342 
    * https://arxiv.org/pdf/2106.03253.pdf (deep learning vs boosting)
    * https://coderzcolumn.com/tutorials/machine-learning/simple-guide-to-optuna-for-hyperparameters-optimization-tuning (optuna framework)
    * https://cse.iitk.ac.in/users/piyush/courses/ml_autumn16/771A_lec21_slides.pdf (basic best)
    * https://people.cs.pitt.edu/~milos/courses/cs2750-Spring04/lectures/class23.pdf
    * https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d
# Clustering Algorithms
Using clustering, we can group similar items, products, and users together. This grouping, or segmenting, works across industries. Clustering is the process of dividing the entire data into groups (also known as clusters) based on the patterns in the data.
## Properties of clusters
* All the data points in a cluster should be similar to each other.
* The data points from different clusters should be as different as possible. 
## Application of clustering
* Customer Segmentation
* Document Clustering
* Image Segmentation
* Recommendation Engines
## Evaluation metrics
* Inertia or Intra cluster distance:  Inertia actually calculates the sum of distances of all the points within a cluster from the centroid of that cluster. The inertial value should be as low as possible.
* Dunn Index = min(Inter cluster distance)/max(Intra cluster distance)
We want to maximize the Dunn index. The more the value of the Dunn index, the better will be the clusters.
## k-means clustering
### Steps
    1. Choose the number of clusters k
    2. Select k random points from the data as centroids
    3. Assign all the points to the closest cluster centroid
    4. Recompute the centroids of newly formed clusters
    5. Repeat steps 3 and 4
### Stopping criteria for k-means
There can be 3 stopping criteria
1. Centroids of newly formed clusters do not change
2. Points remain in the same cluster
3. Maximum number of iterations are reached

        We can stop the algorithm if the centroids of newly formed clusters are not changing. Even after multiple iterations, if we are getting the same centroids for all the clusters, we can say that the algorithm is not learning any new pattern and it is a sign to stop the training.
        Another clear sign that we should stop the training process if the points remain in the same cluster even after training the algorithm for multiple iterations.
        Finally, we can stop the training if the maximum number of iterations is reached. Suppose if we have set the number of iterations as 100. The process will repeat for 100 iterations before stopping.
More details: https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/
## k-means++
* K-Means++ is a smart centroid initialization technique and and the rest of the algorithm is the same as that of K-Means. This method should give the same results(same output clusters unlike k-means) The steps to follow for centroid initialization are:
    * Pick the first centroid point (C_1) randomly.
    * Compute distance of all points in the dataset from the selected centroid. The distance of x_i point from the farthest centroid can be computed by
        * d_i = max(j=1 to m) || x_i - c_j||^2
        d_i: Distance of x_i point from the farthest centroid
        m: number of centroids already picked
    * Make the point x_i as the new centroid that is having maximum         probability proportional to d_i.
    * Repeat the above two steps till you find k-centroids.
* details: https://www.quora.com/What-is-the-difference-between-k-medoid-and-k-mean-Which-is-the-best-for-clustering

## k-medoid
* k-medoid is more robust to outliers and noise.
* The k-medoids algorithm is a clustering algorithm related to the k-means algorithm and the medoidshift algorithm. Both the k-means and k-medoids algorithms are partitional (breaking the dataset up into groups). K-means attempts to minimize the total squared error, while k-medoids minimizes the sum of dissimilarities between points labeled to be in a cluster and a point designated as the center of that cluster. In contrast to the k-means algorithm, k-medoids chooses datapoints as centers ( medoids or exemplars).
* K-medoids is also a partitioning technique of clustering that clusters the data set of n objects into k clusters with k known a priori. A useful tool for determining k is the silhouette.
* It could be more robust to noise and outliers as compared to k-means because it minimizes a sum of general pairwise dissimilarities instead of a sum of squared Euclidean distances. The possible choice of the dissimilarity function is very rich but in our applet we used the Euclidean distance.
* A medoid of a finite dataset is a data point from this set, whose average dissimilarity to all the data points is minimal i.e. it is the most centrally located point in the set.
* The most common realisation of k-medoid clustering is the Partitioning Around Medoids (PAM) algorithm and is as follows:
    1. Initialize: randomly select k of the n data points as the medoids
    2. Assignment step: Associate each data point to the closest medoid.
    3. Update step: For each medoid m and each data point o associated to m swap m and o and compute the total cost of the configuration (that is, the average dissimilarity of o to all the data points associated to m). Select the medoid o with the lowest cost of the configuration.
Repeat alternating steps 2 and 3 until there is no change in the assignments.

# Comparison of different algorithms
* Decision tree is not much robust to noise. Noisy examples can lead to overfitting.
* If you have any rare occurrences avoid using decision trees.
* Random Forest is intrinsically suited for multiclass problems, while SVM is intrinsically two-class. For multiclass problem you will need to reduce it into multiple binary classification problems.
* Adaboost only works for weak model(all the models must commit some errors) while Gradient boosting can work with strong models.
* Adaboost is the special case for gradient boosting.
* Tree based models are generally robust to the imbalance datasets.(important)
* Difference between random forest and boosting
    * This difference between random forests and gradient boosting have many implications. For example, in random forests you do not expect to get a significant gain from moving from an ensemble of size 50 to an ensemble of size 100 since the reduction in variance is of the order of the square root of the size of the ensemble. However, in gradient boosting you may keep improving as the ensemble size increases. Another difference is the tree sizes: in random forests you need large trees because individual trees have to overfit such that the random skewness will be observed and canceled by the averaging mechanism. However, in gradient boosting you can use small trees since they act as “weak learners”: as long as they can follow vaguely the direction of improvement this is sufficient.
* Comparisons of tree based model compare to DNN:
    * It is quite non-trivial to augment a tree ensemble model with other trainable components, such as embeddings for discrete features. Such practices typically require joint training of the model with the component/feature, while the tree ensemble model assumes that the features themselves need not be trained.
    * Tree models do not work well with sparse id features such as skill ids, company ids, and member ids that we may want to utilize for talent search ranking. Since a sparse feature is non-zero for a relatively small number of examples, it has a small likelihood of being chosen by the tree generation at each boosting step, especially since the learned trees are shallow in general.
    * After saying a lot of things we might risk an answer saying that if we have data that is homogeneous (same type) and relationships between features are important then a deep NN might (should?) outperform a tree-based method.
    * Tree models lack flexibility in model engineering. It might be desirable to use novel loss functions, or augment the current objective function with other terms. Such modifications are not easily achievable with GBDT models, but are relatively straightforward for deep learning models based on differentiable programming. A neural network model with a final (generalized) linear layer also makes it easier to adopt approaches such as transfer learning and online learning
    * For structured data (not images or text) the representation problem is (kind of) already solved so the only thing left is to apply a classification algorithm and then Xgboost is usually better than a neural network.
    * One important thing to notice is that in images or text all the features built by a Deep Learning method are of the same type and category while on an arbitrary dataset we can have a mixture of different feature types, some numeric, some categorical, etc. These days tree-based methods are much better than NNs dealing with this kind of heterogeneous information, in each split you can choose a single feature and it doesn’t matter it’s type.
* K means is not good for outliers. Since it assigns every point to some clusters, it will be problem in anamoly detection kind of problem. Good for spherical shape clusters.
* K means guarantee to converge. It may converge to local minima.
* DBScan clustering: Handles outliers. Separates high density clusters with low density clusters. Able to detect arbitary shapes of clusters unlike k means
* Details
    * https://arxiv.org/pdf/2106.03253.pdf (deep learning vs Boosting)
    * https://www.quora.com/What-is-the-difference-between-k-medoid-and-k-mean-Which-is-the-best-for-clustering
    * https://www.thekerneltrip.com/statistics/random-forest-vs-svm/
    * https://datascience.stackexchange.com/questions/6838/when-to-use-random-forest-over-svm-and-vice-versa