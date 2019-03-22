from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA, FastICA, SparsePCA, LatentDirichletAllocation
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit, learning_curve
from sklearn.metrics.cluster import completeness_score, contingency_matrix, mutual_info_score, silhouette_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize


def load_data(data_path):
    data = pd.read_csv(data_path)
    y = data.iloc[:, -1]
    X = data.iloc[:, :-1]
    return X, y


def split_data(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test


def time_score_fit(algorithm, X):
    start = timer()
    trained_algorithm = algorithm.fit(X)
    end = timer()
    labels = trained_algorithm.predict(X)
    try:
        score = trained_algorithm.inertia_
    except:
        score = trained_algorithm.score(X)
    train_time = end - start

    return trained_algorithm, score, train_time


# function from Scikit learn url:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def run_neural_network(X_train, X_test, y_train, y_test):
    title = "Learning Curves Neural Network"
    if data_set is 'wine':
        clf = MLPClassifier(max_iter=2000, solver='adam', alpha=0.0001, activation='tanh', learning_rate_init=0.001)
    else:
        clf = MLPClassifier(max_iter=2000, solver='adam', alpha=0.0001, learning_rate_init=0.001,
                            activation='logistic')

    start = timer()
    clf.fit(X_train, y_train)
    end = timer()
    train_time = end - start
    estimator = clf

    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    # Plot Learning Curve
    plot_learning_curve(estimator, title, X, y, cv=cv, n_jobs=-1)

    # Final test on held out training data
    y_pred = estimator.predict(X_test)
    acc = accuracy_score(y_pred, y_test)

    # Check Training Time If used cross validation

    # For nueral network, graph training iterations to accuracy
    plt.figure()
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.title('Neural Network Loss vs Epochs')
    plt.plot(clf.loss_curve_)
    plt.show()

    # Print Results
    print("Neural Network:")
    print("Train Time: {:10.6f}s".format(train_time))
    print("Accuracy: {:3.4f}%".format(acc))


# select data set
data_set = 'wine'  # data set can be either wine or loan

if __name__ == '__main__':
    # Load data
    if data_set is 'wine':
        data_path = 'winequality-red.csv'
    else:
        data_path = 'UCI_Credit_Card.csv'

    X, y = load_data(data_path)

    # Data normalization
    X = normalize(X)

    # Cluster algorithms (K-means, EM)

    # to plot the clusters: Use cluster feature connection to plot
    # show for each feature, the feature contribution

    #K means
    # wine has 8 labels, dataset has 2 labels
    # Don't use these labels to choose K this is cheating, BUT use these labels to validate my choice
    # cluster validation: If k matches the number of classes I have in the data, I can measure the error.

    # Use elbow method to choose K
    num_runs = 10
    k = range(2, 21)
    algo_list = [KMeans(n_clusters=num_clusters, init='random', n_init=num_runs, n_jobs=-1) for num_clusters in k]
    score = []
    fitted_algos = []
    train_times = []
    for algo in algo_list:
        trained_algo, algo_score, algo_time = time_score_fit(algo, X)
        fitted_algos.append(trained_algo)
        score.append(algo_score)
        train_times.append(algo_time)

    # Choose algo with best score
    best_score = max(score)
    index = 0
    best_algo = None
    for i in score:
        if i == best_score:
            best_algo = fitted_algos[index]
        index += 1

    # Plot scores
    plt.title('K vs Score for K-Means')
    plt.xlabel('k value')
    plt.ylabel('Inertia')
    plt.xticks(range(min(k), max(k) + 1, 1))
    plt.plot(k, score)
    plt.show()

    # Plot Train times
    plt.title('K vs Train time for K-means')
    plt.xlabel('k value')
    plt.ylabel('Train time (s)')
    plt.xticks(range(min(k), max(k) + 1, 1))
    plt.plot(k, train_times)
    plt.show()

    # EM
    # Elbow method again to choose num components
    num_runs = 10
    init_params = 'random' # or k-means
    k = range(2, 21)
    algo_list = [GaussianMixture(n_components=num_components, n_init=num_runs, init_params=init_params) for num_components in k]
    score = []
    fitted_algos = []
    train_times = []
    for algo in algo_list:
        trained_algo, algo_score, algo_time = time_score_fit(algo, X)
        fitted_algos.append(trained_algo)
        score.append(algo_score)
        train_times.append(algo_time)

    # Choose algo with best score
    best_score = max(score)
    index = 0
    best_algo = None
    for i in score:
        if i == best_score:
            best_algo = fitted_algos[index]
        index += 1

    # Plot scores
    plt.title('K vs Score for EM')
    plt.xlabel('k value')
    plt.ylabel('Score')
    plt.xticks(range(min(k), max(k) + 1, 1))
    plt.plot(k, score)
    plt.show()

    # Plot Train times
    plt.title('K vs Train time for EM')
    plt.xlabel('k value')
    plt.ylabel('Train time (s)')
    plt.xticks(range(min(k), max(k) + 1, 1))
    plt.plot(k, train_times)
    plt.show()

    # Dim reduction Show projections. Visual just the first few components.
    # Are these projections meaningful? Why or why not?
    # PCA (pca_data)  #plot the eigen values to know what to drop
    # num components remains same as you increase number

    # ICA (ica_data) # plot kurtosis values
    # the number of components has an impact on the outputs. How good is each k for ICA? Then on the best K, pick the highest
    # kurtosis components
    # run a classifier on the ica components to find best K. Choose highest accuracy

    # Random Projections (rp_data) reconstruction error

    # LatentDirichletAllocation

    # Clustering on Reduced data (This will be eight combinations 4 dim reduction, 2 clusters)
    # what clusters did I get? Compare with the same clusters from part one

    # Make sure to hold out a test set for this part
    # Train Model
    X_train, X_test, y_train, y_test = split_data(X, y, 0.2)
    # only do the dim reduction and clusters on the training set
    # will need to re-cluster these things

    # Run Neural network on reduced data. Compare performance and time with assignment 1
    run_neural_network(pca_X, pca_y)
    run_neural_network(ica_X, ica_y)
    run_neural_network(rp_X, rp_y)
    run_neural_network(lda_X, lda_y)

    # Clusters as new features Neural Network. Compare performance and time with assignment 1 AND compare
    # with the dim reduced neural nets
    # Use the clustering output as the new features
    # K means data
    kmeans_X, kmeans_y = None, None

    # EM data
    em_X, em_y = None, None

    run_neural_network(kmeans_X, kmeans_y)
    run_neural_network(em_X, em_y)
