from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA, FastICA, NMF, LatentDirichletAllocation, FactorAnalysis
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.model_selection import train_test_split, ShuffleSplit, learning_curve
from sklearn.metrics.cluster import completeness_score, normalized_mutual_info_score, homogeneity_score, v_measure_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize, OneHotEncoder
from scipy.stats import kurtosis
import scipy.sparse as sps
from scipy.linalg import pinv


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
        score = trained_algorithm.bic(X)
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
    fig = plt.figure()
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.title('Neural Network Loss vs Epochs')
    plt.plot(clf.loss_curve_)

    # Print Results
    print("Neural Network:")
    print("Train Time: {:10.6f}s".format(train_time))
    print("Accuracy: {:3.4f}%".format(acc))

    return plt


def reconstruction_error(X_train=None, X_projected=None, X=None, algo=None):
    if X_projected is None:
            W = algo.components_
            if sps.issparse(W):
                W = W.todense()
            p = pinv(W)
            reconstructed = ((p @ W) @ (X.T)).T  # Unproject projected data
            errors = np.square(X - reconstructed)
            return np.nanmean(errors)

    return ((X_train - X_projected) ** 2).mean()


def run_k_means(X, y, data_set, n_comp=None, best=False):
    if best:
        best_algo = KMeans(n_clusters=n_comp, init='random', n_init=10, n_jobs=-1)  # 4 clusters

        kmeans_labels = best_algo.fit_predict(X)

        # print("Homogeneity: {0:.3f}".format(homogeneity_score(y, kmeans_labels)))
        # print("Completeness: {0:.3f}".format(completeness_score(y, kmeans_labels)))
        # print("V-measure: {0:.3f}".format(v_measure_score(y, kmeans_labels)))
        return kmeans_labels

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

    # Chosen with elbow method
    if data_set == 'wine':
        best_algo = fitted_algos[2]  # 4 clusters
    else:
        best_algo = fitted_algos[3]  # 5 clusters

    kmeans_labels = best_algo.predict(X)

    print("Homogeneity: {0:.3f}".format(homogeneity_score(y, kmeans_labels)))
    print("Completeness: {0:.3f}".format(completeness_score(y, kmeans_labels)))
    print("V-measure: {0:.3f}".format(v_measure_score(y, kmeans_labels)))

    # Plot scores
    # plt.title('K vs Inertia for K-Means')
    # plt.xlabel('k value')
    # plt.ylabel('Inertia')
    # plt.xticks(range(min(k), max(k) + 1, 1))
    # plt.plot(k, score)
    # plt.show()
    #
    # # Plot Train times
    # plt.title('K vs Train time for K-means')
    # plt.xlabel('k value')
    # plt.ylabel('Train time (s)')
    # plt.xticks(range(min(k), max(k) + 1, 1))
    # plt.plot(k, train_times)
    # plt.show()

    return kmeans_labels


def run_EM(X, y, data_set, n_comp=None, best=False):
    if best:
        best_algo = GaussianMixture(n_components=n_comp, n_init=10, init_params='random', random_state=10)

        em_labels = best_algo.fit_predict(X)
        #
        # print("Homogeneity: {0:.3f}".format(homogeneity_score(y, kmeans_labels)))
        # print("Completeness: {0:.3f}".format(completeness_score(y, kmeans_labels)))
        # print("V-measure: {0:.3f}".format(v_measure_score(y, kmeans_labels)))
        return em_labels

    num_runs = 10
    init_params = 'random'  # or k-means
    k = range(2, 21)
    algo_list = [GaussianMixture(n_components=num_components, n_init=num_runs, init_params=init_params, random_state=10)
                 for num_components in k]
    score = []
    fitted_algos = []
    train_times = []
    for algo in algo_list:
        trained_algo, algo_score, algo_time = time_score_fit(algo, X)
        fitted_algos.append(trained_algo)
        score.append(algo_score)
        train_times.append(algo_time)

    # Choose algo with best score
    best_score = min(score)
    index = 0
    best_algo = None
    for i in score:
        if i == best_score:
            best_algo = fitted_algos[index]
        index += 1

    em_labels = best_algo.predict(X)
    print("Homogeneity: {0:.3f}".format(homogeneity_score(y, em_labels)))
    print("Completeness: {0:.3f}".format(completeness_score(y, em_labels)))
    print("V-measure: {0:.3f}".format(v_measure_score(y, em_labels)))

    # Plot scores
    # plt.title('K vs BIC for EM')
    # plt.xlabel('k value')
    # plt.ylabel('Score')
    # plt.xticks(range(min(k), max(k) + 1, 2))
    # plt.plot(k, score)
    # plt.show()
    #
    # # Plot Train times
    # plt.title('K vs Train time for EM')
    # plt.xlabel('k value')
    # plt.ylabel('Train time (s)')
    # plt.xticks(range(min(k), max(k) + 1, 2))
    # plt.plot(k, train_times)
    # plt.show()
    return em_labels


# Choose which parts to run
run_clustering = False
run_dim_reduction = False
run_clustering_on_dim_data = False
run_NN_on_dim_data = False
run_NN_on_clusters = True

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

    if run_clustering:
        # K means
        # wine has 8 labels, dataset has 2 labels
        # Don't use these labels to choose K this is cheating, BUT use these labels to validate my choice
        # cluster validation: If k matches the number of classes I have in the data, I can measure the error.

        if data_set=='wine':
            kmeans_labels_orig = run_k_means(X, y, data_set, 4, True)
        else:
            kmeans_labels_orig = run_k_means(X, y, data_set, 5, True)

        # EM
        # Elbow method again to choose num components

        if data_set == 'wine':
            em_labels_orig = run_EM(X, y, data_set, 9, True)
        else:
            em_labels_orig = run_EM(X, y, data_set, 18, True)

        # Compare clusters
        print("Mutual Info Score ORIG: {0:.3f}".format(normalized_mutual_info_score(kmeans_labels_orig, em_labels_orig)))
        print()

    if run_dim_reduction:
        # Dim reduction Show projections. Visual just the first few components.
        # Are these projections meaningful? Why or why not?

        # PCA (pca_data)  #plot the eigen values to know what to drop
        # num components remains same as you increase number
        pca = PCA(random_state=10)
        pca.fit(X)
        pca_X = pca.transform(X)
        pca_reconst = pca.inverse_transform(pca_X)
        e_values = pca.explained_variance_
        num_components = range(1, pca.n_components_ + 1)
        # num_components = range(1, 11)

        plt.title('PCA Components vs Eigenvalues')
        plt.xlabel('Num Components')
        plt.ylabel('E-Values')
        plt.scatter(num_components, e_values)
        plt.show()

        # print('PCA Reconstruction Error: {0:.6f}'.format(reconstruction_error(pca_X, pca_reconst)))
        print('PCA Reconstruction Error: {0:.6f}'.format(reconstruction_error(None, None, X, pca)))

        # ICA (ica_data) # plot kurtosis values
        # the number of components has an impact on the outputs. How good is each k for ICA?
        # Then on the best K, pick the highest
        # kurtosis components
        # run a classifier on the ica components to find best K. Choose highest accuracy

        ica = FastICA(random_state=10)
        ica.fit(X)
        ica_X = ica.transform(X)
        ica_reconst = ica.inverse_transform(ica_X)
        kurtosis_vals = kurtosis(ica_X)
        kurtosis_vals = -np.sort(-kurtosis_vals)
        num_components = range(1, ica_X.shape[1] + 1)
        # num_components = range(1, 13)

        plt.title('ICA Components vs Kurtosis')
        plt.xlabel('Num Components')
        plt.ylabel('Kurtosis')
        plt.scatter(num_components, kurtosis_vals)
        plt.show()

        #print('ICA Reconstruction Error: {0:.6f}'.format(reconstruction_error(ica_X, ica_reconst)))
        print('ICA Reconstruction Error: {0:.6f}'.format(reconstruction_error(None, None, X, ica)))


        # Random Projections (rp_data) reconstruction error
        # errors = []
        # for _ in range(1, 101):
        #     error = []
        #     for i in range(1, X.shape[1] + 1):
        #         rp = SparseRandomProjection(n_components=i)
        #         rp.fit(X)
        #         error.append(reconstruction_error(None, None, X, rp))
        #     errors.append(error)
        #
        # plt.title('RP Components vs AVG Reconstruction Error')  # Over 100 runs
        # plt.xlabel('Num Components')
        # plt.ylabel('AVG Error')
        # plt.plot(range(1, X.shape[1] + 1), np.average(errors, axis=0))
        # plt.show()

        for _ in range(1, 101):
            error = []
            for i in range(1, X.shape[1] + 1):
                rp = SparseRandomProjection(n_components=i)
                rp.fit(X)
                error.append(reconstruction_error(None, None, X, rp))

            plt.title('RP Components vs Reconstruction Error')  # Over 100 runs
            plt.xlabel('Num Components')
            plt.ylabel('Error')
            plt.plot(range(1, X.shape[1] + 1), error)
        plt.show()

        # Factor Analysis
        fa = FactorAnalysis(random_state=10, svd_method='randomized')
        fa.fit(X)
        fa_X = fa.transform(X)
        num_components = range(1, fa_X.shape[1] + 1)
        #num_components = range(1, 15)
        variance = fa.noise_variance_
        variance = -np.sort(-variance)

        print('FA Reconstruction Error: {0:.6f}'.format(reconstruction_error(None, None, X, fa)))

        plt.title('FA Components vs Noise Variance')
        plt.xlabel('Num Components')
        plt.ylabel('Noise Variance')
        plt.scatter(num_components, variance)
        plt.show()

    if run_clustering_on_dim_data:
        # Clustering on Reduced data (This will be eight combinations 4 dim reduction, 2 clusters)
        # what clusters did I get? Compare with the same clusters from part one
        # PCA
        if data_set == 'wine':
            n_comp = 5
            k_comp = 4
            em_comp = 5
        else:
            n_comp = 10
            k_comp = 5
            em_comp = 10
        pca = PCA(random_state=10, n_components=n_comp)
        pca_X = pca.fit_transform(X)

        #Cluster
        print("PCA:")
        kmeans_labels_orig = run_k_means(X, y, data_set, k_comp, True)
        kmeans_labels = run_k_means(pca_X, y, data_set, k_comp, True)
        em_labels_orig = run_EM(X, y, data_set, em_comp, True)
        em_labels = run_EM(pca_X, y, data_set, em_comp, True)
        # Compare clusters
        print("PCA K means Mutual Info Score: {0:.3f}".format(normalized_mutual_info_score(kmeans_labels, kmeans_labels_orig)))
        print("PCA EM Mutual Info Score: {0:.3f}".format(normalized_mutual_info_score(em_labels, em_labels_orig)))
        print()

        #ICA
        if data_set == 'wine':
            n_comp = 6
            k_comp = 4
            em_comp = 6
        else:
            n_comp = 12
            k_comp = 5
            em_comp = 12
        ica = FastICA(random_state=10, n_components=n_comp)
        ica_X = ica.fit_transform(X)

        #Cluster
        print("ICA:")
        kmeans_labels_orig = run_k_means(X, y, data_set, k_comp, True)
        kmeans_labels = run_k_means(ica_X, y, data_set, k_comp, True)
        em_labels_orig = run_EM(X, y, data_set, em_comp, True)
        em_labels = run_EM(ica_X, y, data_set, em_comp, True)
        # Compare clusters
        print("ICA K means Mutual Info Score: {0:.3f}".format(normalized_mutual_info_score(kmeans_labels, kmeans_labels_orig)))
        print("ICA EM Mutual Info Score: {0:.3f}".format(normalized_mutual_info_score(em_labels, em_labels_orig)))
        print()

        # RP
        if data_set == 'wine':
            n_comp = 7
            k_comp = 4
            em_comp = 7
        else:
            n_comp = 14
            k_comp = 5
            em_comp = 14
        rp = SparseRandomProjection(n_components=n_comp)
        rp_X = rp.fit_transform(X)

        #Cluster
        print("RP:")
        kmeans_labels_orig = run_k_means(X, y, data_set, k_comp, True)
        kmeans_labels = run_k_means(rp_X, y, data_set, k_comp, True)
        em_labels_orig = run_EM(X, y, data_set, em_comp, True)
        em_labels = run_EM(rp_X, y, data_set, em_comp, True)
        # Compare clusters
        print("RP K means Mutual Info Score: {0:.3f}".format(normalized_mutual_info_score(kmeans_labels, kmeans_labels_orig)))
        print("RP EM Mutual Info Score: {0:.3f}".format(normalized_mutual_info_score(em_labels, em_labels_orig)))
        print()

        # FA
        if data_set == 'wine':
            n_comp = 6
            k_comp = 4
            em_comp = 6
        else:
            n_comp = 14
            k_comp = 5
            em_comp = 14
        fa = FactorAnalysis(n_components=n_comp)
        fa_X = fa.fit_transform(X)

        #Cluster
        print("FA:")
        kmeans_labels_orig = run_k_means(X, y, data_set, k_comp, True)
        kmeans_labels = run_k_means(fa_X, y, data_set, k_comp, True)
        em_labels_orig = run_EM(X, y, data_set, em_comp, True)
        em_labels = run_EM(fa_X, y, data_set, em_comp, True)
        # Compare clusters
        print("FA K means Mutual Info Score: {0:.3f}".format(normalized_mutual_info_score(kmeans_labels, kmeans_labels_orig)))
        print("FA EM Mutual Info Score: {0:.3f}".format(normalized_mutual_info_score(em_labels, em_labels_orig)))

    # Make sure to hold out a test set for this part
    # Train Model
    #X_train, X_test, y_train, y_test = split_data(X, y, 0.2)
    # only do the dim reduction and clusters on the training set
    # will need to re-cluster these thing
    if run_NN_on_dim_data:
        # PCA
        print("PCA:")
        if data_set == 'wine':
            n_comp = 5
        else:
            n_comp = 10
        pca = PCA(random_state=10, n_components=n_comp)
        pca_X = pca.fit_transform(X)
        X_train, X_test, y_train, y_test = split_data(pca_X, y, 0.2)
        run_neural_network(X_train, X_test, y_train, y_test)

        # ICA
        print("ICA:")
        if data_set == 'wine':
            n_comp = 6
        else:
            n_comp = 12
        ica = FastICA(random_state=10, n_components=n_comp)
        ica_X = ica.fit_transform(X)
        X_train, X_test, y_train, y_test = split_data(ica_X, y, 0.2)
        run_neural_network(X_train, X_test, y_train, y_test)

        # RP
        print("RP:")
        if data_set == 'wine':
            n_comp = 7
        else:
            n_comp = 14
        rp = SparseRandomProjection(n_components=n_comp)
        rp_X = rp.fit_transform(X)
        X_train, X_test, y_train, y_test = split_data(rp_X, y, 0.2)
        run_neural_network(X_train, X_test, y_train, y_test)

        # FA
        print("FA:")
        if data_set == 'wine':
            n_comp = 6
        else:
            n_comp = 14
        fa = FactorAnalysis(n_components=n_comp)
        fa_X = fa.fit_transform(X)
        X_train, X_test, y_train, y_test = split_data(fa_X, y, 0.2)
        run_neural_network(X_train, X_test, y_train, y_test)

        plt.show()

    if run_NN_on_clusters:
        # Clusters as new features Neural Network. Compare performance and time with assignment 1 AND compare
        # with the dim reduced neural nets

        # Kmeans
        if data_set=='wine':
            kmeans_labels = run_k_means(X, y, data_set, 4, True)
        else:
            kmeans_labels = run_k_means(X, y, data_set, 5, True)

        kmeans_labels = kmeans_labels.reshape((-1, 1))

        # encode labels into feature vectors
        # K means
        print("K Means")
        enc = OneHotEncoder()
        kmeans_X = enc.fit_transform(kmeans_labels)
        X_train, X_test, y_train, y_test = split_data(kmeans_X, y, 0.2)
        run_neural_network(X_train, X_test, y_train, y_test)

        # EM
        # Elbow method again to choose num components

        if data_set == 'wine':
            em_labels = run_EM(X, y, data_set, 9, True)
        else:
            em_labels = run_EM(X, y, data_set, 18, True)

        em_labels = em_labels.reshape((-1, 1))

        # EM
        print("EM")
        enc = OneHotEncoder()
        em_X = enc.fit_transform(em_labels)
        X_train, X_test, y_train, y_test = split_data(em_X, y, 0.2)
        run_neural_network(X_train, X_test, y_train, y_test)

        plt.show()

