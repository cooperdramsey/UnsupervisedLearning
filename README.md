# UnsupervisedLearning
Code for a supervised learning project at Georgia Tech University.

Link to my code on GitHub: https://github.com/cooperdramsey/UnsupervisedLearning

I used an anaconda interpreter set to python 3.6. All of the packages loaded are specified in the requirements.txt file.
You can create the exact anaconda interpreter I used by loading the packages found in the requirements file.
The core pacakges I installed where matplotlib v3.0.2, numpy v1.15.4, pandas v0.24.0 and scikit-learn v0.20.2.
All of the other packages were automatically installed with those core libraries.

All of the source code is in the main.py file. Each section of the project is split up in the file and can be run by setting the boolean variables
at the start of the file to true or false. The variables are: **run_clustering**, **run_dim_reduction**, **run_clustering_on_dim_data**,
**run_NN_on_dim_data**, and **run_NN_on_clusters**.

You can select which dataset you want to run the code on by setting the **data_set** variable to either 'wine' or 'loan'. Note that the wine dataset is 1,600 records while the loan data set is 30,000 so the run times are very different.

For some of the graphs, the code is commented out to avoid flooding the screen with charts. To see them simply uncomment the code involving the plots (anything using plt).

All of the graphs for each type of dimensionality reduction and etc can be found in the Images folder.

The datasets are included in the repository and are in the root of the project so the code can read directly from them using a relative path. You don't need to change anything in the csv files for the code to work.

Data Sets:

Wine Quality P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009. Data available from: https://archive.ics.uci.edu/ml/datasets/wine+quality

UCI Credit Card Data Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480. Data Available from: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
