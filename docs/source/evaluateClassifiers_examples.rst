Evaluating classifiers examples
==================================================================

In the PyPPI, we integrate several machine learning classifiers from sklearn and implement several classical deep learning models for users to perform performance tests, for which we provide two easy-to-use functions for machine learning classifiers and deep learning models respectively.

Importing related modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: py

    from PyPPI.evaluateClassifiers import evaluateDLclassifers, evaluateMLclassifers
    from PyPPI.Features import generateStructureFeatures, generateBIOFeatures, generatePhysChemFeatures, generateLanguageModelFeatures
    from PyPPI.filesOperation import getLabel
    from PyPPI.featureSelection import cife

Evaluating various machine learning classifiers using biological features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A total of 11 machine learning classifiers are included in the ``PyPPI``. After the function finishes running, an ``ML_evalution_metrics.csv`` is generated, which contains the performance metrics of each classifier on the dataset.

.. code-block:: py

    label_path = '/home/wangyansong/wangyubo/PyPPI/datasets/label_72.npy'

    # read labels from the given path
    label = read_label(label_path)

    # Generating biological features for example.
    bioFeature = generateBIOFeatures(ECO=True, HSP=True, RAA=True, Pro2Vec_1D=True, PSSM=True, Anchor=True, windowSize=25)

    # Perform feature selection to refine the biological features
    refined_biological_features = cife(bioFeature, label, num_features=10)

    # Perform a 5-fold cross-validation of the machine learning classifier using biological features, and store the result file in the current folder.
    evaluateMLclassifers(refined_biological_features, folds=5, labels=label, file_path='./', shuffle=True)

output:
    ::

        Starting runnning machine learning classifiers using 5-fold cross-validation, please be patient...
        running LogisticRegression...
        finish
        running KNeighborsClassifier...
        finish
        running DecisionTreeClassifier...
        finish
        running GaussianNB...
        finish
        running BaggingClassifier...
        finish
        running RandomForestClassifier...
        finish
        running AdaBoostClassifier...
        finish
        running GradientBoostingClassifier...
        finish
        running SVM...
        finish
        running LinearDiscriminantAnalysis...
        finish
        running ExtraTreesClassifier...
        finish
        All classifiers have finished running, the result file are locate in ./

Evaluating various deep learning models using dynamic semantic information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In ``PyPPI`` we implement six classes of classical deep learning models, including ``CNN``, ``LSTM``, ``GRU``, ``ResNet-1D``, ``MLP`` and ``CapsuleNet``. After the function finishes running, an ``DL_evalution_metrics.csv`` is generated, which contains the performance metrics of each model on the dataset.

We use the same dataset as previous example to evaluate deep learning models.

.. code-block:: py

    # Generating biological features for evaluating models.
    bioFeature = generateBIOFeatures(ECO=True, HSP=True, RAA=True, Pro2Vec_1D=True, PSSM=True, Anchor=True, windowSize=25)

    # read labels from the given path
    label = getLabel(label_npy='datasets/label_72.npy')

    # Perform a 5-fold cross-validation of the deep learning classifier using biological features, and store the result file in the current folder.
    evaluateDLclassifers(dynamic_semantic_information, folds=5, labels=label, file_path='./', shuffle=True)

output:
    ::

        Starting runnning deep learning models using 5-fold cross-validation, please be patient...
        running CNN...
        (some log information)
        finish
        running LSTM...
        (some log information)
        finish
        running GRU...
        (some log information)
        finish
        running ResNet-1D...
        (some log information)
        finish
        running MLP
        (some log information)
        finish
        running CapsuleNet...
        (some log information)
        finish
        All models have finished running, the result file are locate in ./

.. note:: The performance in the package is for reference only, and targeted hyperparameters need to be set for specific datasets to perform at their best.
