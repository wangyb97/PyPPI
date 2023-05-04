PyPPI.featureSelection
==============================================

The ``PyPPI`` integrates feature selection methods based on four different evaluation approaches (information theoretical based, similarity based, sparse learning based and statistical based).

Information theoretical based
----------------------------------------------

.. py:method:: PyPPI.featureSelection.cife(features, labels, num_features=10)

    This function uses Conditional Infomax Feature Extraction [CIFE]_ for feature selection and returns the corresponding best feature matrix.

    :Parameters:
        .. class:: features:numpy array, shape (n_samples, n_features)

                Sample feature matrix to be processed.

        .. class:: labels:numpy array, shape (n_samples, )

                Class labels according to the input samples.

        .. class:: num_features:int, default=10

                Number of selected features.

.. [CIFE] Dahua Lin and Xiaoou Tang. 2006. Conditional infomax learning: An integrated framework for feature extraction and fusion. In ECCV. 68–82.