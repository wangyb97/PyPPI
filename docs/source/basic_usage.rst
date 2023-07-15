PyPPI basic usage flow
=================================
This example illustrates the basic usage of ``PyPPI``, including loading the dataset, generating features, feature selection, training the model, and performance and feature analysis.

This example uses:

- ``PyPPI.filesOperation``
- ``PyPPI.Features``
- ``PyPPI.evaluateClassifiers``
- ``PyPPI.metricsPlot``
- ``PyPPI.featureSelection``

.. code-block:: py

    from PyPPI.filesOperation import getLabel
    from PyPPI.Features import generateStructureFeatures, generateBIOFeatures, generatePhysChemFeatures, generateLanguageModelFeatures
    from PyPPI.featureSelection import cife # Here we use cife method as an example.
    from PyPPI.evaluateClassifiers import evaluateDLclassifers, evaluateMLclassifers
    from sklearn.svm import SVC

Load the dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Load a seq_72 dataset as example.

.. code-block:: py

    # Set the fasta file path of the protein sequence, edit bashrc file: vi ~/.bashrc
    export INPUT_FN=/home/wangyansong/wangyubo/PyPPI/datasets/seq_72.fasta 

    # Set the label file path of the protein sequence
    label=getLabel(label_npy='datasets/label_72.npy')

Generate features for sequences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generate biological features, physicochemical information, structural information and semantic information.

.. code-block:: py

    # generate biological features
    bioFeature=generateBIOFeatures(ECO=True, HSP=True, RAA=True, Pro2Vec_1D=True, PSSM=True, Anchor=True, windowSize=25)
    print('bioFeature:',bioFeature.shape)
    # generate physical and chemical information
    PhysChemFeatures=generatePhysChemFeatures(HYD=True, PHY_Prop=True, PKA=True, PHY_Char=True, windowSize=25)
    print('PhysChemFeatures:',PhysChemFeatures.shape)
    # generate structural information
    structuralFeatures=generateStructuralFeatures(RSA=True)
    print('structuralFeatures:',structuralFeatures.shape)
    # generate semantic information
    dynamicFeatures = generateLanguageModelFeatures(model='ProtT5')
    print('dynamicFeatures:',dynamicFeatures.shape)

Perform feature selection to refine the biological features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We take the cife method as example.

.. code-block:: py

    print('bioFeature:',bioFeature.shape)
    bioFeature_2D = np.sum(bioFeature, axis=2)
    print('bioFeature_2D:',bioFeature_2D.shape)
    refined_biological_features = cife(bioFeature_2D, label, num_features=10)
    print('refined_biological_features:',refined_biological_features.shape)

The output of the above code is as follows:

    ::

        bioFeature: (18140, 25, 25)
        bioFeature_2D: (18140, 25)
        refined_biological_features: (18140, 10)

Evaluate deep/machine learning classifier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We take the deep learning models as example.

.. code-block:: py

    evaluateDLclassifers(PhysChemFeatures, folds=10, labels=label, file_path='./', shuffle=True) # 10-fold cross-validation of deep learning models using dynamic semantic information

After the function finishes running, a ``DL_evalution_metrics.csv`` will be saved in the path specified by ``file_path``, with the following format.

    ::

        clf_name,metrics,metric_name
        CNN,0.999745345,AUC
        CNN,0.995525096	ACC
        CNN,0.991053703,MCC
        CNN,0.994975596,Recall
        CNN,0.9955248,F1_Scores
        LSTM,0.998232352,AUC
        LSTM,0.988452081,ACC
        LSTM,0.976941921,MCC
        LSTM,0.983881982,Recall
        LSTM,0.988370228,F1_Scores
        GRU,0.987232352,AUC
        GRU,0.979452081,ACC
        GRU,0.986741921,MCC
        GRU,0.990881982,Recall
        GRU,0.982370228,F1_Scores
        ResNet,0.999630465,AUC
        ResNet,0.995900484,ACC
        ResNet,0.991807785,MCC
        ResNet,0.994295177,Recall
        ResNet,0.995894144,F1_Scores
        MLP,0.988894799,AUC
        MLP,0.935463968,ACC
        MLP,0.867734521,MCC
        MLP,0.978133195,Recall
        MLP,0.951769181,F1_Scores
        CapsuleNet,0.994232352,AUC
        CapsuleNet,0.985452081,ACC
        CapsuleNet,0.971941921,MCC
        CapsuleNet,0.983851982,Recall
        CapsuleNet,0.987650227,F1_Scores


Visualize performance and feature analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We use the SVM trained with refined_biological_features for feature shap value calculation as an example.

.. code-block:: py

    clf = SVC(probability=True)
    shap_interaction_scatter(refined_biological_features, label, clf=clf, sample_size=(0, 100), feature_size=(0, 10), image_path='./')  # Plotting the interaction between biological features in SVM

An ``interaction_scatter.png`` will be saved under ``file_path``, as follows.

.. image:: ./images/interaction_scatter.png
    :align: center
    :alt: shap_interaction_scatter
