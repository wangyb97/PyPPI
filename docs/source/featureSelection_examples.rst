Feature selection examples
================================

The PyPPI integrates several feature selection methods and provides a simple interface, which requires only the features to be selected, the dataset label, and the number of features you want to selected.

Importing related modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: py

    from PyPPI.filesOperation import getLabel
    from PyPPI.Features import generateStructureFeatures, generateBIOFeatures, generatePhysChemFeatures, generateLanguageModelFeatures
    from PyPPI.featureSelection import cife # Here we use cife method as an example.

Prepare three types of features for feature selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The example uses the seq_72.fasta dataset.

.. code-block:: py

    # Set the fasta file path of the protein sequence
    # Edit bashrc file: vim ~/.bashrc
    export INPUT_FN=/home/wangyansong/wangyubo/PyPPI/datasets/seq_72.fasta

    # Set the label file path of the protein sequence
    label=getLabel(label_npy='datasets/label_72.npy')

    # generate biological features
    bioFeature=generateBIOFeatures(ECO=True, HSP=True, RAA=True, Pro2Vec_1D=True, PSSM=True, Anchor=True, windowSize=25)
    print('bioFeature:',bioFeature.shape)

    # generate physical and chemical information
    PhysChemFeatures=generatePhysChemFeatures(HYD=True, PHY_Prop=True, PKA=True, PHY_Char=True, windowSize=25)
    print('PhysChemFeatures:',PhysChemFeatures.shape)

    # generate structural Information
    structuralFeatures=generateStructuralFeatures(RSA=True)
    print('structuralFeatures:',structuralFeatures.shape)

    # generate semantic Information
    dynamicFeatures = generateLanguageModelFeatures(model='ProtT5')
    print('dynamicFeatures:',dynamicFeatures.shape)


output:
    ::

        bioFeature: (18140, 25, 25)
        PhysChemFeatures: (18140, 25, 12)
        structuralFeatures: (18140, 25, 1)
        dynamicFeatures: (1, 18140, 1024)

Feature selection procedure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It should be noted that the feature dimension to be passed into the feature selection method needs to be two-dimensional, so features with semantic information and secondary structure information need to be downscaled for feature selection (the same applies to machine learning classifiers).

.. code-block:: py

    # The first method of dimensionality reduction: multiplying the last two dimensions.
    print('-------------------first method------------------------')
    bioFeature_2D_1 = np.reshape(bioFeature, (bioFeature.shape[0], bioFeature.shape[1] * bioFeature.shape[2]))
    print(bioFeature_2D_1.shape)

    PhysChemFeatures_2D_1 = np.reshape(PhysChemFeatures, (PhysChemFeatures.shape[0], PhysChemFeatures.shape[1] * PhysChemFeatures.shape[2]))
    print(PhysChemFeatures_2D_1.shape)

    # The second method of dimensionality reduction: sum operation according to one of the last two dimensions.
    print('------------------second method-----------------------')
    print('----------compress the third dimension----------------')
    bioFeature_2D_2 = np.sum(bioFeature, axis=2)
    print(bioFeature_2D_2.shape)
    PhysChemFeatures_2D_2 = np.sum(PhysChemFeatures, axis=2)
    print(PhysChemFeatures_2D_2.shape)

    print('---------compress the second dimension----------------')
    bioFeature_2D_3 = np.sum(bioFeature, axis=1)
    print(bioFeature_2D_3.shape)
    PhysChemFeatures_2D_3 = np.sum(PhysChemFeatures, axis=1)
    print(PhysChemFeatures_2D_3.shape)


output:
    ::

        -------------------first method-----------------------
        (18140, 625)
        (18140, 300)
        ------------------second method-----------------------
        ----------compress the third dimension----------------
        (18140, 25)
        (18140, 25)
        ---------compress the second dimension----------------
        (18140, 25)
        (18140, 12)

Input the three processed features into the feature selection method for refinement (here we use the ``CIFE`` method as an example).

.. code-block:: py


    refined_biological_features = cife(bioFeature_2D_1, label, num_features=10)
    print(refined_biological_features.shape)

    refined_PhysChem_features = cife(PhysChemFeatures_2D_1, label, num_features=10)
    print(refined_PhysChem_features.shape)
    

output:
    ::

        (18140, 10)
        (18140, 10)

.. note:: The calculation process of some feature selection methods is more complicated, so the running time is longer, please be patient.