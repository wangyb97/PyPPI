PyPPI.Features
==================================

.. py:method:: PyPPI.Features.generateBIOFeatures(ECO=False, HSP=False, scoringMatrix=1,RAA=False, Pro2Vec_1D=False, PSSM=False, Anchor=False,windowSize=1)

    This function is used to generate biological features.
    
    :Parameters:
        .. class:: ECO:bool, default=False
            
            Whether to use ECO algorithm to generate features.
        
        .. class:: HSP:bool, default=False
            
            Whether to use HSP algorithm to generate features.
        
        .. class:: scoringMatrix:int, default=1
            
            The scoring matrix used by HSP, its value is limited to ``[1, 2, 3]``. 
            
            1 means using PAM120, 2 means using BLOSUM80, and 3 means using BLOSUM62. The default value is 1.
        
        .. class:: RAA:bool, default=False
            
            Whether to use RAA algorithm to generate features.
        
        .. class:: Pro2Vec_1D:bool, default=False
            
            Whether to use Pro2Vec_1D algorithm to generate features.
        
        .. class:: PSSM:bool, default=False
            
            Whether to use PSSM algorithm to generate features.
            
        .. class:: Anchor:bool, default=False
        
            Whether to use Anchor algorithm to generate features.

        .. class:: windowSize:int, default=1
        
            Used to determine the final feature shape, can be set to any meaningful value. The default value is 1.
    
    :Attribute:
        .. class:: If using ECO, HSP, RAA, Pro2Vec_1D or Anchor:
        
            The shape will be generated as ``(total number of amino acids, windowSize, 1)``
        
        .. class:: If using PSSM:
        
            The shape will be generated as ``(total number of amino acids, windowSize, 20)``

    .. note:: The ``ECO``, ``HSP`` and ``PSSM`` algorithms are time-consuming, please choose according to your needs and wait patiently.

.. py:method:: PyPPI.Features.generatePhysChemFeatures(HYD=False, PHY_Prop=False, PKA=False, PHY_Char=False, windowSize=1)
    
    This function is used to generate physical and chemical features.
   
    :Parameters:
        .. class:: HYD:bool, default=False
            
            Whether to use HYD algorithm to generate features.
        
        .. class:: PHY_Prop:bool, default=False
            
            Whether to use PHY_Prop algorithm to generate features.
        
        .. class:: PKA:bool, default=False
            
            Whether to use PKA algorithm to generate features.
        
        .. class:: PHY_Char:bool, default=False
            
            Whether to use PHY_Char algorithm to generate features.

        .. class:: windowSize:int, default=1
        
            Used to determine the final feature shape, can be set to any meaningful value. The default value is 1.

    :Attribute:
        .. class:: If using HYD or PKA:
        
            The shape will be generated as ``(total number of amino acids, windowSize, 1)``
        
        .. class:: If using PHY_Prop:
        
            The shape will be generated as ``(total number of amino acids, windowSize, 7)``
            
        .. class:: If using PHY_Char:
        
            The shape will be generated as ``(total number of amino acids, windowSize, 3)``

    .. note:: Please choose features according to your needs.

.. py:method:: PyPPI.Features.generateStructuralFeatures(dssp=False, RSA=False, windowSize=1)

    This function is used to generate protein structural features.

    :Parameters:
        .. class:: dssp:bool, default=False
            
            Whether to use dssp algorithm to generate features.
        
        .. class:: RSA:bool, default=False
            
            Whether to use RSA algorithm to generate features.
        
        .. class:: windowSize:int, default=1
        
            Used to determine the ``RSA`` feature shape, can be set to any meaningful value. The default value is 1.
    
    .. note:: The calculation methods of ``dssp`` and ``RSA`` are different, so the shape dimension is different, but the generated features are both protein structural features. So set one of them to true. 
        
        And ``dssp`` only suppose the datasets of seq_186.fasta and seq_164.fasta.

.. py:method:: PyPPI.Features.generateLanguageModelFeatures(model='ProtT5')

    This function is used to generate Language Model features.

    :Parameters:
        .. class:: model:str, default=''
            
            Choose which algorithm to use. The default is the ProtT5 algorithm. Its value is limited to ``['ProtT5', 'ESM_1b', 'ProGen2']``
        
    :Attribute:
        .. class:: If using ProtT5:
        
            The shape will be generated as ``(1, total number of amino acids, 1024)``
        
        .. class:: If using ESM_1b:
        
            The shape will be generated as ``(total number of amino acids, 3, 33)``

        .. class:: If using ProGen2:
        
            The shape will be generated as ``(total number of amino acids, 32)``

    .. note:: ``ProtT5``, ``ESM_1b``, and ``ProGen2`` all use natural language processing to calculate protein semantic information, but different methods lead to different shapes. So choose one of them to use.