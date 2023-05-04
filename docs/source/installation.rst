Installation
===========================

Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The following packages are requirements:

- ``gensim``
- ``glove``

这里写requirement.txt里的包

Install PyPPI dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    conda create -n PyPPI python=3.7.7
    conda activate PyPPI
    git clone https://github.com/wangyb97/PyPPI.git
    cd PyPPI
    pip install -r requirement.txt
    
    chmod +x chmodFiles.sh
    ./chmodFiles.sh

Set path in bashrc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the following method to set PRO_DIR, INPUT_FN and TMP_DIR in bashrc, and replace the path with your own.

Edit bashrc file: ``vim ~/.bashrc``

::

    # Set the project path
    export PRO_DIR=/home/wangyansong/wangyubo/PyPPI
    # Set the fasta file path of the protein sequence
    export INPUT_FN=/home/wangyansong/wangyubo/PyPPI/datasets/seq_2.fasta
    # Set folder path for temporary features
    export TMP_DIR=/home/wangyansong/wangyubo/PyPPI/feature_computation/txtFeatures

If using the ECO algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose one of the following four methods to install hhsuite
-------------------------------------------------------------------

::

    # install via conda
    conda install -c conda-forge -c bioconda hhsuite 
    # install docker
    docker pull soedinglab/hh-suite
    # static SSE2 build
    wget https://github.com/soedinglab/hh-suite/releases/download/v3.3.0/hhsuite-3.3.0-SSE2-Linux.tar.gz; tar xvfz hhsuite-3.3.0-SSE2-Linux.tar.gz; export PATH="$(pwd)/bin:$(pwd)/scripts:$PATH"
    # static AVX2 build
    wget https://github.com/soedinglab/hh-suite/releases/download/v3.3.0/hhsuite-3.3.0-AVX2-Linux.tar.gz; tar xvfz hhsuite-3.3.0-AVX2-Linux.tar.gz; export PATH="$(pwd)/bin:$(pwd)/scripts:$PATH"

.. note:: AVX2 is roughly 2x faster compared to SSE2. But the system needs to support AVX2.

Set database
-------------

`Download uniprot20_2015_06 database <https://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/old-releases/uniprot20_2015_06.tgz>`_


::

    tar zxvf uniprot20_2015_06.tgz -C specifiedPath # decompress to the specified path.

Edit bashrc file: ``vim ~/.bashrc``, and replace the path with your own.

::

    export database="/home/wangyansong/wangyubo/PyPPI/feature_computation/ECO/uniprot20_2015_06/uniprot20_2015_06"
    export hhblits="/home/wangyansong/wangyubo/programs/hhsuite-3.3.0-AVX2/bin/hhblits"

If using the HSP algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install SPRINT
--------------------

Make sure that g++, boost library and OpenMP is available on the system.

::

    git clone https://github.com/lucian-ilie/SPRINT.git
    git checkout DELPHI_Server
    make compute_HSPs_parallel

Edit bashrc file: ``vim ~/.bashrc``

::

    # Set your own SPRINT_program path
    export SPRINT_program=/home/wangyansong/wangyubo/programs/SPRINT/bin/compute_HSPs

If using the PSSM algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Make sure that ``blast`` program is available on the system.

.. ::

..     sudo apt-get install ncbi-blast+

Edit bashrc file: ``vim ~/.bashrc``

::

    # Set your own pssm_database path
    export pssm_database_path=/home/houzilong/uniref90/uniref90

