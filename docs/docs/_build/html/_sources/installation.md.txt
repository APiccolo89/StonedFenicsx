# Installation 
First, the user needs to be in the folder containing **stonedfenicsx**, then, the user needs to open the terminal. To install the package is necessary to have conda (miniforge,conda). The first step is to use the following command in the terminal: 
``
conda env create -f stonedenvironment.yml
`` 
After all the required packages are installed, it is necessary to install **stonedfenicsx** as an editable package. 
``
pip install -e .
``
All the dependency are carried out by the *.toml* and *.yml* files. After the installation, the user can check whether or not the package has been installed by opening the REPL of python and typing: 
``
import stonedfenicsx
``

**Note**: Certain clusters have problem with *conda* installations and in general with the MPI of conda. In this case, it is necessary to understand how to install 
the package only via *.toml* file, and then use other containers (e.g. *spack*). In the main folder, the user can find an example used to install the package on the Aire cluster of Leeds. 