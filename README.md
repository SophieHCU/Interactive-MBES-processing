# Interactive-MBES-processing

Tool based on Python and Jupyter Notebook for interactive processing of Kongsberg MBES bathymetry and backscatter data. Consists of:

- Kongsberg ALL preprocessing module for bathymetry and backscatter
- Jupyter Notebook for processing (filters and corrections) based on Entwine, PDAL and Potree

Important notes:

- To recreate the conda environment two options can be used (:
	- By using the `spec-file.txt` [file](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#building-identical-conda-environments) &rightarrow; conda command `conda create --name myenv --file spec-file.txt`
		- myenv should be replaced by the desired environment name
		- Will work only under win-64
	- Or alternatively by using the `enironment.yml` [file](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#sharing-an-environment) &rightarrow; conda command `conda env create -f environment.yml`

- Hints and troubleshooting:
	- When activating the `Hide input all` extension, the code cells can be hidden which tiedies the notebook up.
	- Some of the cells are locked with the `Freeze` extension and have to be unlocked for modification.
	- When the Potree viewer does not update, this is most probably caused by the browser cache. Deactivating or clearing it will help.
