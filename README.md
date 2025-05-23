# KTT-Project

- Source Files:
  - project.ipynb: notebook to run the analysis step by step
  - utlil_func.py: source code contains utility functions
  - feature_config.py: config file defines constant variables
  - Project_Report.ipynb: notebook to generate the final report
  - Project_Report.html: Final Project Report


- Notes:
  - Make sure to follow the guidance in 'project.ipynb' to run the analysis step by step, espeicially creating the data folders to save intermediate results as described.
  - To generate the report from notebook, just run below command in terminal:
     jupyter nbconvert Project_Report.ipynb --to html   --TagRemovePreprocessor.enabled=True   --TagRemovePreprocessor.remove_input_tags='["remove-input"]'
