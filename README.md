# TypicalityCoding

Materials to support the blog article [Detecting Data Differences Using the Sphering Transform](https://win-vector.com/2023/08/20/detecting-data-differences-using-the-sphering-transform/).

[Simple example notebook](https://github.com/WinVector/TypicalityCoding/blob/main/example_sphering_transform.ipynb) of how to use an embedding plus sphering/whitening transform to measure difference in distribution.  

Further Examples:
* [The sphering transform for anomaly detection on CNC vibration data.](https://github.com/WinVector/TypicalityCoding/blob/main/test_cnc_anomaly.ipynb)
* [An example of applying the sphering transform to text embeddings: detecting changes in a text corpus.](https://github.com/WinVector/TypicalityCoding/blob/main/text_embedding_example.ipynb)

To install environment:
    `conda env create --file=tc_env_def.yaml`

Specific versions used saved as:
    `conda list --export > tc_env_list.txt`

To use environment:
    `conda activate tc_env`

Download data from:
    https://drive.google.com/drive/folders/1sXtDqNF-D0mypORzfevir_eH9FdG2668?usp=sharing
    and then unzip 

To run tests:
    `python -m pytest tests`

