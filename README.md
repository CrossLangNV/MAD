# MAD
Misalignment detection (MAD)

Repository contains code for the paper: Misalignment detection for web scraped corpora: a supervised regression approach.

### Dependencies
python3 <br/>
numpy <br/>
scipy <br/>
scikit-learn <br/>
pandas <br/>
pytorch (tested with 0.4) <br/>
InferSent (git clone https://github.com/facebookresearch/InferSent.git ) <br/>

Dockerfile can be found in docker_mad/Dockerfile <br/>

### Labeled dataset
Labeled dataset used to train supervised model:   <br/>
DATA/supervised_dataset/labeled.tsv  <br/>

A tab separated file, with at each line:  <br/>
source    target    source_2_cross    target_2_cross    source_2_en    target_2_en  <br/>

### Notebook

Notebook where machine learning pipeline is explained:  <br/>
https://github.com/ArneDefauw/MAD/blob/master/1-MAD_notebook.ipynb

### Pre-trained MAD model
MAD/models/estimator_SVC.save

### Code to work with MAD
MAD.py

example:

python MAD.py \

--path_src en.sentences  #path to English sentences  \

--path_tgt ga.sentences #path to foreign sentences \

--path_tgt_translated ga.sentence.translated #path to foreign sentences translated to English  \

--output_folder /MAD
