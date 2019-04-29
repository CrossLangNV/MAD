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
source /t target /t source_2_cross /t target_2_cross /t source_2_en /t target_2_en /t MAD_score  <br/>

### Notebook

Notebook where machine learning pipeline is explained:  <br/>
https://github.com/ArneDefauw/MAD/blob/master/1-MAD_notebook.ipynb

### Pre-trained MAD model
MAD/models/estimator_SVC.save

### Code to work with MAD
MAD.py

example:

if en.sentences is a file containing English sentences; ga.sentences is file containing Irish sentences; and ga.sentences.translated contains translation of ga.sentences to English, then we can calculate MAD score for each sentence pair via. Code will create a file MAD_scores with the MAD score for each sentence pair.


python MAD.py \

--path_src en.sentences   \

--path_tgt ga.sentences \

--path_tgt_translated ga.sentence.translated  \

--output_folder /MAD
