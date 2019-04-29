FROM nvidia/cuda:8.0-runtime-ubuntu16.04

MAINTAINER arne <arnedefauw@gmail.com>

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

# CUDA 8.0-specific steps
RUN conda install -y -c pytorch \
    cuda80=1.0 \
    magma-cuda80=2.3.0 \
    pytorch=0.4.1 \
#    torchvision=0.2.1 \
 && conda clean -ya

# Install HDF5 Python bindings
RUN conda install -y \
    h5py \
 && conda clean -ya
RUN pip install h5py-cache

# Install Torchnet, a high-level framework for PyTorch
RUN pip install git+https://github.com/pytorch/tnt.git@master

# Install Requests, a Python library for making HTTP requests
RUN conda install -y requests && conda clean -ya

# Install Graphviz
RUN conda install -y graphviz=2.38.0 \
 && conda clean -ya
RUN pip install graphviz


# Python packages from conda
#RUN conda install -y \
#    scikit-learn \
#    scikit-image \
#    flask \
#    pillow \
#&& conda clean -ya

RUN pip install -U scikit-learn==0.19

RUN conda install faiss-gpu -c pytorch \
&& conda clean -ya

#Some python packages from pip
RUN pip install \ 
        gensim \
        ipython \
        jupyter \
        #tensorflow cpu (for GPflowopt)
        --upgrade tensorflow==1.6 \
        gensim \
       nltk 

RUN apt-get update

RUN apt-get -y install --reinstall build-essential

RUN apt-get -y install gcc

RUN pip install Cython 


#install beautifullsoup
RUN pip install beautifulsoup4
RUN pip install lxml

RUN pip install -U spacy
RUN pip install langdetect
RUN pip install editdistance
RUN pip install pytorch-nlp
RUN pip install mosestokenizer
RUN pip install seaborn
RUN pip install python-Levenshtein
RUN pip install pyxDamerauLevenshtein


RUN pip install --upgrade pip


EXPOSE 8888

VOLUME /notebook
WORKDIR /notebook

CMD ["jupyter", "notebook", "--no-browser", "--allow-root", "--ip='0.0.0.0'"]


