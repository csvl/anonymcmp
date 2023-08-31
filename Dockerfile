FROM tensorflow/tensorflow:2.8.3-gpu-jupyter

RUN apt-get update && apt-get install -qq -y \
  build-essential wget --no-install-recommends

ENV INSTALL_PATH /anonym

ENV PYTHONPATH="$PYTHONPATH:$INSTALL_PATH"

WORKDIR $INSTALL_PATH

RUN pip install --upgrade pip

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY mondrian mondrian
COPY anonymcmp_utils anonymcmp_utils

RUN wget https://github.com/IBM/differential-privacy-library/archive/refs/tags/0.6.2.tar.gz \
&& tar zxvpf 0.6.2.tar.gz --strip-components=1 && rm 0.6.2.tar.gz 

ENV NOTEBOOKS_PATH /tf/dev/notebooks
COPY notebooks $NOTEBOOKS_PATH
RUN mkdir -p $NOTEBOOKS_PATH/datasets
WORKDIR $NOTEBOOKS_PATH

ENV TESTS_PATH /tf/dev/tests
COPY tests/ $TESTS_PATH/
RUN mkdir -p $TESTS_PATH/datasets

