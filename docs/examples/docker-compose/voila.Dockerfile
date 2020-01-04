FROM continuumio/miniconda3:4.6.14-alpine

RUN /opt/conda/bin/pip install --upgrade pip

RUN mkdir /home/anaconda/app
WORKDIR /home/anaconda/app

# install superintendent
RUN /opt/conda/bin/pip install superintendent

# install some extra dependencies
COPY docker-requirements.txt docker-requirements.txt
RUN /opt/conda/bin/pip install -r docker-requirements.txt

ONBUILD COPY . .
ONBUILD RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

ENTRYPOINT ["/opt/conda/bin/voila"]
CMD ["app.ipynb"]
