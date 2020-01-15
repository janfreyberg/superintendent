FROM continuumio/miniconda3:4.6.14-alpine

RUN /opt/conda/bin/pip install --upgrade pip

RUN mkdir /home/anaconda/app
WORKDIR /home/anaconda/app

# install some extra dependencies
RUN /opt/conda/bin/pip install voila>=0.1.2
RUN /opt/conda/bin/pip install superintendent

ENTRYPOINT ["/opt/conda/bin/voila", "--debug"]
CMD ["app.ipynb"]
