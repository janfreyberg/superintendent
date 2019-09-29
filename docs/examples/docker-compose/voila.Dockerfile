FROM continuumio/miniconda3:4.6.14-alpine

RUN /opt/conda/bin/pip install --upgrade pip

RUN mkdir /home/anaconda/app
WORKDIR /home/anaconda/app

COPY docker-requirements.txt docker-requirements.txt
RUN /opt/conda/bin/pip install -r docker-requirements.txt

# install superintendent from pypi
COPY . .
RUN /opt/conda/bin/pip install --user .
# RUN /opt/conda/bin/pip install superintendent

ONBUILD COPY . .
ONBUILD RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

ENTRYPOINT ["/opt/conda/bin/voila"]
CMD ["app.ipynb"]
