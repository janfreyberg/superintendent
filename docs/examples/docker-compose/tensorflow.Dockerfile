FROM tensorflow/tensorflow:2.10.0

RUN pip install "ipyannotations>=0.5.1"
RUN pip install --pre superintendent==0.6.0.rc1

RUN mkdir /app
WORKDIR /app

ENTRYPOINT ["python"]
CMD ["app.py"]
