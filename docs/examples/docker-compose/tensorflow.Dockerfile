FROM tensorflow/tensorflow:latest-py3

RUN pip install superintendent

RUN mkdir /app
WORKDIR /app

ENTRYPOINT ["python"]
CMD ["app.py"]
