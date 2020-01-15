import time
import os

import sqlalchemy
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score
from tensorflow import keras
from superintendent.distributed import ClassLabeller


def keras_model():
    model = keras.models.Sequential(
        [
            keras.layers.Conv2D(
                filters=8,
                kernel_size=3,
                activation="relu",
                input_shape=(8, 8, 1),
            ),
            keras.layers.MaxPool2D(2),
            keras.layers.Conv2D(filters=16, kernel_size=3, activation="relu"),
            keras.layers.GlobalMaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        keras.optimizers.Adam(), keras.losses.CategoricalCrossentropy()
    )
    return model


def evaluate_keras(model, x, y):
    return cross_val_score(model, x, y, scoring="accuracy", cv=3)


def wait_for_db(db_string):
    database_up = False
    connection = sqlalchemy.create_engine(db_string)
    while not database_up:
        time.sleep(2)
        try:
            print("attempting connection...")
            connection.connect()
            database_up = True
            print("connected!")
        except sqlalchemy.exc.OperationalError:
            continue


model = keras.wrappers.scikit_learn.KerasClassifier(keras_model, epochs=5)

user = os.getenv("POSTGRES_USER")
pw = os.getenv("POSTGRES_PASSWORD")
db_name = os.getenv("POSTGRES_DB")

db_string = f"postgresql+psycopg2://{user}:{pw}@db:5432/{db_name}"

# wait some time, so that the DB has time to start up
wait_for_db(db_string)

# create our superintendent class:
widget = ClassLabeller(
    connection_string=db_string,
    model=model,
    eval_method=evaluate_keras,
    acquisition_function="entropy",
    shuffle_prop=0.1,
    model_preprocess=lambda x, y: (x.reshape(-1, 8, 8, 1), y),
)

# if we've never added any data to this db, load it and add it:
if len(widget.queue) == 0:
    digit_data = load_digits().data
    widget.add_features(digit_data)

if __name__ == "__main__":
    widget.orchestrate(interval_seconds=30, interval_n_labels=10)
