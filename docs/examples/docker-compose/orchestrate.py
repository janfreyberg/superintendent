import time

from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sqlalchemy.exc import OperationalError
from superintendent.distributed import SemiSupervisor

model = LogisticRegression(solver="lbfgs", max_iter=2000, multi_class="auto")

db_connection = (
    "postgresql+psycopg2://superintendent:superintendent" "@db:5432/labelling"
)

time.sleep(15)
connection_made = False
tries = 0

# wait for our database to come up:
while not connection_made and tries < 25:
    try:
        widget = SemiSupervisor.from_images(
            connection_string=db_connection,
            options=range(10),
            classifier=model,
            reorder="entropy",
            shuffle_prop=0.2,
        )
        connection_made = True
    except OperationalError:
        time.sleep(2)
        tries += 1

# if we've never added any data to this db, load it and add it:
if len(widget.queue) == 0:
    digit_data = load_digits().data
    widget.add_features(digit_data)

widget.orchestrate(interval_seconds=30, interval_n_labels=10)
