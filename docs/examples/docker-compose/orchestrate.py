from sklearn.linear_model import LogisticRegression

from superintendent.distributed import SemiSupervisor

model = LogisticRegression(solver="lbfgs", max_iter=2000, multi_class="auto")

db_connection = (
    "postgresql+psycopg2://superintendent:superintendent" "@db:5432/labelling"
)

widget = SemiSupervisor.from_images(
    connection_string=db_connection,
    options=range(10),
    classifier=model,
    reorder="entropy",
    shuffle_prop=0.2,
)

widget.orchestrate(interval_seconds=30, interval_n_labels=10)
