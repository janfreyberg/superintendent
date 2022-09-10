# Custom pre-processing steps as part of superintendent

Data often needs to be passed to the display widget and the model in different
formats. For displaying, since the display widget is custom and supplied by the
user, we recommend designing your own - and an example of this is shown here.

For modelling, you have multiple options. A common option which is very
transferable is to write a [scikit-learn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html),
in which you re-format your data so that the model you build can process it.

However, to make life slightly easier, superintendent also provides an optional
`model_preprocess` argument. This function needs to accept two arguments - x
and y - which are the features and annotations. It needs to return two values,
as well.

To demonstrate all components of pre-processing, we will write a tool to annotate
emails, and classify them.

## Data: Emails with metadata

First, let's create some dummy data:

```{jupyter-execute}
import pandas as pd

n_rows = 50

example_emails = [
    "Hi John,\nthis is just to say nice work yesterday.\nBest,\nJim",
    "Hi Mike,\nthis is just to say terrible work yesterday.\nBest,\nJim",
]

example_recipients = ["John", "Mike"]

example_timestamps = ["2018-02-01 15:00", "2018-02-01 15:03"]

example_df = pd.DataFrame({
    'email': example_emails,
    'recipient': example_recipients,
    'timestamp': example_timestamps
})

display(example_df)
```

As you can see, the dataframe contains several columns, and it would be nice
to display this data to the user for annotation. However, we also likely only
want to pass one column (the email text) to the model for classification.

## Display function and annotation widget

First, let's write a widget that displays the email in a way that's natural
to the user. We first define a function that accepts a single data point (a row
of our dataframe) and displays it:

```{jupyter-execute}
from IPython.display import display, Markdown

def display_email(row):
    """
    The display function gets passed your data - in the
    case of a dataframe, it gets passed a row - and then
    has to "display" your data in whatever way you want.
    
    It doesn't need to return anything
    """
    display(Markdown("**To:** " + row["recipient"]))
    display(Markdown("**At:** " + row["timestamp"]))

    display(Markdown(row["email"].replace("\n", "\n\n")))
```

With a display function like this, we can then define an annotation widget
with the {py:class}`ipyannotations.generic.ClassLabeller`:

```{jupyter-execute}
import ipyannotations.generic

annotation_widget = ipyannotations.generic.ClassLabeller(
    options=['positive', 'negative'],
    display_function=display_email,
)
annotation_widget.display(example_df.iloc[0])
annotation_widget
```

## Model Pipeline

We only want to pass the E-Mail text to our model, and to achieve this we can write a small pre-processing function that is applied to **both** the features and labels whenever a model is fit.

We then can write a model that uses scikit-learn's feature-vectorizer and applies a logistic regression.

```{jupyter-execute}
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def preprocessor(x, y):
    # only take Email column, leave everything else
    return x["email"], y

model = Pipeline([
    ('tfidf_vectorizer', TfidfVectorizer()),
    ('logistic_regression', LogisticRegression())
])
```

## The widget

Now that we have assembled the necessary components, we can create our widget:

```{jupyter-execute}
from superintendent import Superintendent

widget = Superintendent(
    features=example_df,
    model=model,
    model_preprocess=preprocessor,
    labelling_widget=annotation_widget,
    acquisition_function='margin',
)

widget
```
