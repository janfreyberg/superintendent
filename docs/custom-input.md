(custom-input)=
# Writing a custom input widget

Since superintendent does not provide any annotation facilities - i.e. the UI
elements that capture the labels and annotations - you will have to either
employ third party UI elements, or write your own.

An annotation widget needs to implement two or three methods:

1. A `display` method which accepts a data point and displays it to the user.
2. The method `on_submit`, which needs to be triggered whenever the user submits
   an annotation. Superintendent will use this to hook in and store the
   annotations.
3. The optional method `on_undo`. This does not need to exist but can be nice,
   as users likely want to undo submission of annotations from time to time.

## Capturing user ratings

An example might be a user rating on a sliding scale. This is common in
psychology (the [Likert scale](https://en.wikipedia.org/wiki/Likert_scale)).

To start, we will need to import ipywidgets, which is the original jupyter
widget library:

### Slider widgets for user ratings

```{jupyter-execute}
import ipywidgets
```

Ipywidgets has a suitable widget by default, which we are going to adapt:

```{jupyter-execute}
ipywidgets.SelectionSlider(
    options=[
        'strongly disagree',
        'disagree',
        'neither agree nor disagree',
        'agree',
        'strongly agree',
    ],
    value='neither agree nor disagree',
    orientation='horizontal',
    readout=True
)
```

However, by itself this widget does not allow the user to submit their rating
once they think it is correct. This needs to be implemented by us. One way to
do this is to add a button that users click once they are ready.

```{jupyter-execute}
ipywidgets.Button(
    description='Submit',
    button_style='success',
    icon='check'
)
```

### Combining two widgets into one

Let's put this together as a single class.


```{jupyter-execute}
class LikertScale(ipywidgets.HBox):

    def __init__(self):

        self.slider = ipywidgets.SelectionSlider(
            options=[
                'strongly disagree',
                'disagree',
                'neutral',
                'agree',
                'strongly agree',
            ],
            value='neutral',
            layout=ipywidgets.Layout(width='300px'),
            readout=False,
        )
        self.button = ipywidgets.Button(
            description='Submit',
        )
        ipywidgets.link((self.slider, 'value'), (self.button, 'description'))

        super().__init__([self.slider, self.button])

LikertScale()
```

### Adding data annotation methods

Now that we have a layout of two widgets that work together to capture user
input, we need to add the methods that will be triggered when the user clicks
submit.

For the widget to conform to the API expected by `superintendent`, we need to
add:

1. A display method, which accepts a data point. For now, let's simply print
   the input.
2. An `on_submit` method, which accepts a function that will then get triggered
   when the user clicks submit.

```{jupyter-execute}
class LikertScale(ipywidgets.VBox):

    def __init__(self):

        self.slider = ipywidgets.SelectionSlider(
            options=[
                'strongly disagree',
                'disagree',
                'neutral',
                'agree',
                'strongly agree',
            ],
            value='neutral',
            layout=ipywidgets.Layout(width='300px'),
            readout=False,
        )
        self.button = ipywidgets.Button(
            description='Submit',
        )
        self.button.on_click(self.submit)
        ipywidgets.link((self.slider, 'value'), (self.button, 'description'))
        self.output = ipywidgets.Output()

        self.submission_functions = []

        super().__init__([
            self.output,
            ipywidgets.HBox([self.slider, self.button])
        ])
    
    def display(self, data_point):
        with self.output:
            print(data_point)

    def on_submit(self, callback):
        self.submission_functions.append(callback)
    
    def submit(self, *_):
        for callback in self.submission_functions:
            callback(self.slider.value)

widget = LikertScale()

widget.display('Do you agree with this statement?')
widget.on_submit(lambda annotation: print(annotation))
widget
```

At this point, we have a functioning annotation widget for Superintendent!
When you pass this annotation UI to Superintendent, it will call the `on_submit`
method to register a callback that allows Superintendent to hook into the
annotation workflow. And when Superintendent advances to the next data point,
it will call `display` in order to show the data point to the user.

If you want to, you can also add an "Undo" button, in which case you would need
to add an `on_undo` callback registration method.
