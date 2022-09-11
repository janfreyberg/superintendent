# Labelling text

Let's assume we have a text dataset that contains some labelled sentences and some unlabelled sentences. For example, we could get the headlines for a bunch of UK news websites (the code for this comes from the great github project [compare-headlines](https://github.com/isobelweinberg/compare-headlines/blob/master/scrape-headlines.ipynb) by [isobelweinberg](https://github.com/isobelweinberg)):

```{jupyter-execute}
import requests
from bs4 import BeautifulSoup
import datetime

headlines = []
labels = []

r = requests.get('https://www.theguardian.com/uk').text #get html
soup = BeautifulSoup(r, 'html5lib') #run html through beautiful soup
headlines += [headline.text for headline in
              soup.find_all('span', class_='js-headline-text')][:10]
labels += ['guardian'] * (len(headlines) - len(labels))

soup = BeautifulSoup(requests.get('http://www.dailymail.co.uk/home/index.html').text, 'html5lib')
headlines += [headline.text.replace('\n', '').replace('\xa0', '').strip()
              for headline in soup.find_all(class_="linkro-darkred")][:10]
labels += ['daily mail'] * (len(headlines) - len(labels))
```

```{jupyter-execute}
from superintendent import Superintendent
from ipyannotations.text import ClassLabeller

input_widget = ClassLabeller(options=['professional', 'not professional'])
input_data = headlines
data_labeller = Superintendent(
    features=input_data,
    labelling_widget=input_widget,
)

data_labeller

```

For further options of labelling text, including tagging parts of text, check
the [text widgets](https://ipyannotations.readthedocs.io/en/latest/widget-list.html#text-widgets)
in ipyannotations.
