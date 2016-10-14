## Surge Forecast

Matthew Swogger, October 2016

## Forecast of first 8 points

![](forecast_1.png)

## Forecast of second 8 points

![](forecast_2.png)

#### Overview

View the app live at Ryd.io

The idea of this project is to allow a user, or marketing lead, to explore and learn about a city from the frequency and volume of taxi drop off points to focus marketing efforts. My goal was to provide an alternative view of a city that has been clustered into subgroups outlining different weekly and daily ride distributions.

#### Example

The model is built to predict where and when and how many rides will get dropped off given any location in NYC. Every single block in NYC has as a story to tell. Finding a resource that can tell this story in a meaningful manner is a difficult task. A marketing team foreign to NYC will have little idea about how to navigate this concrete jungle.

Using Ryd.io this team will be able to focus their efforts accordingly and minimize misallocation of resources. Let's check out an example query

#### Example query: "Terra Blues, Thursday 9pm"



#### How it works

Once a user enters a query, Google API returns a latitude and longitude of that location...and then the magic happens.

With that latitude and longitude we pull a geo bounding box of data from elastic search signaling to us the relavent rides. Now that we know how many rides came into the users location per day for the year, we can feed this information into our SARIMA model. The output of the SARIMA model will tell us how many rides it predicts that location will have at a given point in the future.

Knowing how many rides will land in a day is great, but we really would like to know at which hour these rides will come. Utlizing the cluster map (below) we can find the nearest cluster point that the users location is near and apply that hourly distribution to our 'rides per day estimate' output from the SARIMA model. This output signals to us how many rides will come in a given hour.

But where do we put the points on the map? We can't just put them randomly anywhere in our bounding box...some of them might land on top of buildings...not great. In order to adjust for this we will use a multivariate KDE that randomly resamples from a distrubtion that was built off of where rides appeared in the past. Basically we are saying..."hey Ryd, where have all the rides landed for this location the past? Now if you were to guess where these rides would arrive this time where would it be?"



#### Project Pipeline

NYC Taxi Data Set -> Elasticsearch/BigQuery
Return per day totals for over 9000 points in NYC -> Python Pandas
Save this file as json/csv for future use
User queries location -> Flask/AJAX/Python
Pull data for a user queried location -> Elastic
Train SARIMA model -> sklearn
Cross reference daily estimate with cluster hourly distribution
Apply resampled KDE with hourly distrubtion for geo estimates
Project points on map and display meta data

#### Future Steps

### Viz
To have this tool have appeal to a non-technical audience I'd like to make a much more interactive web-app complete with sliders and more minute query preferences at the users disposal. It would be awesome to be able to ask Ryd, "I want to see map of high density areas that have been labeled with art, farmers markets, and have low crime."

### Model
As Ryd stands right now we are focused solely on dropoff locations and times. However, dropoff locations and times make up only 15% of the information available through the NYC Taxi data set. There is a TON more analysis to be done on pick up locations, number of people riding in the car and combinations of it all.

Once we're ready, we can easily incorporate other data sources as well. It would be great to see an overlay of restaurants and bars in NYC and the correlate the connection between drop offs and establishment frequency.

#### Packages used

* bigquery for python
* pandas
* time
* numpy
* itertools
* os.istdir
* glob
* os
* rv_discrete
* scipy.stats
* requests
* json
* threading
* Queue
* rauth
* BeautifulSoup
* Matplotlib.pyplot
