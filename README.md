# Movie-Recommender-System

A content based movie recommender system which takes as input the movies watched by the user and suggests movies based on genre and the titles.

## Data
The *IMDb movie dataset* was used to get the genres to which each movie belongs. The following 10 most frequently occuring genres were considered and the genres for each movie were one-hot encoded according to the acquired data : *Action, Adventure, Fantasy, Sci-fi, Thriller, Comedy, Romance, Mystery, Horror, Animation*.
## Model
Given the genres of the movies watched by the user, we train the model to calculate the likeliness of the user towards the different genres using logistic regression. 
## Predictions
Given a movie from the dataset, we can now find the extent to which the user would like the movie based on the genres of the test movie using the parameters calculated for the genres. We could then calculate this likeliness for all the movies and suggest the most likely ones. Apart from the genres, movies with similar titles are also suggested, based on the number of common words between the watched and the unwatched movies.