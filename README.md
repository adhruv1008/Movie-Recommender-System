# Movie-Recommender-System

The Movie-Recommender-System is a restricted boltzmann machine model trained on dataset having 1 million ratings by 6000 users on 4000 movies.

Link to dataset - http://files.grouplens.org/datasets/movielens/ml-1m.zip

Users have rated their movies on the scale of 1-5, but the output of the recommender system is binary i.e. 1 for a like and 0 for a dislike. Ratings 1 and 2 are treated as dislike and ratings 3,4 and 5 are treated as like

I have used mean absolute error as the measure of performance of the model, the model gives a score of 0.35 i.e. for a user rating 100 movies , the model predicts the choice correctly for 65 movies out of 100
