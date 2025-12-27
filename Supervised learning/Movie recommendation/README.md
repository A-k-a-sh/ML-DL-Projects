# Movie Recommendation System

Content-based movie recommendation system using the TMDB (The Movie Database) dataset.

## Overview
- Builds a recommendation engine to suggest similar movies
- Uses movie metadata including titles, genres, cast, and crew
- Implements content-based filtering approach
- Works with TMDB 5000 Movie Metadata

## Dataset
- TMDB 5000 Movies Dataset
  - `tmdb_5000_movies.csv`: Movie information
  - `tmdb_5000_credits.csv`: Cast and crew details
- Contains movie titles, genres, budgets, revenues, and more
- Includes cast and crew information for each film

## Features
- Movie metadata analysis
- Content-based similarity calculations
- Recommendation generation based on movie features
- Data merging from multiple sources (movies and credits)

## Technologies Used
- NumPy, Pandas
- Matplotlib (for visualization)
- scikit-learn (potentially for similarity metrics)

## Approach
- Exploratory data analysis of movie metadata
- Feature engineering from movie attributes
- Similarity computation between movies
- Recommendation generation based on content similarity

## Use Cases
- Find movies similar to ones you've enjoyed
- Discover films based on genres, cast, or crew
- Explore movie databases systematically
