# DS 4320 Project 2: <PROJECT TITLE>

### Executive Summary
This repository contains the complete data pipeline and analysis for predicting performance decline in professional soccer players. Using the European Soccer Database (Kaggle), I built a relational database with DuckDB, performed feature engineering to create player-specific and match-context features, and trained multiple machine learning models (Random Forest, Gradient Boosting, Logistic Regression) to predict when a player's performance will decline in their next match. The Logistic Regression model achieved the best performance with a ROC-AUC of 0.938, demonstrating that recent form (rolling 3-game average) is the strongest predictor of future performance decline.

| Spec | Value |
| --- | --- |
| Name | Crystal McEnhimer |
| NetID | vcx4ka |
| DOI | [![DOI](https://zenodo.org/badge/1189840726.svg)](https://doi.org/10.5281/zenodo.19378548) |
| Press Release | [PRESS-RELEASE.md](PRESS-RELEASE.md) |
| Data | [UVA OneDrive Data Folder](https://myuva-my.sharepoint.com/:f:/g/personal/vcx4ka_virginia_edu/IgDkm3bSO4ddRrfrgJUvnP2FARFKLKoxtnxQnPdcKKCrPbU?e=kzwiJS) |
| Pipeline | [solution-pipeline.ipynb](solution-pipeline.ipynb) [solution-pipeline.md](solution-pipeline.md) |
| License State | [MIT License](LICENSE.md) |

---
<br>

<br>

## Problem Definition
### General and Specific Problem
* **General Problem:** Projecting athletic performance.
* **Specific Problem:** Using historical match data from professional soccer players, we aim to predict whether a player’s performance in a given match will decline compared to their previous match based on their recent performance, technical skills, physical attributes, and match context.
### Rationale
While athletic performance is influenced by many factors such as fitness, fatigue, and strategy, detailed wellness data is often unavailable in public datasets. Therefore, this project refines the problem to focus on observable match performance metrics, such as finishing, dribbling, passing, and player ratings. By using these measurable indicators, we can construct a consistent and scalable definition of performance and analyze trends over time.
### Motivation
Predicting declines in player performance has practical applications in sports analytics, including lineup decisions, player rotation, and scouting. By identifying patterns that precede performance drops, teams can make more informed decisions to maintain competitive advantage. This project demonstrates how publicly available match data can be transformed into actionable insights using data science techniques.
### Press Release Headline and Link
[**New Machine Learning Model Predicts Soccer Player Performance with High Accuracy**](PRESS-RELEASE.md)

<br>
<br>

## Domain Exposition

### Terminology
| Term | Definition | Context in Project |
|------|------------|------|
| **Player** | An individual professional soccer athlete | Each player has a unique ID and attributes like height, weight, and technical skills |
| **Match** | A single soccer game between two teams, lasting 90 minutes | Each record in the matches table represents one professional game |
| **Home Team** | The team playing in their own stadium | Home advantage is captured by the `is_home` feature |
| **Away Team** | The team traveling to play at the opponent's stadium | Away players may face additional fatigue and travel-related performance impacts |
| **Performance Score** | A calculated metric (0-100) measuring player effectiveness | Based on weighted combination of rating, finishing, dribbling, and passing |
| **Rating (Overall Rating)** | A player's general skill level from 0-100 (FIFA game rating) | Core attribute used to calculate performance_score |
| **Finishing** | A player's ability to score goals when given opportunities | Technical skill attribute (0-100) from FIFA ratings |
| **Dribbling** | A player's ability to control the ball while moving past opponents | Technical skill attribute that influences performance_score |
| **Passing** | A player's accuracy and effectiveness when distributing the ball to teammates | Combined from short_passing and long_passing attributes |
| **Rolling Average** | Average of a player's last 3 performance scores | Captures recent form; the strongest predictor of decline |
| **Decline Flag** | Binary indicator (1 = performance decreased, 0 = no decline) | Target variable for prediction |
| **Goal Difference** | Home team goals minus away team goals | Positive = team won, negative = team lost |
| **Home Advantage** | The statistical tendency for home teams to perform better | Captured by `is_home` feature; home teams win approximately 55% of matches |
| **Form** | A player's recent performance trend | Measured by rolling_avg_3; hot/cold streaks impact decline likelihood |
| **Career Volatility** | Standard deviation of a player's historical performance scores | High volatility indicates inconsistent players more prone to declines |

### Paragraph
Paragraph explaining the domain the project lives in

### Background Reading
| Title | Brief Description | Link |
|-------|-------------------|------|
| Can AI Predict Player Performance in New Team Environments? | A blog post introducing a study that uses "Large Event Models" to evaluate a player's potential impact on a soccer team. | [Link](https://github.com/vcx4ka/ds4320-project1/blob/main/background-reading/Can-AI-Predict-Player-Performance-in-New-Team-Environments.pdf) |
| Key Performance Indicators Predictive of Success in Soccer: A Comprehensive Analysis of the Greek Soccer League | This study analyzes all matches from the 2020-2021 Greek Football League season and identifies a set of factors that influence match outcomes. This analysis explores what features a "winning team" possesses. | [Link](https://github.com/vcx4ka/ds4320-project1/blob/main/background-reading/Performance-Indicators-Predictive-of-Success.pdf) |
| Forecasting extremes of football players’ performance in matches | This study evaluates models that forecast extreme performance metrics in soccer matches. These models utilize data from team training sessions to accurately predict real match performance. | [Link](https://github.com/vcx4ka/ds4320-project1/blob/main/background-reading/Forecasting-Extremes-of-Football-Players-Performance-in-Matches.pdf) |
| Predict soccer match outcome based on player performance | This article builds a model to predict the outcome of a match based on the performance of individual players, rather than by historical results. It aims to quantify the unpredictability of match outcomes. | [Link](https://github.com/vcx4ka/ds4320-project1/blob/main/background-reading/Predict-soccer-match-outcome-based-on-player-performance.pdf) |
| From Practice To Performance: Predicting Soccer Match Outcomes from Training Data | This study analyzes training session data from soccer players to predict match performance. It examines the relationship between training metrics and match outcomes. | [Link](https://github.com/vcx4ka/ds4320-project1/blob/main/background-reading/From-Practice-To-Performance-Predicting-Soccer-Match-Outcomes-from-Training-Data.pdf) |






<br>
<br>

## Data Creation

**Item 1. Paragraph or two explaining the raw data acquisition process (provenance).**

The raw data was sourced from the MovieLens dataset, a public dataset maintained by the GroupLens Research Lab

It contains user-generated movie ratings

The data was accessed using the download.py script through publicly available csv files. Once imported, the data was cleaned (clean.py), transformed into document format (transform.py) and after it was fully processed, loaded into MongoDB (mongo_load.py). This relational-structured data was transformed into documents to better support recommendation analysis.

The dataset used in this project is derived from the MovieLens dataset, a widely used public dataset collected and maintained by the GroupLens Research Lab at the University of Minnesota. The dataset contains user-generated movie ratings along with metadata about movies such as titles and genres.

The data was accessed via publicly available CSV files and imported into a Python environment for preprocessing. During preprocessing, the data was cleaned by removing missing values, standardizing formats, and restructuring the data into a document-oriented format suitable for MongoDB. Specifically, relational tables (users, movies, ratings) were transformed into nested documents to better support recommendation queries and analysis pipelines.

<br>

**Item 2. Code Table showing the code used to create the data, one row per file, with a brief description and link to code in repo**

| Code File | Brief Description | Link |
|-----------|-------------------|------|
| download.py | Extracts Player, Match, and Player_Attributes tables from SQLite to CSV | [Link](https://github.com/vcx4ka/ds4320-project2/blob/main/data-creation-code/download.py) |
| clean.py | Creates clean players table with height/weight handling | [Link](https://github.com/vcx4ka/ds4320-project2/blob/main/data-creation-code/clean.py) |
| transform.py | Creates clean matches table with team IDs and scores | [Link](https://github.com/vcx4ka/ds4320-project2/blob/main/data-creation-code/transform.py) |
| mongo_load.py | Reshapes wide match data into long format and joins with player attributes | [Link](https://github.com/vcx4ka/ds4320-project2/blob/main/data-creation-code/mongo_load.py) |
| solution-pipeline.ipynb | Complete ML pipeline with model training and evaluation | [Link](https://github.com/vcx4ka/ds4320-project2/blob/main/solution-pipeline.ipynb) |

<br>

**Item 3. Bias Identification description of how bias could be/was introduced in the data collection process**

Bias in this dataset arises primarily from the user population and rating behavior. Since the MovieLens dataset is composed of voluntary user ratings, it overrepresents users who are more active and engaged with movie rating platforms. Additionally, the dataset may be biased toward popular or mainstream films, as these are more likely to receive a large number of ratings. There is also potential demographic bias, as the dataset may not accurately reflect the diversity of the general population.

<br>

**Item 4. Bias Mitigation description of how biases can be handled/quantified/accounted for in analysis**

To mitigate these biases, several strategies can be applied during analysis. For example, normalization techniques can be used to reduce the influence of highly active users. Weighting methods can also be applied to account for imbalances in movie popularity. Additionally, recommendations can incorporate diversity constraints to avoid over-recommending popular content and instead promote a broader range of movies.

<br>

**Item 5. Rationale for critical decisions especially judgement calls, and places that can introduce/mitigate uncertainty**

Several key decisions were made during the data preparation process. First, the data was restructured from a relational format into a document model to better support flexible queries and nested relationships between users and movies. This introduced some uncertainty in how to best structure documents, particularly whether to embed ratings within user documents or movie documents.

A hybrid approach was considered, but embedding ratings within user documents was ultimately chosen to optimize for user-centric recommendation queries. Another important decision involved handling missing or sparse data, where thresholds were applied to filter out users or movies with too few ratings. These decisions may introduce bias by excluding less active users or niche content, but they improve the reliability of downstream analysis.






## Metadata

**Item 1. Implicit Schema Guidelines for document structure**

The database follows a user-centric document model in which each document represents a single user and their associated movie rating activity. Each user document contains a unique user identifier and an embedded array of rating objects.

Each rating object includes a movie identifier, rating value, timestamp, and selected movie metadata such as title and genres. Embedding ratings within user documents allows for efficient retrieval of user preferences and supports recommendation queries based on past behavior.

While the schema is implicit and does not enforce strict structure, guidelines are established to ensure consistency across documents. All user documents should contain a user_id field and a ratings array. Each rating object should consistently include movie_id, rating, and timestamp, while optional fields such as genres may be included when available. This balance allows flexibility while maintaining enough structure for reliable querying and analysis.

<br>

**Item 2. Data Table listing all of the tables in the dataset, one line per table, brief description and link to csv file**

| Feature | Description |
|---------|-------------|
| Total Users | ~600 (MovieLens 100K dataset |
| Total Movies | ~9,000 |
| Total Ratings | ~100,000 |
| Avg Ratings per User | ~165 |
| Data Structure | User-centric documents with embedded ratings |

<br>

**Item 3. Data Dictionary Table with one row per feature in the data set containing: name, data type, description, example**

| Field Name          | Data Type      | Description                                     | Example                 |
| ------------------- | -------------- | ----------------------------------------------- | ----------------------- |
| `user_id`           | Integer        | Unique identifier for each user                 | 123                     |
| `ratings`           | Array          | List of rating objects for the user             | […]                     |
| `ratings.movie_id`  | Integer        | Unique identifier for a movie                   | 50                      |
| `ratings.title`     | String         | Movie title                                     | "Toy Story (1995)"      |
| `ratings.genres`    | Array (String) | List of genres for the movie                    | ["Animation", "Comedy"] |
| `ratings.rating`    | Float          | User’s rating of the movie (typically 0.5–5.0)  | 4.5                     |
| `ratings.timestamp` | Integer        | Time when rating was submitted (Unix timestamp) | 964982703               |


<br>

**Item 4. Data Dictionary quantification of uncertainty for numerical features**

Uncertainty in the dataset primarily arises from the subjective nature of user ratings and variability in user behavior. Ratings may not consistently reflect true user preferences, as they can be influenced by mood, context, or temporal factors. Additionally, sparsity in the dataset introduces uncertainty, as many users rate only a small subset of available movies.

This uncertainty can be partially quantified by examining variance in user ratings, distribution of ratings per movie, and confidence intervals for average ratings. Movies with a small number of ratings exhibit higher uncertainty in their average scores compared to widely rated movies. Similarly, users with fewer ratings provide less reliable signals for preference modeling.
