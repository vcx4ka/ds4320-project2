# DS 4320 Project 2: Personalized Movie Recommendation System

### Executive Summary
This repository contains the complete data pipeline and analysis for a movie recommendation system using collaborative filtering. Using the MovieLens 1M dataset, I built a document database in MongoDB, transformed relational data into user-centric documents, and implemented a k-Nearest Neighbors collaborative filtering model to predict user preferences. The model achieves 7.1% better mean absolute error (0.871 vs 0.938) compared to a baseline that constantly predicted the mean (a rating of 3.58 stars). These results demonstrated that preference-based recommendations outperform a one-size-fits-all approach to predictions.

| Spec | Value |
| --- | --- |
| Name | Crystal McEnhimer |
| NetID | vcx4ka |
| DOI | [![DOI](https://zenodo.org/badge/1189840726.svg)](https://doi.org/10.5281/zenodo.19378548) |
| Press Release | [PRESS-RELEASE.md](PRESS-RELEASE.md) |
| Pipeline | [solution-pipeline.ipynb](solution-pipeline.ipynb) [solution-pipeline.md](solution-pipeline.md) |
| License State | [MIT License](LICENSE.md) |

---
<br>

<br>

## Problem Definition
### General and Specific Problem
* **General Problem:** Recommending content on streaming platforms.
* **Specific Problem:** Using historical rating data from movie viewers, we aim to predict which movies a user will enjoy most based on their past ratings and the ratings/behavior of users with similar preferences or tastes.

### Rationale
While content-based recommendation systems rely on movie metadata (genres, actors, directors), collaborative filtering offers a different approach: leveraging the collective experiences of users with similar taste. This method is applicable in many circumstances, especially in the event a movie lacks metadata. It also has the added benefit of discovering unexpected connections, where users who like one movie tend to like another unrelated movie. By using this dataset to compare user profiles and their preferences, we can construct a reliable and logical approach to content recommendation.

### Motivation
Streaming platforms lose money annually to churn caused by poor content discovery. Viewers waste excess amounts of time scrolling to find something they may like instead of watching. A better recommendation system directly impacts user retention, as users are more likely to stick with their platform if they can find content they love quickly. A better system would also allow users to discover more niche content, improving return on investment for content licensing. Additionally, user satisfaction would generally improve, as users will spend less time frustrated, and more time enjoying the content on the platform. This project demonstrates how publicly available rating data can be transformed into personalized recommendations using document databases and data science techniques.

### Press Release Headline and Link
[**New Machine Learning Model Creates Personalized Movie Recommendations Based On User Reviews**](PRESS-RELEASE.md)

<br>
<br>

## Domain Exposition

### Terminology
| Term | Definition |
|------|------------|
| **Streaming Service** | On-demand video platform (Netflix, Hulu, Disney+, Amazon Prime) |
| **Content Library** | The complete collection of movies and shows available on a platform |
| **Viewer Retention** | The ability to keep subscribers engaged and prevent cancellations |
| **Churn Rate** | Percentage of subscribers who cancel their service each month |
| **Binge-Worthy** | Content so engaging that viewers watch multiple episodes/movies in one sitting |
| **Backlog** | The accumulation of unwatched content saved to a user's list |
| **Niche Content** | Movies or shows appealing to a specific, often smaller audience segment |
| **Mainstream Blockbuster** | High-budget movies designed for mass audience appeal |
| **Taste Profile** | A viewer's unique pattern of preferences across genres, directors, actors, etc. |
| **Discovery** | The process of finding new content a viewer hasn't seen before |

### Paragraph
This project operates within the digital streaming and home entertainment domain. This industry has developed rapidly over the last few decades and has fundamentally transformed how audiences consume movies. Streaming platforms like Netflix, Hulu, Disney+, Amazon Prime, and Max have replaced traditional movie watching methods (theaters, DVD rentals, cable TV) as the primary way viewers access films. The streaming market is increasingly saturated and competitive lately, with many people subscribing to multiple streaming services simultaneously.

Streaming platforms offer massive content libraries with a wide variety of titles, but viewers are plagued with a paradox of choice. With a plethora of options, users have a hard time selecting a show to watch. Often times people scroll for several minutes before deciding on something, give up without watching anything, or simply rewatch familiar content. Poor content discovery directly limits customer engagement, impacting a platform's success. Platforms that solve this problem gain a competitive advantage, as good recommendations increase sales and viewership.


### Background Reading
| Title | Brief Description | Link |
|-------|-------------------|------|
| The ‘Streaming Wars:’ 5 Big Questions on Netflix, Disney Plus and the Future of Online Video | Darden professor analyzes the competitive landscape amongst streaming services and content libraries. | [Link](background-reading/The-Streaming-Wars-Darden.pdf) |
| Streaming services are causing a 'mass of customer confusion' and decision paralysis | Explores the struggle of viewers to both differentiate between streaming services and find content that they enjoy, leading to a paralysis of choice. |[Link](background-reading/Mass-Decision-Paralysis.pdf) |
| How Netflix's Recommendations System Works | Official Netflix documentation explaining how their personalization algorithms work for non-technical audiences | [Link](background-reading/Netflix-Recommendation-System.pdf) |
| Because You Watched: How Do Streaming Services’ Recommender Systems Influence Aesthetic Choice? | Examines how algorithmic recommendations shape viewer preferences and considers potential biases embedded in recommendation systems. | [Link](background-reading/Because-You-Watched.pdf) |
| Streaming Wars: The Hidden Battle of Digital Media Licensing | Analyzes how digital media licensing agreements control content distribution, and the relationship between licensing costs and which content gets recommended. | [Link](background-reading/Digital-Media-Licensing.pdf) |



<br>
<br>

## Data Creation

**Item 1. Paragraph or two explaining the raw data acquisition process (provenance).**

The raw data used in this project is the MovieLens 1M dataset, a public dataset maintained by the GroupLens Research Lab.The dataset contains 6,040 volunteer users and their demographic information, nearly 4,000 movies, and over a million ratings on 1-5 star scale. The data was originally collected through the MovieLens website.

The raw CSV files were downloaded via an automated python script titled 01_download.py. It was then transformed from relational format into a document-oriented format suitable for MongoDB in the 02_mongo_load.py. The transformation process nested each user's ratings as an embedded array within their user document. In the same script, that data was uploaded to a collection in MongoDB Atlas. After the upload, a final script (03_validate.py) validates the data, checking for missing values, data quality issues, and providing summary statistics.

<br>

**Item 2. Code Table showing the code used to create the data, one row per file, with a brief description and link to code in repo**

| Code File | Brief Description | Link |
|-----------|-------------------|------|
| 01_download.py | Downloads MovieLens 1M dataset from GroupLens website | [Link](data-creation-code/01_download.py) |
| 02_mongo_load.py | Transforms MovieLens dataset from CSV files into a collection of documents, then loads them into MongoDB. | [Link](data-creation-code/02_mongo_load.py) |
| 03_validate.py | Validates the data quality after loading it into MongoDB, and provides summary statistics. | [Link](data-creation-code/03_validate.py) |
| solution-pipeline.ipynb | Complete ML pipeline with model training and evaluation | [Link](solution-pipeline.ipynb) |


<br>

**Item 3. Rationale for critical decisions especially judgement calls, and places that can introduce/mitigate uncertainty**

Several key decisions were made during the data preparation process. First, the data was restructured from a relational format into a document model to better support flexible queries and nested relationships between users and movies. This introduced some uncertainty in how to best structure documents, particularly whether to embed ratings within user documents or movie documents.

Embedding ratings within user documents was ultimately chosen as it best aligns with our goal to make recommendations tailored to specific users. This setup allowed optimization for user-centric recommendation queries. Another important decision involved handling missing or sparse data, where thresholds were applied to filter out users or movies with too few ratings. These decisions may introduce bias by excluding less active users or niche content, but they improve the reliability of downstream analysis.

<br>

**Item 4. Bias Identification description of how bias could be/was introduced in the data collection process**

Bias in this dataset arises primarily from the user population and rating behavior. Since the MovieLens dataset is composed of voluntary user ratings, it overrepresents users who are more active and engaged with movie rating platforms. Additionally, the dataset may be biased toward popular or mainstream films, as these are more likely to receive a large number of ratings. There is also potential demographic bias, as the dataset has significantly more male than female participants (~75% male, ~25% female). Additionally, there may be temporal bias in this data as the dataset is from 2000-2003.

<br>

**Item 5. Bias Mitigation description of how biases can be handled/quantified/accounted for in analysis**

To mitigate these biases, several strategies can be applied during analysis. For example, normalization techniques can be used to reduce the influence of highly active users. Weighting methods can also be applied to account for imbalances in movie popularity. Additionally, recommendations can incorporate diversity constraints to avoid over-recommending popular content and instead promote a broader range of movies.





## Metadata

**Item 1. Implicit Schema Guidelines for document structure**

The database follows a user-centric document model in which each document represents a single user and their associated movie rating activity. Each user document contains a unique user identifier and an embedded array of rating objects. Each rating object includes a movie identifier, rating value, timestamp, and selected movie metadata such as title and genres. Embedding ratings within user documents allows for efficient retrieval of user preferences and supports recommendation queries based on past behavior.

While the schema is implicit and does not enforce strict structure, guidelines are established to ensure consistency across documents. All user documents should contain a user_id field and a ratings array. Each rating object should consistently include movie_id, rating, and timestamp, while optional fields such as genres could be included when available. This balance allows flexibility while maintaining enough structure for reliable querying and analysis.

<br>

**Item 2. Data Summary Summary of Database contents**

| Feature | Description |
|---------|-------------|
| Total Users | 6,040 |
| Total Movies | 3,706 |
| Total Ratings | 1,000,209 |
| Avg Ratings per User | ~165 |
| Rating Range | 1.0 - 5.0 |
| Sparsity | 95.5% (% of possible user-movie pairs without ratings) |
| Data Structure | User-centric documents with embedded ratings and demographic information |

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

Uncertainty in the dataset primarily arises from the subjective nature of user ratings and variability in user behavior. Ratings may not consistently reflect true user preferences, as they can be influenced by mood, context, or temporal factors. Additionally, sparsity in the dataset introduces uncertainty, as many users rate only a small subset of available movies. Sparsity in this dataset is 95.5%, which means only 4.5% of possible user-movie pairs have ratings.

This uncertainty can be partially quantified by examining variance in user ratings, distribution of ratings per movie, and confidence intervals for average ratings. Movies with a small number of ratings exhibit higher uncertainty in their average scores compared to widely rated movies. Similarly, users with fewer ratings provide less reliable signals for preference modeling.

Uncertainty can also be quantified by performance metrics. The simple baseline model that simply predicted the global mean (3.58 stars) achieved an RMSE of 1.12 stars, meaning the typical prediction errors are +/- 1.12 stars with large errors penalized more heavily. The collaborative filtering model achieves a MAE of 0.871 stars compared to the baseline MAE of 0.938 stars, which is a 7.1% improvement in average prediction accuracy. RMSE remains similar (1.126 vs 1.121), indicating the model occasionally makes larger errors than baseline. This indicates that the model tends to have larger errors for movies with high rating variance.