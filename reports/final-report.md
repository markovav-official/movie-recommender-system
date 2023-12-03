# Introduction

My project centered on applying machine learning techniques to analyze and predict movie ratings, a task integral to the development of sophisticated recommendation systems and targeted marketing strategies. This work not only aimed at understanding complex user behaviors and preferences but also at enhancing the user experience in the entertainment industry through personalized content curation. By accurately predicting movie ratings, the project contributes to improved customer satisfaction and retention, showcasing the impact of data-driven insights in predictive analytics and their broad applicability across various sectors.

# Data Analysis

At the outset of my project, I embarked on an analysis of Movie Lens 100K dataset. The dataset includes:

* **Users Data:** demographics of the users, including age, gender, occupation, and zip code.
* **Ratings Data:** ratings that users assigned to different movies.
* **Movies Data:** movies titles, genres and release date.

My analysis involved importing these datasets, exploring their structures, and understanding the relationships between different data points. I paid close attention to the characteristics and patterns within the data, which were pivotal in guiding the direction of my model development and evaluation strategies.

# Model Implementation

In my project, I implemented two distinct models to tackle the challenge of predicting movie ratings: the Nearest Neighbors approach and the Light Graph Convolutional Network (LightGCN).

### Nearest Neighbors Approach

The first approach I employed was the Nearest Neighbors model. This method is based on the concept of similarity, where ratings are predicted based on the preferences of similar users. I chose this approach due to its intuitive nature and ease of implementation. By analyzing user data, I was able to identify patterns and group users with similar tastes, thereby predicting movie ratings based on the ratings of these 'neighbor' users. This model is particularly effective in capturing local patterns in the data.

### Light Graph Convolutional Network (LightGCN)

The second approach I explored was the Light Graph Convolutional Network, a more sophisticated model that leverages the power of graph neural networks. LightGCN is particularly suited for recommendation systems as it efficiently captures the complex relationships between users and items (movies, in this case) in a graph structure. By implementing LightGCN, I was able to model the interactions between users and movies, allowing for a deeper understanding of the latent factors influencing user preferences and ratings. This approach is noted for its ability to handle sparse data and extract meaningful insights from the connections within the graph.

# Model Advantages and Disadvantages

In my project, I critically evaluated the strengths and weaknesses of both the Nearest Neighbors and Light Graph Convolutional Network (LightGCN) models. Understanding these aspects was key to optimizing their performance and applicability.

### Nearest Neighbors Approach

**Advantages:**

* **Simplicity:** The Nearest Neighbors model is straightforward to understand and implement. Its intuitive nature makes it easy to explain and justify the predictions it makes.
* **Efficiency in Identifying Local Patterns:** This approach is effective in capturing local patterns and similarities among users, making it suitable for scenarios where close user interactions are more indicative of preferences.

**Disadvantages:**

* **Scalability Issues:** As the dataset grows, the computational complexity increases significantly, making it less efficient for large-scale applications.
* **Sensitivity to Sparse Data:** The model can struggle with sparse datasets where users have rated only a few movies, leading to less reliable predictions.

### Light Graph Convolutional Network (LightGCN)

**Advantages:**

* **Handling Sparse Data:** LightGCN excels in environments with sparse data, effectively capturing the underlying structure and relationships in the user-item graph.
* **Deep Insights:** This model provides deeper insights into user preferences and behaviors by leveraging graph neural networks, which are adept at extracting latent factors in the data.

**Disadvantages:**

* **Complexity:** The implementation and tuning of LightGCN can be more complex compared to simpler models, requiring a deeper understanding of graph neural networks.
* **Computational Resources:** Due to its sophisticated nature, LightGCN typically demands more computational resources, which can be a limitation for resource-constrained projects.

# Training Process

The training processes for the Nearest Neighbors and LightGCN models in my project involved distinct steps tailored to each model's unique characteristics and requirements.

### Nearest Neighbors Training Process

For the Nearest Neighbors model, the training process was relatively straightforward but crucial for achieving accurate predictions. My focus was on:

* **Data Preparation:** I ensured that the data was clean and well-structured, with users and their corresponding ratings accurately represented.
* **Feature Selection:** Identifying and selecting the most relevant features was a key step. This included user demographics, movie genres, and previous ratings.
* **Similarity Metrics:** I experimented with various similarity metrics like cosine similarity and Euclidean distance to find the most effective way to identify 'neighbor' users.

### LightGCN Training Process

The training process for LightGCN was more complex, involving several advanced steps:

* **Graph Construction:** An integral part of the process was constructing a user-movie graph, accurately representing the interactions between users and movies.
* **Feature Engineering:** Unlike traditional models, LightGCN required minimal feature engineering, as the model primarily works with the structure of the graph itself.
* **Layer Tuning:** Determining the optimal number of layers in the LightGCN was critical to capture the right level of user-item interactions without overfitting.
* **Optimization:** I used advanced optimization techniques and loss functions tailored for graph neural networks to effectively train the LightGCN model.

# Evaluation

In the evaluation of both the Nearest Neighbors and Light Graph Convolutional Network (LightGCN) models, I focused on using Precision@10 (P@10) and Recall@10 (R@10) as the primary metrics. These metrics were pivotal in assessing the effectiveness of the models in predicting the top 10 movie recommendations for users.

### Evaluation of Nearest Neighbors Model

For the Nearest Neighbors model, I conducted evaluations with different rating thresholds to understand the model's performance in various scenarios:

* **With a rating threshold of 3:**
  * **Precision@10:** 0.84
  * **Recall@10:** 0.24
* **With a rating threshold of 4:**
  * **Precision@10:** 0.61
  * **Recall@10:** 0.21

These results indicated that the model was more precise and selective at a lower threshold, with a notable performance in identifying relevant recommendations.

### Evaluation of LightGCN Model

For the LightGCN model, I also used a rating threshold of 3 for consistency in comparison:

* **Precision@10:** 0.30
* **Recall@10:** 0.23

Although the precision of the LightGCN model was lower compared to the Nearest Neighbors approach, it still demonstrated a significant ability to capture relevant recommendations, especially considering the complexity of user-item interactions in the graph structure.

Through these evaluations, I gained valuable insights into the strengths and limitations of each model, allowing me to understand their applicability in different contexts of recommendation systems.

# Results

The implementation of the Nearest Neighbors and Light Graph Convolutional Network (LightGCN) models yielded insightful results:

* **Nearest Neighbors Model:** Demonstrated high precision with a Precision@10 of 0.8377 at a rating threshold of 3, indicating strong relevancy in its recommendations. However, its recall was lower, suggesting a more selective approach to recommendations.
* **LightGCN Model:** Showed a balanced performance with a Precision@10 of 0.2996 and Recall@10 of 0.2324 at the same threshold, highlighting its capability to handle complex user-item interactions despite a lower precision compared to the Nearest Neighbors model.

These results underscore the effectiveness of both models in different aspects of recommendation systems, with Nearest Neighbors excelling in precision and LightGCN in capturing complex relationships.
