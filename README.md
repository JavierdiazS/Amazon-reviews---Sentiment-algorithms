# Amazon reviews - Sentiment algorithms

- [Summary](#summary)
- [Results](#results)
- [Reference](#reference)

## Summary

This is a sentiment analysis model with NLP that classifies Amazon books reviews on a scale of 1 to 5, with 1 being very bad and 5 being very good.

Three tests are performed with different types of vectorizations, feature selection, and classification models to find the best discrete numerical classification algorithm.

<details>
    <summary>❌ Proof 1:</summary> 
  
    Data Preprocessing
      * Loads the book review data from a text file and splits it into reviews and labels.
      * Removes punctuation and stop words from the reviews.
      * Converts the reviews to TF-IDF vectors.

    Feature Selection
      * Uses the chi-squared test to select the most important features from the TF-IDF vectors.

    Model Selection and Training
      * Trains 5 different machine learning models to predict the sentiment of the book reviews: LogisticRegression(),       RandomForestClassifier(), BernoulliNB(), ComplementNB() and MultinomialNB().
      * Evaluates the performance of each model using the accuracy metric.

    Confusion Matrix
      * Generates a confusion matrix to visualize the performance of the Bernoulli Naive Bayes model, which had the best score.
</details>

<details>
    <summary>❌ Proof 2:</summary> 
  
    Data Preprocessing
      * Bag-of-Words (BoW) Representation: The code constructs a bag-of-words representation of the book reviews.
      * Filtering: To reduce the dimensionality of the BoW representation, the code filters out words that appear less than 100 times in the entire dataset.
      * Vectorization: The code converts the filtered BoW representation into a numerical format suitable for machine learning algorithms.

    Model Selection and Training
      * Trains 4 different machine learning models to predict the sentiment of the book reviews: RandomForestClassifier(), BernoulliNB(), ComplementNB() and MultinomialNB().
      * Evaluates the performance of each model using the accuracy metric.

    Confusion Matrix
      * Generates a confusion matrix to visualize the performance of the Random Forest Classifier model, which had the best score.
</details>

<details>
    <summary>✔️ Proof 3:</summary> 
  
    Data Preprocessing
      * Bag-of-Words (BoW) Representation: The code constructs a bag-of-words representation of the book reviews.
      * Stop Word Removal: To reduce noise and focus on more informative words, the code removes stop words (common words that don't add much meaning to the text) using the NLTK Natural Language Toolkit.
      * Filtering: To reduce the dimensionality of the BoW representation, the code filters out infrequent words that appear less than 50 times in the entire dataset.
      * Bigram Extraction: The code extracts bigrams, which are pairs of consecutive words, to capture additional semantic information from the reviews. It identifies the 250 most frequent bigrams to focus on the most informative ones.
      * Vectorization: The code converts the filtered BoW representation and the extracted bigrams into numerical vectors suitable for machine learning algorithms.

    Model Selection and Training
      * Trains 4 different machine learning models to predict the sentiment of the book reviews: RandomForestClassifier(), BernoulliNB(), ComplementNB() and MultinomialNB().
      * Evaluates the performance of each model using the accuracy metric.

    Confusion Matrix
      * Generates a confusion matrix to visualize the performance of the Random Forest Classifier model, which had the best score: 95%.
</details>

## Results 

Score for RandomForestClassifier = 0.9545867393278837

Confusion Matrix:

![image](https://github.com/JavierdiazS/Amazon-reviews---Sentiment-algorithms/assets/75210642/596b0c11-d8c7-4796-8813-5f787c808669)

## Reference

* Blitzer, J., Dredze, M., & Pereira, F. (2007). Biographies, Bollywood, Boom-boxes and Blenders: Domain Adaptation for Sentiment Classification. In Proceedings of the 45th Annual Meeting of the Association for Computational Linguistics (ACL), (pp. 440-448). Association for Computational Linguistics.

  The processed_stars.tar.gz dataset was downloaded from the following website:
  https://www.cs.jhu.edu/~mdredze/datasets/sentiment/

## License

It is released under the MIT license. See the [LICENSE](/LICENSE) file for details.
 
