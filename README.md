# tweet-sentiment-pyspark
Tweets Sentiment Classification Using PySpark's NaiveBayes.
Sentiment Analysis on tweets Dataset using NaiveBayes binary classification Model and Bag of words technique to make feature vectors to feed NaiveBayes.
<br>
tweets are classified as positive=1, negative=0.
<br>
Dataset contains contains 1,578,627 classified tweets, each row is marked as 1 for positive sentiment and 0 for negative sentiment. Dataset can be downloaded from the this <a href="http://thinknook.com/wp-content/uploads/2012/09/Sentiment-Analysis-Dataset.zip">Tweets Dataset </a>.
<br>
With this script i acheived upto 60% Accuracy on unlabeled test dataset.
<br>
I took about 20 min. max on my i5-2.3u, 4gb ram machine to train on 90% of the dataset and test on remaining 10%.

<br>
<b>Dependencies</b>
<br>
<ul>
<li>
Apache Spark and pyspark
</li>
<li>
Pandas
</li>

<li>
Python 2.7
</li>
</ul>
<b>TODO</b>
<br>
One can further improve accuracy by Lemmatisation of dataset and using word2vec technique. On which i am still working on. And you can also try different classification models like Random Forest, SVM or Even try Deep Learning, CNN, RNN.
