from __future__ import print_function

import re

def clean_tweet(row):
    """
    description:Clean right,left spaces from the tweet, links and other characters
    and return a tuple(sentiment, [list_of_words])
    """
    split_row=row.split(",")
    sentiment=split_row[1]
    tweet=split_row[3]
    tweet.strip()
    
    #remove url from the tweet
    tweet=re.sub(r"http\S+", "", tweet)
    
    #remove other characters from the tweet
    tweet=re.sub("[^a-zA-z]", " ", tweet)
    
    tweet=tweet.lower()
    #convert tweet in to list of words
    words_list=tweet.split()
    
    return(sentiment, words_list)
    

if __name__=="__main__":
    from pyspark.sql import SparkSession

    from pyspark import SparkContext
    sc=SparkContext(appName="TweetSentiApp")    
    
    spark = SparkSession.builder.appName("TweetSentiApp").getOrCreate()

    #to start a spark context
    
    
    #read data file
    data=sc.textFile("Sentiment_Analysis_Dataset.csv")
    head=data.first()
    data=data.filter(lambda r: r!=head)
    dataset=data.map(lambda r: clean_tweet(r))
    
    #convert rdd into sql dataframe to remove stop words, make ngram model and turn review into a feature
    dataframe=spark.createDataFrame(dataset, ["sentiment", "tweet"])

    #now remove stopwords from the review(list of words)    
    from pyspark.ml.feature import StopWordsRemover
    
    remover=StopWordsRemover(inputCol="tweet", outputCol="filtered")
    filtered_df=remover.transform(dataframe)
    
    #now make 2-gram model
    from pyspark.ml.feature import NGram

    ngram=NGram(n=2, inputCol="filtered", outputCol="2gram")
    gram_df=ngram.transform(filtered_df)
    
    #now make term frequency vectors out of data frame to feed machine
    from pyspark.ml.feature import HashingTF,IDF
    hashingtf=HashingTF(inputCol="2gram", outputCol="tf", numFeatures=20000)
    tf_df=hashingtf.transform(gram_df)
    
    #tf-idf
    idf=IDF(inputCol="tf", outputCol="idftf")
    idfModel=idf.fit(tf_df)
    idf_df=idfModel.transform(tf_df)
    
    #convert dataframe t rdd, to make a LabeledPoint tuple(label, feature, vector) for machine
    tf_rdd=tf_df.rdd
    
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.linalg import  Vectors as MLLibVectors
    
    #we also need to convert ml.sparsevector mllib.sparse vector, because naive bayes only accepts mllib.sparsevector type
    train_dataset=tf_rdd.map(lambda x: LabeledPoint(float(x.sentiment), MLLibVectors.fromML(x.tf)))
    
    #split dataset into train, test
    
    train, test=train_dataset.randomSplit([0.9, 0.1], seed=11L)
    
    print(train.first())
    print(test.first())
    
    #create Model
    #now train and save the model
    from pyspark.mllib.classification import NaiveBayes
    import shutil
    
    #training
    print("************************TRAINIG*******************************")
    model=NaiveBayes.train(train, 1.0)
    print("*****************************TRAINING COMPLETE")
#    
#    #saving the model
    output_dir = '/home/sohaib/Documents/myNaiveBayesModel_Tweet3'
    shutil.rmtree(output_dir, ignore_errors=True)
    model.save(sc,output_dir)
    
    #testing on test data
    print("************************TESTING***********************************")
    predictionAndLabel=test.map(lambda x: (x.label, model.predict(x.features)))
    
    accuracy=1.0*predictionAndLabel.filter(lambda x: x[0]==x[1]).count()/test.count()
    
    print("Model Accuracy is ", accuracy)
    print("*****************TESTING COMPLETED*****************************")


    
    
    
    sc.stop()