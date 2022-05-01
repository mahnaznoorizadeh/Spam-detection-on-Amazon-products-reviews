import findspark
findspark.init()
import pyspark as ps
import warnings
from pyspark.sql import SQLContext

from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator


try:
    # create SparkContext on all CPUs available: in my case I have 4 CPUs on my laptop
    sc = ps.SparkContext('local[4]')
    sqlContext = SQLContext(sc)
    print("Just created a SparkContext")
except ValueError:
    warnings.warn("SparkContext already exists in this scope")

sc.master
df = sqlContext.read.format('org.apache.spark.sql.json').load('Cell_Phones_and_Accessories.json')
#org.apache.spark.sql.json

type(df)
df.show(5)

df = df.dropna()
df.count()
(train_set, val_set, test_set) = df.randomSplit([0.90, 0.05, 0.05], seed = 2000)

tokenizer = Tokenizer(inputCol="reviewText", outputCol="words")
hashtf = HashingTF(numFeatures=2**16, inputCol="words", outputCol='tf')
idf = IDF(inputCol='tf', outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
label_stringIdx = StringIndexer(inputCol = "class", outputCol = "label")
pipeline = Pipeline(stages=[tokenizer, hashtf, idf, label_stringIdx])

pipelineFit = pipeline.fit(train_set)
train_df = pipelineFit.transform(train_set)
val_df = pipelineFit.transform(val_set)
train_df.show(5)

lr = LogisticRegression(maxIter=100)
lrModel = lr.fit(train_df)
predictions = lrModel.transform(val_df)


evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
evaluator.evaluate(predictions)

evaluator.getMetricName()
accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(val_set.count())
print("-----------------------------------")
print(accuracy)
