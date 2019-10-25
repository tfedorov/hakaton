package com.tfedorov.shortdesc

import com.tfedorov.shortdesc.CalculateApp.intersectedDF
import com.tfedorov.shortdesc.PrepareDataApp.Normal
import org.apache.spark.ml.feature.{StopWordsRemover, Tokenizer}
import org.apache.spark.sql.{Dataset, SparkSession}

object AnotherApp extends App {

  println("start AnotherApp")
  // Input data: Each row is a bag of words from a sentence or document.
  val spark = SparkSession
    .builder()
    .appName("Spark AnotherApp permid")
    .master("local[*]")
    .getOrCreate()

  import spark.implicits._

  case class Docs(fileName: String, content: String)

  //file:///c:/work/data/hakaton/decriptions-data/tmpTrain/
  val testDocsDS = spark.sparkContext.wholeTextFiles("c:/work/data/hakaton/decriptions-data/sigTestDataFilesInput/*")
    .map(t => Docs(t._1.substring("file:/c:/work/data/hakaton/decriptions-data/sigTestDataFilesInput/".length), t._2)).toDF().as[Docs]

  //C:\work\workspace\hakaton\src\main\resources\projectData\taggedTestSet.csv
  case class TaggedFile(docId: String, alias: String, label: String, permid: String)

  val taggedDS = spark.sqlContext.read
    .format("com.databricks.spark.csv")
    .option("header", "true")
    .option("mode", "DROPMALFORMED")
    .load("C:/work/workspace/hakaton/src/main/resources/projectData/taggedTestSet.csv").as[TaggedFile]

  val joinedTextDF = testDocsDS.join(taggedDS, $"fileName" === $"docId", "inner").select("fileName", "content", "alias", "label", "permid")

  val tokenizerText = new Tokenizer().setInputCol("content").setOutputCol("textTokens")
  val removerText = new StopWordsRemover().setInputCol("textTokens").setOutputCol("textFilteredTokens")

  val joinedTokens = removerText.transform(tokenizerText.transform(joinedTextDF))

  val tfIdfDS: Dataset[Normal] = spark.sqlContext.read
    .format("com.databricks.spark.csv")
    .option("header", "false")
    .option("mode", "DROPMALFORMED")
    .load("c:\\work\\data\\hakaton\\decriptions-data\\tfIdf2000000\\final.csv")
    .map { row => Normal(row.getString(0), row.getString(1), row.getString(2).toDouble) }

  val result = tfIdfDS.join(joinedTokens, $"permid" === $"id", "inner")
  result.printSchema()
  val m = result.take(3)
  println("end AnotherApp")
}
