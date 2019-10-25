package com.tfedorov.shortdesc

import com.tfedorov.shortdesc.PrepareDataApp.{DescrCSVRow, Normal}
import org.apache.spark.ml.feature.{StopWordsRemover, Tokenizer}
import org.apache.spark.sql.{Dataset, SparkSession}

object CalculateApp extends App {

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

  val descDS = spark.sqlContext.read
    .format("com.databricks.spark.csv")
    .option("header", "true")
    .option("mode", "DROPMALFORMED")
    .load("c:/work/data/hakaton/decriptions-data/desc.csv").as[DescrCSVRow]
    .map(_.addLegalName2Desc)
  val joinedDF = joinedTextDF.join(descDS, $"permid" === $"id", "inner")

  val tokenizerDesc = new Tokenizer().setInputCol("desc").setOutputCol("descTokens")
  val removerDesc = new StopWordsRemover().setInputCol("descTokens").setOutputCol("descFilteredTokens")

  val tokenizerText = new Tokenizer().setInputCol("content").setOutputCol("textTokens")
  val removerText = new StopWordsRemover().setInputCol("textTokens").setOutputCol("textFilteredTokens")

  val descTokensDF = removerDesc.transform(tokenizerDesc.transform(joinedDF))

  case class JoinedTokened(permid: String, fileName: String, descFilteredTokens: Seq[String], textFilteredTokens: Seq[String], alias: String, label: String) {
    def intersect = {
      val in = descFilteredTokens.intersect(textFilteredTokens)
      IntersectdTokened(this.permid, this.fileName, in.mkString("|"), this.alias, this.label)
    }
  }

  case class IntersectdTokened(permid: String, fileName: String, intersect: String, alias: String, label: String)

  val intersectedDF: Dataset[IntersectdTokened] = removerText.transform(tokenizerText.transform(descTokensDF))
    .select("permid", "fileName", "descFilteredTokens", "textFilteredTokens", "alias", "label")
    .as[JoinedTokened]
    .map(_.intersect)
    .as[IntersectdTokened]

  val tfIdfDS: Dataset[Normal] = spark.sqlContext.read
    .format("com.databricks.spark.csv")
    .option("header", "false")
    .option("mode", "DROPMALFORMED")
    .load("c:\\work\\data\\hakaton\\decriptions-data\\tfIdf2000000\\final.csv")
    .map { row => Normal(row.getString(0), row.getString(1), row.getString(2).toDouble) }


  case class Grouped(permid: String, fileName: String, alias: String, label: String, maxTfIDF: Double) {
    def merge(another: Grouped) = {
      Grouped(this.permid, this.fileName, this.alias, this.label, this.maxTfIDF + another.maxTfIDF)
    }
  }
  case class GroupedLabeled(permid: String, fileName: String, alias: String, label: Double, maxTfIDF: Double)

  val result = intersectedDF.join(tfIdfDS, $"permid" === $"id", "inner")
    //.filter(r => r.getString(2).contains(r.getString(6)))

  result.write
    .format("com.databricks.spark.csv")
    .option("header", "false")
    .mode("append")
    .save("c:/work/data/hakaton/decriptions-data/groupedByTokens")

  println("end AnotherApp")
}
