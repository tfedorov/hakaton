package com.tfedorov.shortdesc

// $example off$
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

object PrepareDataApp extends App {

  println("start Prepare App")
  // Input data: Each row is a bag of words from a sentence or document.
  val spark = SparkSession
    .builder()
    .appName("Spark Build tfIDF model")
    .master("local[*]")
    .getOrCreate()


  import spark.implicits._

  case class DescrCSVRow(id: String, legalName: String, desc: String) {
    def addLegalName2Desc = DescrCSVRow(this.id, this.legalName, this.desc + " " + this.legalName)
  }

  case class DescrWithToken(id: String, legalName: String, desc: String, rawTokens: Seq[String], words: Seq[String]) {
    def flat1token(token: String) = DescrWithToken(this.id, this.legalName, this.desc, this.rawTokens, Seq(token))
  }

  case class Normal(id: String, words: String, maxTfIDF: Double)

  case class ResultDS(id: String, legalName: String, desc: String, words: Seq[String], rawFeatures: Vector, features: Vector) {
    def toNormal = Normal(this.id, this.words.head, this.features.toArray.max)
  }

  val descDS = spark.sqlContext.read
    .format("com.databricks.spark.csv")
    .option("header", "true")
    .option("mode", "DROPMALFORMED")
    .load("c:/work/data/hakaton/decriptions-data/desc.csv")
    .limit(100000)
    .as[DescrCSVRow]
    .map(_.addLegalName2Desc)

  val tokenizer = new Tokenizer().setInputCol("desc").setOutputCol("rawTokens")
  val remover = new StopWordsRemover().setInputCol("rawTokens").setOutputCol("words")
  val idf: IDF = new IDF().setInputCol("rawFeatures").setOutputCol("features")

  val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures")
  val descrTokenDF = tokenizer.transform(descDS)

  val noStopWordsDS = remover.transform(descrTokenDF).as[DescrWithToken]
  val tfDF: DataFrame = hashingTF.transform(noStopWordsDS)
  val idfModel: IDFModel = idf.fit(tfDF)


  val token1dDS: Dataset[DescrWithToken] = noStopWordsDS.flatMap(originDes => originDes.words.map(originDes.flat1token)).limit(2000000)

  val hashingTFDF: DataFrame = hashingTF.transform(token1dDS)

  val resultDS: Dataset[ResultDS] = idfModel.transform(hashingTFDF).as[ResultDS]

  val normalDS: Dataset[Normal] = resultDS.map(_.toNormal)//.filter(_.maxTfIDF > 4.0)
  println("before save")

  normalDS.show
  //  println(normalDS.count)
  /*
  normalDS.write
    .format("com.databricks.spark.csv")
    .option("header", "false")
    .mode("overwrite")
    .save("c:/work/data/hakaton/decriptions-data/tmpIDF")
  */
  println("end Prepare App")
}
