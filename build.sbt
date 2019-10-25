name := "CompanyDescr"

version := "0.1"

scalaVersion := "2.11.8"

val sparkVersion = "2.4.0"

libraryDependencies += "org.apache.spark" %% "spark-core" % sparkVersion

libraryDependencies += "org.apache.spark" %% "spark-sql" % sparkVersion

libraryDependencies += "org.apache.spark" %% "spark-mllib" % sparkVersion

//libraryDependencies += "org.scalatest" %% "scalatest" % "1.9.1" % "test"