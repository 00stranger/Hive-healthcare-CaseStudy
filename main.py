"""
Healthcare Hive Partitioning & Bucketing Pipeline

Creates partitioned + bucketed Hive tables for medical visits and
prescriptions, loads a day/region's worth of data into them, and runs the
three business reports (visits by region/date, top prescribed drugs by
region, patient demographics for a given drug).
"""

import logging

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, count, desc, lit

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

#Config -- edit these per run, or wire up argparse/env vars if this ever
#needs to run for more than one date/region without manual edits.
VISITS_CSV_PATH = "/path/to/medical_visits_2023-09-01_North.csv"
PRESCRIPTIONS_CSV_PATH = "/path/to/prescriptions_2023-09-01_North.csv"
VISIT_DATE = "2023-09-01"
PRESCRIPTION_DATE = "2023-09-01"
REGION = "North"
DEMOGRAPHICS_DRUG = "Lisinopril"


#Spark session
def get_spark_session() -> SparkSession:
    return (
        SparkSession.builder.appName("HealthcareDataPartitioningBucketing")
        .config("hive.exec.dynamic.partition", "true")
        .config("hive.exec.dynamic.partition.mode", "nonstrict")
        .enableHiveSupport()
        .getOrCreate()
    )

#Table creation (idempotent -- safe to call every run)
def create_tables(spark: SparkSession) -> None:
    logger.info("Ensuring medical_visits table exists")
    spark.sql(
        """
        CREATE TABLE IF NOT EXISTS medical_visits (
            visit_id   INT,
            patient_id INT,
            diagnosis  STRING,
            treatment  STRING
        )
        PARTITIONED BY (visit_date STRING, region STRING)
        CLUSTERED BY (diagnosis, patient_id) INTO 10 BUCKETS
        STORED AS ORC
        """
    )

    logger.info("Ensuring prescriptions table exists")
    spark.sql(
        """
        CREATE TABLE IF NOT EXISTS prescriptions (
            prescription_id INT,
            visit_id        INT,
            patient_age     INT,
            patient_gender  STRING,
            drug_name       STRING,
            dosage          STRING
        )
        PARTITIONED BY (prescription_date STRING, region STRING)
        CLUSTERED BY (patient_age, drug_name) INTO 10 BUCKETS
        STORED AS ORC
        """
    )

#Ingestion
def _reorder_to_table_schema(spark: SparkSession, df: DataFrame, table_name: str) -> DataFrame:
    target_columns = spark.table(table_name).columns
    missing = set(target_columns) - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns required by '{table_name}': {missing}")
    return df.select(*target_columns)


def load_and_write_partition(
    spark: SparkSession,
    csv_path: str,
    table_name: str,
    partition_values: dict,
) -> None:
    logger.info("Reading %s", csv_path)
    try:
        df = spark.read.csv(csv_path, header=True, inferSchema=True)
    except Exception as exc:
        raise FileNotFoundError(f"Could not read source file at '{csv_path}': {exc}") from exc

    for col_name, value in partition_values.items():
        df = df.withColumn(col_name, lit(value))

    df = _reorder_to_table_schema(spark, df, table_name)

    logger.info("Writing %d rows into %s partition %s", df.count(), table_name, partition_values)
    df.write.mode("append").insertInto(table_name)


#Reports
def visits_by_region_and_date(spark: SparkSession, visit_date: str, region: str) -> DataFrame:
    return (
        spark.table("medical_visits")
        .filter((col("visit_date") == visit_date) & (col("region") == region))
        .agg(count("visit_id").alias("total_visits"))
    )


def top_prescribed_drugs_by_region(spark: SparkSession, region: str) -> DataFrame:
    return (
        spark.table("prescriptions")
        .filter(col("region") == region)
        .groupBy("drug_name")
        .agg(count("drug_name").alias("total_prescriptions"))
        .orderBy(desc("total_prescriptions"))
    )


def patient_demographics_for_drug(spark: SparkSession, drug_name: str) -> DataFrame:
    return (
        spark.table("prescriptions")
        .filter(col("drug_name") == drug_name)
        .groupBy("patient_age", "patient_gender")
        .agg(count("drug_name").alias("prescriptions"))
    )


#Main
def main():
    spark = get_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    try:
        create_tables(spark)

        load_and_write_partition(
            spark,
            VISITS_CSV_PATH,
            "medical_visits",
            {"visit_date": VISIT_DATE, "region": REGION},
        )
        load_and_write_partition(
            spark,
            PRESCRIPTIONS_CSV_PATH,
            "prescriptions",
            {"prescription_date": PRESCRIPTION_DATE, "region": REGION},
        )

        logger.info("Report: Patient visits by region and date")
        visits_by_region_and_date(spark, VISIT_DATE, REGION).show()

        logger.info("Report: Top prescribed drugs by region")
        top_prescribed_drugs_by_region(spark, REGION).show()

        logger.info("Report: Patient demographics for drug '%s'", DEMOGRAPHICS_DRUG)
        patient_demographics_for_drug(spark, DEMOGRAPHICS_DRUG).show()

    finally:
        spark.stop()


if __name__ == "__main__":
    main()