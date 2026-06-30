from pyspark.sql import SparkSession

#Initialize Spark Session with Hive support
spark = SparkSession.builder \
    .appName("HealthcareDataPartitioningBucketing") \
    .config("hive.exec.dynamic.partition", "true") \
    .config("hive.exec.dynamic.partition.mode", "nonstrict") \
    .enableHiveSupport() \
    .getOrCreate()

#---------------------------------------------------------
#1.Create the medical_visits table (partitioned + bucketed)
spark.sql("""
CREATE TABLE IF NOT EXISTS medical_visits (
    visit_id INT,
    patient_id INT,
    diagnosis STRING,
    treatment STRING
)
PARTITIONED BY (visit_date STRING, region STRING)
CLUSTERED BY (diagnosis, patient_id) INTO 10 BUCKETS
STORED AS ORC
""")

#---------------------------------------------------------
#2.Create the prescriptions table (partitioned + bucketed)
spark.sql("""
CREATE TABLE IF NOT EXISTS prescriptions (
    prescription_id INT,
    visit_id INT,
    patient_age INT,
    patient_gender STRING,
    drug_name STRING,
    dosage STRING
)
PARTITIONED BY (prescription_date STRING, region STRING)
CLUSTERED BY (patient_age, drug_name) INTO 10 BUCKETS
STORED AS ORC
""")

#---------------------------------------------------------
#3.Load source data into DataFrames
#---------------------------------------------------------
#Assumes raw CSV files (no partition columns embedded yet)
visits_df = spark.read.csv(
    "/path/to/medical_visits_2023-09-01_North.csv",
    header=True,
    inferSchema=True
)

prescriptions_df = spark.read.csv(
    "/path/to/prescriptions_2023-09-01_North.csv",
    header=True,
    inferSchema=True
)

#---------------------------------------------------------
#Write data into partitioned/bucketed Hive tables
#(insertInto respects existing partitioning/bucketing
#defined on the target table)
from pyspark.sql.functions import lit

#Add partition columns explicitly if not already in source data
visits_df = visits_df.withColumn("visit_date", lit("2023-09-01")) \
                      .withColumn("region", lit("North"))

prescriptions_df = prescriptions_df.withColumn("prescription_date", lit("2023-09-01")) \
                                    .withColumn("region", lit("North"))

#Enable dynamic partitioning for the write
spark.conf.set("hive.exec.dynamic.partition", "true")
spark.conf.set("hive.exec.dynamic.partition.mode", "nonstrict")

visits_df.write \
    .mode("append") \
    .format("hive") \
    .insertInto("medical_visits")

prescriptions_df.write \
    .mode("append") \
    .format("hive") \
    .insertInto("prescriptions")

#---------------------------------------------------------
#5a.Patient Visits by Region and Date
visits_by_region_date = spark.sql("""
    SELECT COUNT(visit_id) AS total_visits
    FROM medical_visits
    WHERE visit_date = '2023-09-01' AND region = 'North'
""")
visits_by_region_date.show()

# Equivalent using DataFrame API
visits_table_df = spark.table("medical_visits")
visits_by_region_date_df = visits_table_df \
    .filter((col("visit_date") == "2023-09-01") & (col("region") == "North")) \
    .agg(count("visit_id").alias("total_visits"))
visits_by_region_date_df.show()

#---------------------------------------------------------
#5b.Top Prescribed Drugs by Region
top_drugs_by_region = spark.sql("""
    SELECT drug_name, COUNT(drug_name) AS total_prescriptions
    FROM prescriptions
    WHERE region = 'North'
    GROUP BY drug_name
    ORDER BY total_prescriptions DESC
""")
top_drugs_by_region.show()

#Equivalent using DataFrame API
from pyspark.sql.functions import col, count, desc

prescriptions_table_df = spark.table("prescriptions")
top_drugs_by_region_df = prescriptions_table_df \
    .filter(col("region") == "North") \
    .groupBy("drug_name") \
    .agg(count("drug_name").alias("total_prescriptions")) \
    .orderBy(desc("total_prescriptions"))
top_drugs_by_region_df.show()

#---------------------------------------------------------
#5c.Patient Demographic Analysis for a Specific Drug
demographics_for_drug = spark.sql("""
    SELECT patient_age, patient_gender, COUNT(drug_name) AS prescriptions
    FROM prescriptions
    WHERE drug_name = 'Lisinopril'
    GROUP BY patient_age, patient_gender
""")
demographics_for_drug.show()

#Equivalent using DataFrame API
demographics_for_drug_df = prescriptions_table_df \
    .filter(col("drug_name") == "Lisinopril") \
    .groupBy("patient_age", "patient_gender") \
    .agg(count("drug_name").alias("prescriptions"))
demographics_for_drug_df.show()

#---------------------------------------------------------
#Stop Spark Session
spark.stop()
