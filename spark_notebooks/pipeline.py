from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F

class PySparkJob:

    def __init__(self):
        """Initialize Spark session"""
        self.spark = SparkSession.builder.master("local").appName("Data Cleaning").getOrCreate()

    def filter_medical(self, eligibility: DataFrame, medicals: DataFrame) -> DataFrame:
        """Remove all rows from medical whose memberId is not present in eligibility"""
        return medicals.join(eligibility, "memberId", "inner")

    def generate_full_name(self, eligibility: DataFrame, medicals: DataFrame) -> DataFrame:
        """Populate fullName in medical by concatenating firstName and lastName from eligibility"""
        eligibility = eligibility.withColumn("fullName", F.concat_ws(" ", F.col("firstName"), F.col("lastName")))
        
        medicals = medicals.alias("m").join(
            eligibility.select("memberId", "fullName").alias("e"),
            F.col("m.memberId") == F.col("e.memberId"),
            "left"
        ).select(
            F.col("m.memberId"),
            F.col("e.fullName"),
            F.col("m.paidAmount")
        )

        return medicals

    def find_max_paid_member(self, medicals: DataFrame) -> str:
        """Find the member who has the highest paidAmount and return their memberId"""
        return medicals.orderBy(F.col("paidAmount").desc()).limit(1).collect()[0]["memberId"]

    def find_total_paid_amount(self, medicals: DataFrame) -> int:
        """Find the total sum of paidAmount"""
        return medicals.select(F.sum("paidAmount")).collect()[0][0]

