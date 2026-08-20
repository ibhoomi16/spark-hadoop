from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# 1. CREATE SPARK SESSION
# ============================================================

spark = (
    SparkSession.builder
    .appName("AuthorInfluenceNetwork")
    .master("local[*]")
    .getOrCreate()
)

print("Spark started successfully!")


# ============================================================
# 2. LOAD GUTENBERG DATASET
# ============================================================

books_df = spark.read.text(
    "dataset/D184MB/*.txt",
    wholetext=True
)

books_df = books_df.withColumnRenamed(
    "value",
    "text"
)

books_df = books_df.withColumn(
    "file_path",
    F.input_file_name()
)

books_df = books_df.withColumn(
    "file_name",
    F.regexp_extract(
        F.col("file_path"),
        r"([^/]+\.txt)$",
        1
    )
)

books_df = books_df.select(
    "file_name",
    "text"
)

print("Books loaded:", books_df.count())


# ============================================================
# 3. EXTRACT AUTHOR
# ============================================================

# Pattern 1: Author: Name
author_pattern_1 = (
    r"(?im)^\s*Author(?:s)?\s*:\s*(.+?)\s*$"
)

books_df = books_df.withColumn(
    "author_1",
    F.trim(
        F.regexp_extract(
            F.col("text"),
            author_pattern_1,
            1
        )
    )
)


# Pattern 2: by Author Name
author_pattern_2 = (
    r"(?im)^\s*by\s+"
    r"([A-Z][A-Za-z.'-]+"
    r"(?:\s+[A-Z][A-Za-z.'-]+){0,7})"
    r"\s*(?:,|\r?$)"
)

books_df = books_df.withColumn(
    "author_2",
    F.trim(
        F.regexp_extract(
            F.col("text"),
            author_pattern_2,
            1
        )
    )
)


# Pattern 3: By Author Name
author_pattern_3 = (
    r"(?im)^\s*By\s+"
    r"([A-Z][A-Za-z.'-]+"
    r"(?:\s+[A-Z][A-Za-z.'-]+){0,7})"
    r"\s*$"
)

books_df = books_df.withColumn(
    "author_3",
    F.trim(
        F.regexp_extract(
            F.col("text"),
            author_pattern_3,
            1
        )
    )
)


# Select the first valid author
books_df = books_df.withColumn(
    "author",
    F.when(
        F.col("author_1") != "",
        F.col("author_1")
    )
    .when(
        F.col("author_2") != "",
        F.col("author_2")
    )
    .when(
        F.col("author_3") != "",
        F.col("author_3")
    )
    .otherwise(None)
)


# ============================================================
# 4. EXTRACT RELEASE DATE
# ============================================================

release_date_pattern = (
    r"(?im)^Release Date:\s*(.+)$"
)

books_df = books_df.withColumn(
    "release_date",
    F.trim(
        F.regexp_extract(
            F.col("text"),
            release_date_pattern,
            1
        )
    )
)


# ============================================================
# 5. EXTRACT RELEASE YEAR
# ============================================================

books_df = books_df.withColumn(
    "release_year_string",
    F.regexp_extract(
        F.col("release_date"),
        r"(\d{4})",
        1
    )
)

books_df = books_df.withColumn(
    "release_year",
    F.when(
        F.col("release_year_string") == "",
        None
    )
    .otherwise(
        F.col("release_year_string").cast("int")
    )
)


# ============================================================
# 6. DISPLAY EXTRACTED DATA
# ============================================================

print("\n===== EXTRACTED AUTHOR AND RELEASE DATE =====")

books_df.select(
    "file_name",
    "author",
    "release_date",
    "release_year"
).show(
    20,
    truncate=False
)


# ============================================================
# 7. REMOVE INVALID AUTHORS
# ============================================================

valid_books = books_df.filter(
    (F.col("author").isNotNull()) &
    (F.trim(F.col("author")) != "") &
    (F.col("release_year").isNotNull()) &
    (~F.lower(F.trim(F.col("author"))).isin(
        "project gutenberg",
        "the mount horeb.",
        "unknown"
    ))
)

print(
    "\nBooks with valid author and release year:",
    valid_books.count()
)


# ============================================================
# 8. CREATE TWO DATAFRAMES FOR PAIR COMPARISON
# ============================================================

left = valid_books.select(
    F.col("author").alias("author1"),
    F.col("release_year").alias("year1")
)

right = valid_books.select(
    F.col("author").alias("author2"),
    F.col("release_year").alias("year2")
)


# ============================================================
# 9. DEFINE INFLUENCE WINDOW
# ============================================================

X = 5

print(
    "\nInfluence time window:",
    X,
    "years"
)


# ============================================================
# 10. CREATE AUTHOR PAIRS
# ============================================================

candidate_pairs = (
    left
    .crossJoin(right)
    .where(
        (F.col("author1") != F.col("author2"))
        &
        (
            F.abs(
                F.col("year1") - F.col("year2")
            ) <= X
        )
    )
)


# ============================================================
# 11. CREATE INFLUENCE EDGES
# ============================================================

edges_df = (
    candidate_pairs
    .select(
        "author1",
        "author2"
    )
    .distinct()
)

print(
    f"\n===== AUTHOR INFLUENCE NETWORK EDGES "
    f"(WITHIN {X} YEARS) ====="
)

edges_df.show(
    50,
    truncate=False
)


# ============================================================
# 12. CONVERT EDGES TO RDD
# ============================================================

edges_rdd = edges_df.rdd.map(
    lambda row: (
        row["author1"],
        row["author2"]
    )
)

print("\n===== SAMPLE EDGES AS RDD TUPLES =====")

for edge in edges_rdd.take(10):
    print(edge)


# ============================================================
# 13. TOTAL INFLUENCE RELATIONSHIPS
# ============================================================

total_edges = edges_df.count()

print(
    "\nTotal influence relationships:",
    total_edges
)


# ============================================================
# 14. OUT-DEGREE
# ============================================================

out_degree = (
    edges_df
    .groupBy("author1")
    .agg(
        F.countDistinct("author2")
        .alias("out_degree")
    )
    .orderBy(
        F.desc("out_degree")
    )
)

print(
    "\n===== OUT-DEGREE "
    "(AUTHORS POTENTIALLY INFLUENCED) ====="
)

out_degree.show(
    20,
    truncate=False
)


# ============================================================
# 15. IN-DEGREE
# ============================================================

in_degree = (
    edges_df
    .groupBy("author2")
    .agg(
        F.countDistinct("author1")
        .alias("in_degree")
    )
    .orderBy(
        F.desc("in_degree")
    )
)

print(
    "\n===== IN-DEGREE "
    "(AUTHORS POTENTIALLY INFLUENCING) ====="
)

in_degree.show(
    20,
    truncate=False
)


# ============================================================
# 16. TOP 5 BY OUT-DEGREE
# ============================================================

print(
    "\n===== TOP 5 AUTHORS BY OUT-DEGREE ====="
)

out_degree.show(
    5,
    truncate=False
)


# ============================================================
# 17. TOP 5 BY IN-DEGREE
# ============================================================

print(
    "\n===== TOP 5 AUTHORS BY IN-DEGREE ====="
)

in_degree.show(
    5,
    truncate=False
)


# ============================================================
# 18. STOP SPARK
# ============================================================

spark.stop()

print("\nSpark stopped successfully!")
