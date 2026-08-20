from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    input_file_name,
    regexp_extract,
    trim,
    length,
    avg,
    count,
    col,
    when
)


# ============================================================
# 1. CREATE SPARK SESSION
# ============================================================

spark = SparkSession.builder \
    .appName("Gutenberg Metadata Extraction") \
    .master("local[*]") \
    .getOrCreate()

print("\nSpark started successfully!")


# ============================================================
# 2. LOAD ALL GUTENBERG BOOKS
# ============================================================

books_df = spark.read.text(
    "dataset/D184MB/*.txt",
    wholetext=True
)

# Rename default column "value" to "text"
books_df = books_df.withColumnRenamed(
    "value",
    "text"
)

# Get complete file path
books_df = books_df.withColumn(
    "file_path",
    input_file_name()
)

# Extract only the filename
# Example:
# file:///home/bhoomi/.../10.txt -> 10.txt
books_df = books_df.withColumn(
    "file_name",
    regexp_extract(
        col("file_path"),
        r"([^/]+\.txt)$",
        1
    )
)

# Keep only required columns
books_df = books_df.select(
    "file_name",
    "text"
)

print("Books loaded successfully!")

print("\n===== BOOKS DATAFRAME =====")

books_df.show(
    5,
    truncate=80
)


# ============================================================
# 3. METADATA EXTRACTION
# ============================================================

# ------------------------------------------------------------
# TITLE
# ------------------------------------------------------------

title_pattern = r"(?im)^Title:\s*(.+)$"

books_df = books_df.withColumn(
    "title",
    trim(
        regexp_extract(
            col("text"),
            title_pattern,
            1
        )
    )
)


# ------------------------------------------------------------
# RELEASE DATE
# ------------------------------------------------------------

release_date_pattern = r"(?im)^Release Date:\s*(.+)$"

books_df = books_df.withColumn(
    "release_date",
    trim(
        regexp_extract(
            col("text"),
            release_date_pattern,
            1
        )
    )
)


# ------------------------------------------------------------
# LANGUAGE
# ------------------------------------------------------------

language_pattern = r"(?im)^Language:\s*(.+)$"

books_df = books_df.withColumn(
    "language",
    trim(
        regexp_extract(
            col("text"),
            language_pattern,
            1
        )
    )
)

# Validate extracted language
valid_languages = [
    "English",
    "Latin",
    "French",
    "German",
    "Spanish",
    "Italian",
    "Portuguese",
    "Dutch",
    "Greek",
    "Hebrew"
]

books_df = books_df.withColumn(
    "language",
    when(
        col("language").isin(valid_languages),
        col("language")
    ).otherwise(None)
)


# ------------------------------------------------------------
# CHARACTER SET ENCODING
# ------------------------------------------------------------

encoding_pattern = r"(?im)^Character set encoding:\s*(.+)$"

books_df = books_df.withColumn(
    "encoding",
    trim(
        regexp_extract(
            col("text"),
            encoding_pattern,
            1
        )
    )
)


# ============================================================
# 4. DISPLAY EXTRACTED METADATA
# ============================================================

print("\n===== EXTRACTED METADATA =====")

books_df.select(
    "file_name",
    "title",
    "release_date",
    "language",
    "encoding"
).show(
    20,
    truncate=False
)


# ============================================================
# 5. EXTRACT RELEASE YEAR
# ============================================================

year_pattern = r"(\d{4})"

books_df = books_df.withColumn(
    "release_year_string",
    regexp_extract(
        col("release_date"),
        year_pattern,
        1
    )
)

# Convert empty year strings to NULL
# before converting them to integers
books_df = books_df.withColumn(
    "release_year",
    when(
        trim(col("release_year_string")) == "",
        None
    ).otherwise(
        col("release_year_string")
    ).cast("int")
)


# ============================================================
# 6. NUMBER OF BOOKS RELEASED EACH YEAR
# ============================================================

print("\n===== BOOKS RELEASED EACH YEAR =====")

books_per_year = books_df \
    .filter(
        col("release_year").isNotNull()
    ) \
    .groupBy("release_year") \
    .agg(
        count("*").alias("number_of_books")
    ) \
    .orderBy("release_year")

books_per_year.show(
    100,
    truncate=False
)


# Count books with missing/unknown release year
missing_year_count = books_df \
    .filter(
        col("release_year").isNull()
    ) \
    .count()

print(
    "\nBooks without a valid release year:",
    missing_year_count
)


# ============================================================
# 7. MOST COMMON LANGUAGE
# ============================================================

print("\n===== LANGUAGE COUNTS =====")

language_counts = books_df \
    .filter(
        (col("language").isNotNull()) &
        (trim(col("language")) != "")
    ) \
    .groupBy("language") \
    .agg(
        count("*").alias("book_count")
    ) \
    .orderBy(
        col("book_count").desc()
    )

language_counts.show(
    20,
    truncate=False
)


print("\n===== MOST COMMON LANGUAGE =====")

language_counts.show(
    1,
    truncate=False
)


# ============================================================
# 8. AVERAGE LENGTH OF BOOK TITLES
# ============================================================

print("\n===== AVERAGE TITLE LENGTH =====")

average_title_length = books_df \
    .filter(
        (col("title").isNotNull()) &
        (trim(col("title")) != "")
    ) \
    .select(
        avg(
            length(col("title"))
        ).alias("average_title_length")
    )

average_title_length.show(
    truncate=False
)


# ============================================================
# 9. MISSING METADATA
# ============================================================

print("\n===== MISSING METADATA =====")

missing_metadata = books_df.select(

    count(
        when(
            (col("title").isNull()) |
            (trim(col("title")) == ""),
            True
        )
    ).alias("missing_titles"),

    count(
        when(
            (col("release_date").isNull()) |
            (trim(col("release_date")) == ""),
            True
        )
    ).alias("missing_release_dates"),

    count(
        when(
            (col("language").isNull()) |
            (trim(col("language")) == ""),
            True
        )
    ).alias("missing_languages"),

    count(
        when(
            (col("encoding").isNull()) |
            (trim(col("encoding")) == ""),
            True
        )
    ).alias("missing_encodings")
)

missing_metadata.show()


# ============================================================
# 10. TOTAL NUMBER OF BOOKS
# ============================================================

print("\n===== TOTAL BOOKS =====")

total_books = books_df.count()

print(
    "Total books processed:",
    total_books
)


# ============================================================
# 11. STOP SPARK
# ============================================================

spark.stop()

print("\nSpark stopped successfully!")
