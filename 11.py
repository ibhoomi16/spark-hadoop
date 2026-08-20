from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    input_file_name,
    regexp_extract,
    col,
    lower,
    regexp_replace,
    split,
    trim
)

from pyspark.ml.feature import (
    StopWordsRemover,
    CountVectorizer,
    IDF,
    Normalizer
)

import numpy as np


# ============================================================
# 1. CREATE SPARK SESSION
# ============================================================

spark = SparkSession.builder \
    .appName("TF-IDF Book Similarity") \
    .master("local[*]") \
    .getOrCreate()

print("Spark started successfully!")


# ============================================================
# 2. LOAD ALL GUTENBERG BOOKS
# ============================================================

books_df = spark.read.text(
    "dataset/D184MB/*.txt",
    wholetext=True
)

books_df = books_df.withColumnRenamed(
    "value",
    "text"
)

# Get complete file path
books_df = books_df.withColumn(
    "file_path",
    input_file_name()
)

# Extract only file name
books_df = books_df.withColumn(
    "file_name",
    regexp_extract(
        col("file_path"),
        r"([^/]+\.txt)$",
        1
    )
)

# Keep required columns
books_df = books_df.select(
    "file_name",
    "text"
)

print("Books loaded:", books_df.count())


# ============================================================
# 3. PREPROCESSING
# ============================================================

# Convert text to lowercase
books_df = books_df.withColumn(
    "clean_text",
    lower(col("text"))
)


# ------------------------------------------------------------
# Remove Project Gutenberg header
# ------------------------------------------------------------

books_df = books_df.withColumn(
    "clean_text",
    regexp_replace(
        col("clean_text"),
        r"(?s)^.*?\*\*\* start of this project gutenberg ebook.*?\*\*\*",
        ""
    )
)


# ------------------------------------------------------------
# Remove Project Gutenberg footer
# ------------------------------------------------------------

books_df = books_df.withColumn(
    "clean_text",
    regexp_replace(
        col("clean_text"),
        r"(?s)\*\*\* end of this project gutenberg ebook.*$",
        ""
    )
)


# ------------------------------------------------------------
# Remove punctuation and numbers
# ------------------------------------------------------------

books_df = books_df.withColumn(
    "clean_text",
    regexp_replace(
        col("clean_text"),
        r"[^a-z\s]",
        " "
    )
)


# ------------------------------------------------------------
# Remove extra spaces
# ------------------------------------------------------------

books_df = books_df.withColumn(
    "clean_text",
    regexp_replace(
        col("clean_text"),
        r"\s+",
        " "
    )
)


# ------------------------------------------------------------
# Tokenize into words
# ------------------------------------------------------------

books_df = books_df.withColumn(
    "words",
    split(
        trim(col("clean_text")),
        r"\s+"
    )
)


# ------------------------------------------------------------
# Remove stop words
# ------------------------------------------------------------

remover = StopWordsRemover(
    inputCol="words",
    outputCol="filtered_words"
)

books_df = remover.transform(books_df)


print("\n===== PREPROCESSED TEXT =====")

books_df.select(
    "file_name",
    "filtered_words"
).show(
    10,
    truncate=100
)


# ============================================================
# 4. TERM FREQUENCY (TF)
# ============================================================

print("\n===== CALCULATING TERM FREQUENCY =====")

vectorizer = CountVectorizer(
    inputCol="filtered_words",
    outputCol="tf_features",
    vocabSize=10000,
    minDF=2
)

vectorizer_model = vectorizer.fit(books_df)

books_df = vectorizer_model.transform(books_df)


print("\n===== TERM FREQUENCY (TF) =====")

books_df.select(
    "file_name",
    "tf_features"
).show(
    5,
    truncate=80
)


# ============================================================
# 5. INVERSE DOCUMENT FREQUENCY (IDF)
# ============================================================

print("\n===== CALCULATING INVERSE DOCUMENT FREQUENCY =====")

idf = IDF(
    inputCol="tf_features",
    outputCol="tfidf_features"
)

idf_model = idf.fit(books_df)

books_df = idf_model.transform(books_df)


# ============================================================
# 6. TF-IDF = TF * IDF
# ============================================================

print("\n===== TF-IDF FEATURES =====")

books_df.select(
    "file_name",
    "tfidf_features"
).show(
    5,
    truncate=80
)


# ============================================================
# 7. NORMALIZE TF-IDF VECTORS
# ============================================================
# Normalization is required for cosine similarity.
#
# Cosine similarity:
#
#              A . B
# Cosine =  -----------
#             |A| |B|
#
# After normalization, |A| and |B| become 1,
# so cosine similarity can be calculated using
# the dot product.


normalizer = Normalizer(
    inputCol="tfidf_features",
    outputCol="normalized_tfidf"
)

books_df = normalizer.transform(books_df)


# ============================================================
# 8. COLLECT BOOK VECTORS
# ============================================================

book_vectors = books_df.select(
    "file_name",
    "normalized_tfidf"
).collect()


file_names = [
    row["file_name"]
    for row in book_vectors
]

vectors = np.array([
    row["normalized_tfidf"].toArray()
    for row in book_vectors
])


print("\nTotal book vectors:", len(file_names))


# ============================================================
# 9. CALCULATE COSINE SIMILARITY
# ============================================================

print("\n===== CALCULATING COSINE SIMILARITY =====")

# Since vectors are normalized,
# dot product = cosine similarity

similarity_matrix = np.dot(
    vectors,
    vectors.T
)


# ============================================================
# 10. FIND BOOK 10.TXT
# ============================================================

target_book = "10.txt"

if target_book not in file_names:
    print("10.txt was not found in the dataset.")

else:

    target_index = file_names.index(
        target_book
    )

    target_similarities = similarity_matrix[
        target_index
    ]


    # ========================================================
    # 11. FIND TOP 5 MOST SIMILAR BOOKS
    # ========================================================

    # Sort similarities from highest to lowest
    sorted_indices = np.argsort(
        target_similarities
    )[::-1]


    # Remove 10.txt itself
    sorted_indices = [
        i
        for i in sorted_indices
        if file_names[i] != target_book
    ]


    top_5_indices = sorted_indices[:5]


    print("\n===== TOP 5 BOOKS SIMILAR TO 10.TXT =====")

    for rank, index in enumerate(
        top_5_indices,
        start=1
    ):
        print(
            rank,
            ".",
            file_names[index],
            "-> Cosine Similarity:",
            round(
                float(target_similarities[index]),
                6
            )
        )


# ============================================================
# 12. STOP SPARK
# ============================================================

spark.stop()

print("\nSpark stopped successfully!")
