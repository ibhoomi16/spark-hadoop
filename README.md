# Spark and Hadoop - Gutenberg Book Analysis

This project contains different tasks performed using Apache Spark, PySpark, and Apache Hadoop. The main purpose is to load and analyze Project Gutenberg books and perform different text, metadata, and MapReduce-based analysis.

## Spark / PySpark Tasks

### Task 10 - Book Metadata Analysis

This task extracts information such as:

- Book title
- Release date
- Language
- Character set encoding

It also performs analysis such as:

- Number of books released each year
- Most common language
- Average length of book titles

Regular expressions are used to extract the required metadata from the book text.

### Task 11 - TF-IDF and Book Similarity

This task performs text preprocessing and calculates TF-IDF scores for the books.

The preprocessing includes:

- Removing Project Gutenberg header and footer
- Converting text to lowercase
- Removing punctuation
- Tokenizing the text
- Removing stop words

TF-IDF is then used to represent the books as vectors, and cosine similarity is used to find similar books.

### Task 12 - Author Influence Network

This task creates a simple author influence network based on book release years.

An influence relationship is created when one author's book was released within 5 years of another author's book.

The task includes:

- Extracting author and release year
- Creating influence edges
- Calculating in-degree
- Calculating out-degree
- Finding the top 5 authors by in-degree
- Finding the top 5 authors by out-degree

## Hadoop MapReduce

The `hadoop` folder contains the Hadoop WordCount assignment.

### WordCount

The WordCount program was implemented using Hadoop MapReduce. It includes:

- Mapper
- Reducer
- Combiner
- HDFS input and output
- Word counting using `StringTokenizer`
- Removing punctuation using `replaceAll()`
- Execution time measurement

### Input Dataset

The Gutenberg `200.txt` file was used as the input dataset. The file size was approximately 8.3 MB.

### HDFS Block and Split Experiment

The input file was stored using approximately 2 MB HDFS blocks, resulting in 4 HDFS blocks.

| Split Max Size | Number of Splits | Map Tasks | Execution Time |
|---|---:|---:|---:|
| 2 MB | 4 | 4 | 33.578 seconds |
| 8 MB | 4 | 4 | 41.566 seconds |

### Observation

Changing the split size can affect the performance because it changes how the input file is divided and processed by Mapper tasks. A smaller split size can create more splits and allow more parallel processing, but it can also increase overhead. A larger split size can reduce overhead but may provide less parallel processing.

In this experiment, both 2 MB and 8 MB resulted in 4 splits because the input file was stored in 4 HDFS blocks. The execution time was 33.578 seconds for 2 MB and 41.566 seconds for 8 MB. The difference in execution time can be due to Hadoop and system processing overhead.

## Technologies Used

- Python
- Java
- Apache Spark
- PySpark
- Apache Hadoop
- Hadoop HDFS
- Hadoop MapReduce
- Spark DataFrames
- Spark RDDs
- Regular Expressions
- TF-IDF
- Cosine Similarity

## Dataset

The project uses books from Project Gutenberg.

The dataset is not included in this repository because the book files are large. The code expects the dataset to be available locally.

## Files

```text
10.py                          - Book metadata analysis
11.py                          - TF-IDF and book similarity
12.py                          - Author influence network

hadoop/
├── WordCount.java             - Hadoop WordCount MapReduce program
├── WordCount_backup.java      - Backup WordCount implementation
└── results/
    └── experiment-results.txt - Hadoop experiment results

.gitignore
README.md
