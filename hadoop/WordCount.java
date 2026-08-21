import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class WordCount {

    // Mapper
    public static class TokenizerMapper
            extends Mapper<Object, Text, Text, IntWritable> {

        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();

        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {

            // Remove punctuation
            String line = value.toString()
                    .replaceAll("[^a-zA-Z0-9']", " ");

            // Split the line into words
            StringTokenizer itr = new StringTokenizer(line);

            while (itr.hasMoreTokens()) {
                word.set(itr.nextToken());
                context.write(word, one);
            }
        }
    }

    // Reducer
    public static class IntSumReducer
            extends Reducer<Text, IntWritable, Text, IntWritable> {

        private IntWritable result = new IntWritable();

        public void reduce(
                Text key,
                Iterable<IntWritable> values,
                Context context)
                throws IOException, InterruptedException {

            int sum = 0;

            for (IntWritable val : values) {
                sum += val.get();
            }

            result.set(sum);
            context.write(key, result);
        }
    }

    public static void main(String[] args) throws Exception {

        if (args.length != 2) {
            System.err.println(
                    "Usage: WordCount <input path> <output path>");
            System.exit(2);
        }

        Configuration conf = new Configuration();

        // Q9: Maximum input split size = 2 MB
        conf.setLong(
                "mapreduce.input.fileinputformat.split.maxsize",
                8 * 1024 * 1024L);

        Job job = Job.getInstance(conf, "word count");

        job.setJarByClass(WordCount.class);

        job.setMapperClass(TokenizerMapper.class);
        job.setCombinerClass(IntSumReducer.class);
        job.setReducerClass(IntSumReducer.class);

        // Mapper output / Reducer output types
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(
                job, new Path(args[0]));

        FileOutputFormat.setOutputPath(
                job, new Path(args[1]));

        // Start timer
        long startTime = System.currentTimeMillis();

        // Run MapReduce job
        boolean success = job.waitForCompletion(true);

        // Stop timer
        long endTime = System.currentTimeMillis();

        long executionTime = endTime - startTime;

        System.err.println("========================================");
        System.err.println(
                "Job finished: "
                + (success ? "SUCCESS" : "FAILURE"));
        System.err.println(
                "Total execution time: "
                + executionTime + " ms");
        System.err.println(
                "Total execution time: "
                + (executionTime / 1000.0) + " seconds");
        System.err.println("========================================");

        System.err.flush();

        System.exit(success ? 0 : 1);
    }
}
