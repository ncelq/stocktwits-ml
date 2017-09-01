package com.spark.poc.ml;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IDFModel;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

@Component
public class MLModel {
	private NaiveBayesModel model;
	private StructType schema = DataTypes
			.createStructType(new StructField[] { DataTypes.createStructField("label", DataTypes.DoubleType, false),
					DataTypes.createStructField("sentence", DataTypes.StringType, false) });

	@Autowired
	private SparkSession sparkSession;

	final static Logger logger = LoggerFactory.getLogger(MLModel.class);

	private Dataset<Row> prepare(Dataset<Row> sentenceData) {
		Tokenizer tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words");
		Dataset<Row> wordsData = tokenizer.transform(sentenceData);

		HashingTF hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures");

		Dataset<Row> featurizedData = hashingTF.transform(wordsData);
		// alternatively, CountVectorizer can also be used to get term frequency
		// vectors

		IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");
		IDFModel idfModel = idf.fit(featurizedData);

		return idfModel.transform(featurizedData);
	}

	private String predict(String text) {
		
		List<Row> data = Arrays.asList(RowFactory.create(0.0, text));
		Dataset<Row> predict = model.transform(prepare(sparkSession.createDataFrame(data, schema)));

		predict.show();
		Row row = (Row)predict.javaRDD().collect().get(0);
		double prediction = row.getDouble(row.fieldIndex("prediction"));
		
		if (prediction==1.0) {
			return "Bullish";
		} else if (prediction==0.0) {
			return "Bearish";
		} else {
			return "N/A";
		}

		/*
		predict.javaRDD().collect().forEach((Row row) -> {
			System.out.println(row.getDouble(row.fieldIndex("label")) - row.getDouble(row.fieldIndex("prediction")));
		});
		*/
	}

	public void trigger() {
		Dataset<Row> sentenceData = sparkSession.read().schema(schema)
				.csv(this.getClass().getResource("/data/ml-data.txt").toString());

		Dataset<Row> rescaledData = prepare(sentenceData);

		// rescaledData.select("label", "features").show();

		Dataset<Row>[] splits = rescaledData.randomSplit(new double[] { 0.7, 0.3 }, 1234L);
		Dataset<Row> train = splits[0];
		Dataset<Row> test = splits[1];

		NaiveBayes nb = new NaiveBayes();

		// train the model
		model = nb.fit(train);

		// Select example rows to display.
		Dataset<Row> predictions = model.transform(test);
		
		predictions.show();

		predictions.javaRDD().collect().forEach((Row row) -> {
			if (row.getDouble(row.fieldIndex("label")) - row.getDouble(row.fieldIndex("prediction"))==0.0) {
					System.out.println(row.getString(row.fieldIndex("sentence")) +","+ row.getDouble(row.fieldIndex("prediction")));
			}
		});
		// compute accuracy on the test set
		MulticlassClassificationEvaluator evaluator;

		evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy");
		double accuracy = evaluator.evaluate(predictions);
		
		logger.info("");
		logger.info("*************************");
		logger.info("       Test Report       ");
		logger.info("*************************");
		logger.info("Accurancy: "+accuracy);
		logger.info("");
		
		
		logger.info("------------------------>"+predict("go up soon"));

		logger.info("------------------------>"+predict("Big drop coming"));

		sparkSession.stop();
	}
}