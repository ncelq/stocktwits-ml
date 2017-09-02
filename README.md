# StockTwits Machine Learning

## Prerequisite
- Apache Spark 2.2

## Configuration
The spark home is set in src/main/resources/application.properties
```
spark.home=/<spark home>/Software/spark
```

## Build
```
mvn clean install
```

## Run
```
mvn sprint-boot:run
```

## Model Accurancy
72.%

## Prediction
A web application http://localhost:8080/index.html can query to ML model to make a prediction
