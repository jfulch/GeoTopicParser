$ 
java -classpath tika-app/target/tika-app-<LATEST-VERSION>-SNAPSHOT.jar:
tika-parsers/tika-parsers-ml/tika-parser-nlp-package/target/tika-parser-nlp-package-<LATEST-VERSION>-SNAPSHOT.jar
:$HOME/src/location-ner-model:$HOME/src/geotopic-mime org.apache.tika.cli.TikaCLI -m polar.geot

/Users/jfulch/src/location-ner-model
/Users/jfulch/src/geotopic-mime

java -jar /Users/jfulch/tika-2.6.0/tika-app-2.6.0.jar -m polar.geot

java -classpath /Users/jfulch/tika-2.6.0/tika-app-2.6.0.jar:/path/to/tika-parser-nlp-package.jar:/Users/jfulch/src/location-ner-model:/Users/jfulch/src/geotopic-mime org.apache.tika.cli.TikaCLI -m polar.geot

java -classpath /Users/jfulch/tika-2.6.0/tika-app-2.6.0.jar:/path/to/tika-parser-nlp-package.jar:/Users/jfulch/src/location-ner-model:/Users/jfulch/src/geotopic-mime org.apache.tika.cli.TikaCLI -m polar.geot
Content-Encoding: ISO-8859-1
Content-Length: 881
Content-Type: text/plain; charset=ISO-8859-1
X-TIKA:Parsed-By: org.apache.tika.parser.DefaultParser
X-TIKA:Parsed-By: org.apache.tika.parser.csv.TextAndCSVParser
resourceName: polar.geot

java -classpath /Users/jfulch/tika-2.6.0/tika-app-2.6.0.jar:/path/to/tika-parser-nlp-package.jar:/Users/jfulch/src/location-ner-model:/Users/jfulch/src/geotopic-mime org.apache.tika.cli.TikaCLI -m polar.geot
Content-Encoding: ISO-8859-1
Content-Length: 881
Content-Type: application/geotopic; charset=ISO-8859-1
X-TIKA:Parsed-By: org.apache.tika.parser.DefaultParser
X-TIKA:Parsed-By: org.apache.tika.parser.csv.TextAndCSVParser
resourceName: polar.geot

====real====

java -classpath /Users/jfulch/tika-2.6.0/tika-app-2.6.0.jar:/Users/jfulch/tika-2.6.0/tika-parser-nlp-package-2.6.0.jar:/Users/jfulch/src/location-ner-model:/Users/jfulch/src/geotopic-mime org.apache.tika.cli.TikaCLI -m polar.geot
INFO  [main] 23:49:44,222 org.apache.tika.parser.sentiment.SentimentAnalysisParser Sentiment Model is at https://raw.githubusercontent.com/USCDataScience/SentimentAnalysisParser/master/sentiment-models/src/main/resources/edu/usc/irds/sentiment/en-netflix-sentiment.bin
Content-Length: 881
Content-Type: application/geotopic
Geographic_LATITUDE: 39.76
Geographic_LONGITUDE: -98.5
Geographic_NAME: United States
Optional_LATITUDE1: 35.0
Optional_LONGITUDE1: 105.0
Optional_NAME1: People’s Republic of China
X-TIKA:Parsed-By: org.apache.tika.parser.DefaultParser
X-TIKA:Parsed-By: org.apache.tika.parser.geo.GeoParser
X-TIKA:Parsed-By-Full-Set: org.apache.tika.parser.DefaultParser
X-TIKA:Parsed-By-Full-Set: org.apache.tika.parser.geo.GeoParser
resourceName: polar.geot

====sample====
Content-Length: 881
Content-Type: application/geotopic
Geographic_LATITUDE: 27.33931
Geographic_LONGITUDE: -108.60288
Geographic_NAME: China
Optional_LATITUDE1: 39.76
Optional_LONGITUDE1: -98.5
Optional_NAME1: United States
X-Parsed-By: org.apache.tika.parser.DefaultParser
X-Parsed-By: org.apache.tika.parser.geo.topic.GeoParser
resourceName: polar.geot

-----
java -classpath $HOME/src/location-ner-model:$HOME/src/geotopic-mime:tika-server/tika-server-standard/target/tika-server-standard-<LATEST-VERSION>-SNAPSHOT.jar:tika-parsers/tika-parsers-ml/tika-parser-nlp-package/target/tika-parser-nlp-package-<LATEST-VERSION>-SNAPSHOT.jar org.apache.tika.server.core.TikaServerCli

java -classpath /Users/jfulch/src/location-ner-model:/Users/jfulch/src/geotopic-mime:/Users/jfulch/tika-2.6.0/tika-server-standard-2.6.0.jar:/Users/jfulch/tika-2.6.0/tika-parser-nlp-package-2.6.0.jar org.apache.tika.server.core.TikaServerCli

java -classpath /Users/jfulch/src/location-ner-model:/Users/jfulch/src/geotopic-mime:/Users/jfulch/tika-2.6.0/tika-server-standard-2.6.0.jar:/Users/jfulch/tika-2.6.0/tika-parser-nlp-package-2.6.0.jar org.apache.tika.server.core.TikaServerCli -p 9997

-----

curl -T $HOME/src/geotopicparser-utils/geotopics/polar.geot -H "Content-Disposition: attachment; filename=polar.geot" http://localhost:9998/rmeta

curl -T polar.geot -H "Content-Disposition: attachment; filename=polar.geot" http://localhost:9998/rmeta

curl -T polar.geot -H "Content-Disposition: attachment; filename=polar.geot" http://localhost:9998/rmeta | jq