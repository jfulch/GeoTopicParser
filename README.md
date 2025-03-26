# GeoTopicParser

```bash
cd ~/lucene-geo-gazetteer
lucene-geo-gazetteer -server
```

```bash
java -Dgazetteer.url=http://localhost:8765 -classpath /Users/jfulch/src/location-ner-model:/Users/jfulch/src/geotopic-mime:/Users/jfulch/tika-2.6.0/tika-server-standard-2.6.0.jar:/Users/jfulch/tika-2.6.0/tika-parser-nlp-package-2.6.0.jar org.apache.tika.server.core.TikaServerCli
```

```bash
curl -T polar.geot -H "Content-Disposition: attachment; filename=polar.geot" http://localhost:9998/rmeta | jq
```

Sample Output:
```json
[
  {
    "Geographic_LONGITUDE": "105.0",
    "Geographic_NAME": "People’s Republic of China",
    "X-TIKA:Parsed-By-Full-Set": [
      "org.apache.tika.parser.DefaultParser",
      "org.apache.tika.parser.geo.GeoParser"
    ],
    "resourceName": "polar.geot",
    "Optional_NAME1": "United States",
    "Optional_LATITUDE1": "39.76",
    "Optional_LONGITUDE1": "-98.5",
    "X-TIKA:Parsed-By": [
      "org.apache.tika.parser.DefaultParser",
      "org.apache.tika.parser.geo.GeoParser"
    ],
    "X-TIKA:parse_time_millis": "840",
    "X-TIKA:embedded_depth": "0",
    "Geographic_LATITUDE": "35.0",
    "Content-Length": "881",
    "Content-Type": "application/geotopic"
  }
]
```

curl -T polar.geot -H "Content-Disposition: attachment; filename=test.geot" http://localhost:9998/rmeta | jq