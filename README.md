# sentiment-pulsar

This is a demonstration of how easy it is to deploy your ML models in Apache Pulsar and get real-time results. It uses the Python **nltk** library and creates an Apache Pulsar function that produces a sentiment analysis on any given text.

To bring up the project run 


                ./start.sh

To produce/consume messages just run the producer.py/consumer.py files and make sure the apache client Python library is installed on your machine

                pip install pulsar-client==2.8.1

To monitor the function performance and see the status open your browser at:

                http://localhost:8080/admin/v2/functions/public/default/SentimentFunction/status

The result is a string with a compound/positive/negative/neutral score for any given text.

Sentiment Analysis code from https://www.nltk.org/