echo "Bringing up pulsar"
docker-compose up -d

echo "Install NLTK Python lib"
docker exec -ti pulsar_sentiment pip install nltk

echo "Copy function inside container"
docker cp sentiment_function.py pulsar_sentiment:/pulsar/sentiment_function.py

echo "Delete pre-existing instances of the function"
docker exec -ti pulsar_sentiment ./bin/pulsar-admin functions delete \
  --tenant public \
  --namespace default \
  --name SentimentFunction 

echo "Create new instance of Sentiment Function"
docker exec -ti pulsar_sentiment ./bin/pulsar-admin functions create \
--py /pulsar/sentiment_function.py \
--classname sentiment_function.SentimentFunction \
--inputs persistent://public/default/sentiment-in \
--output persistent://public/default/sentiment-out \
--tenant public \
--namespace default \
--name SentimentFunction


