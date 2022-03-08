import pulsar

client = pulsar.Client('pulsar://localhost:6650')

producer = client.create_producer('sentiment-in')

producer.send(('Hello Dennis this is a beautiful day!').encode('utf-8'))
producer.send(('We can safely assume that Apache Pulsar functions can produce amazing results').encode('utf-8'))
producer.send(('It can be either good or bad depending on how you look at it').encode('utf-8'))
producer.send(('Apache Pulsar is the best thing after sliced potatoes').encode('utf-8'))
producer.send(('Imagine Python and Apache Pulsar combined how they can benefit your organisation').encode('utf-8'))

client.close()