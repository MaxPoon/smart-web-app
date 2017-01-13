import graphlab

graphlab.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 4)

training_data, test_data = graphlab.SFrame.read_csv('../data/training.csv').random_split(0.9)
training_data['word_count'] = graphlab.text_analytics.count_words(training_data['sentence'])

sentiment_word_count_model = graphlab.logistic_classifier.create(training_data, 
																 target='sentiment', 
																 features=['word_count'],
																 max_iterations=60,
																 l2_penalty=0.01,
																 validation_set=None)
training_data['tfidf'] = graphlab.text_analytics.tf_idf(training_data['word_count'])

sentiment_tfidf_model  = graphlab.logistic_classifier.create(training_data, 
															 target='sentiment', 
															 features=['tfidf'],
															 max_iterations=60,
															 l2_penalty=0.01,
															 validation_set=None)

test_data['word_count'] = graphlab.text_analytics.count_words(test_data['sentence'])
test_data['tfidf'] = graphlab.text_analytics.tf_idf(test_data['word_count'])

correct1, correct2 = 0, 0
for data in test_data:
	if sentiment_word_count_model.predict(data)[0] == data['sentiment']: correct1+=1
	if sentiment_tfidf_model.predict(data)[0] == data['sentiment']: correct2+=1
print(correct1/float(len(test_data)), correct2/float(len(test_data)))

