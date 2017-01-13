import graphlab

graphlab.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 4)

training_data = graphlab.SFrame.read_csv('../data/training.csv')
training_data['word_count'] = graphlab.text_analytics.count_words(training_data['sentence'])

sentiment_word_count_model = graphlab.logistic_classifier.create(training_data, 
																 target='sentiment', 
																 features=['word_count'],
																 max_iterations=50,
																 validation_set=None)

training_data['tfidf'] = graphlab.text_analytics.tf_idf(training_data['word_count'])
sentiment_tfidf_model  = graphlab.logistic_classifier.create(training_data, 
															 target='sentiment', 
															 features=['tfidf'],
															 max_iterations=50,
															 validation_set=None)
