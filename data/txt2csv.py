import csv
f = open('training.txt','r')
lines = f.readlines()
f.close()

sentiments = []
sentences = []
for line in lines:
	sentiments.append(int(line[0]))
	sentences.append(line[1:].strip())
with open('training.csv', 'w') as csvfile:
	spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
	spamwriter.writerow(['sentiment', 'sentence'])
	for i in range(len(sentiments)):
		spamwriter.writerow([sentiments[i], sentences[i]])
csvfile.close()