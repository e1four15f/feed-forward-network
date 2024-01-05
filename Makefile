metrics:
	@echo "Train"
	@python3 evaluator/evaluate.py train_predictions.csv data/fashion_mnist_train_labels.csv
	@echo "Test"
	@python3 evaluator/evaluate.py test_predictions.csv data/fashion_mnist_test_labels.csv

clean:
	@rm -r test_predictions.csv train_predictions.csv network target PV021_540500.zip || true

run: clean
	@sh run.sh
	@make metrics

zip: clean
	zip -r PV021_540500.zip . -x "*.git*" -x "*.idea*" -x "*.gitignore"
