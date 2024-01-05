use crate::config::RANDOM;
use csv::ReaderBuilder;
use rand::prelude::*;
use std::error::Error;
use std::iter::zip;

/// Represents a dataset with data and corresponding labels.
pub struct Dataset {
    pub data: Vec<Vec<f32>>,
    pub labels: Vec<u8>,
}

impl Dataset {
    /// Shuffles the dataset using a predefined random number generator.
    /// This method ensures that data and labels are shuffled correctly.
    pub fn shuffle(&mut self) {
        let mut indices: Vec<usize> = (0..self.data.len()).collect();
        unsafe {
            indices.shuffle(&mut *RANDOM);
        }

        self.data = indices.iter().map(|&i| self.data[i].clone()).collect();
        self.labels = indices.iter().map(|&i| self.labels[i]).collect();
    }
}

/// DataLoader responsible for loading and iterating over datasets.
pub struct DataLoader {
    train_dataset: Dataset,
    valid_dataset: Dataset,
    test_dataset: Dataset,
    original_train_dataset: Dataset,
    batch_size: usize,
}

impl DataLoader {
    /// Constructs a new DataLoader.
    /// Splits the training data into train and validation sets based on `valid_size`.
    /// Loads test data and keeps a copy of the original training data
    pub fn new(
        train_data_path: &str,
        train_labels_path: &str,
        test_data_path: &str,
        test_labels_path: &str,
        valid_size: f32,
        batch_size: usize,
    ) -> Self {
        let train_data = read_vectors(train_data_path).unwrap();
        let train_labels = read_labels(train_labels_path).unwrap();
        let test_data = read_vectors(test_data_path).unwrap();
        let test_labels = read_labels(test_labels_path).unwrap();

        // Keep original train dataset for last prediction
        let original_train_data = train_data.clone();
        let original_train_labels = train_labels.clone();

        // Split the training data into training and validation sets
        let valid_len = (valid_size * train_data.len() as f32).round() as usize;
        let valid_data = train_data[train_data.len() - valid_len..].to_vec();
        let valid_labels = train_labels[train_labels.len() - valid_len..].to_vec();
        let train_data = train_data[..train_data.len() - valid_len].to_vec();
        let train_labels = train_labels[..train_labels.len() - valid_len].to_vec();

        DataLoader {
            train_dataset: Dataset {
                data: train_data,
                labels: train_labels,
            },
            valid_dataset: Dataset {
                data: valid_data,
                labels: valid_labels,
            },
            test_dataset: Dataset {
                data: test_data,
                labels: test_labels,
            },
            original_train_dataset: Dataset {
                data: original_train_data,
                labels: original_train_labels,
            },
            batch_size,
        }
    }

    /// Provides an iterator over the training dataset.
    /// Each call to this method will shuffle the training data.
    pub fn train_iterator(&mut self) -> impl Iterator<Item = (Vec<Vec<f32>>, Vec<u8>)> + '_ {
        self.train_dataset.shuffle();
        self.create_iterator(&self.train_dataset.data, &self.train_dataset.labels)
    }

    /// Provides an iterator over the validation dataset.
    pub fn valid_iterator(&self) -> impl Iterator<Item = (Vec<Vec<f32>>, Vec<u8>)> + '_ {
        self.create_iterator(&self.valid_dataset.data, &self.valid_dataset.labels)
    }

    /// Provides an iterator over the test dataset.
    pub fn test_iterator(&self) -> impl Iterator<Item = (Vec<Vec<f32>>, Vec<u8>)> + '_ {
        self.create_iterator(&self.test_dataset.data, &self.test_dataset.labels)
    }

    /// Provides an iterator over the original training dataset.
    pub fn original_train_iterator(&self) -> impl Iterator<Item = (Vec<Vec<f32>>, Vec<u8>)> + '_ {
        self.create_iterator(
            &self.original_train_dataset.data,
            &self.original_train_dataset.labels,
        )
    }

    /// Helper method to create an batch-iterator from given data and labels.
    fn create_iterator<'a>(
        &self,
        data: &'a [Vec<f32>],
        labels: &'a [u8],
    ) -> impl Iterator<Item = (Vec<Vec<f32>>, Vec<u8>)> + 'a {
        zip(data.chunks(self.batch_size), labels.chunks(self.batch_size))
            .map(|(data_chunk, label_chunk)| (data_chunk.to_vec(), label_chunk.to_vec()))
    }
}

/// Reads and returns a vector of vectors from a CSV file.
fn read_vectors(file_path: &str) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .from_path(file_path)?;
    let mut vectors = Vec::new();

    for result in reader.records() {
        let record = result?;
        let vector: Vec<f32> = record
            .iter()
            // Image normalization from [0, 255] to [-1, 1]
            .map(|x| x.parse::<f32>().unwrap() / 127.5 - 1.0)
            .collect();
        vectors.push(vector);
    }

    Ok(vectors)
}

/// Reads and returns a vector of labels from a CSV file.
fn read_labels(file_path: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .from_path(file_path)?;
    let mut labels = Vec::new();

    for result in reader.records() {
        let record = result?;
        let label = record.iter().next().unwrap().parse::<u8>().unwrap();
        labels.push(label);
    }

    Ok(labels)
}
