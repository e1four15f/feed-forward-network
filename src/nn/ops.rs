// Parallelization crate
use rayon::prelude::*;
use std::collections::HashMap;
use std::iter::zip;

/// Performs matrix multiplication between two matrices.
pub fn matmul(left: &Vec<Vec<f32>>, right: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let (d1, d2, d3) = (left.len(), right.len(), right[0].len());

    // Create a mutable result matrix
    let mut result = vec![vec![0.0; d3]; d1];

    // Use parallel iterators for the outer loop
    result.par_iter_mut().enumerate().for_each(|(i, row)| {
        for j in 0..d3 {
            let mut sum = 0.0;
            for k in 0..d2 {
                sum += left[i][k] * right[k][j];
            }
            row[j] = sum;
        }
    });

    return result;
}

/// Transposes a given matrix, swapping its rows and columns.
pub fn transpose_matrix(matrix: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let (rows, cols) = (matrix.len(), matrix[0].len());

    let transposed_data: Vec<Vec<f32>> = (0..cols)
        .into_par_iter()
        .map(|j| (0..rows).into_iter().map(|i| matrix[i][j]).collect())
        .collect();

    transposed_data
}

/// Finds the index of the maximum element in each row of a matrix.
pub fn argmax(matrix: &Vec<Vec<f32>>) -> Vec<u8> {
    let (d1, _d2) = (matrix.len(), matrix[0].len());
    let mut result = vec![0; d1];

    for i in 0..d1 {
        let mut max_index = 0;
        let mut max_value = matrix[i][0];

        for (i, &value) in matrix[i].iter().enumerate() {
            if value > max_value {
                max_value = value;
                max_index = i;
            }
        }

        result[i] = max_index as u8;
    }

    return result;
}

/// Calculates the accuracy of predictions against true labels.
pub fn accuracy(predictions: &Vec<u8>, labels: &Vec<u8>) -> f64 {
    let n_examples = labels.len() as f64;
    let mut correct_by_class = HashMap::new();

    // Ensure that predictions and labels have the same length
    if predictions.len() != labels.len() {
        panic!(
            "Input vectors have different lengths. Got '{}' and '{}'",
            predictions.len(),
            labels.len()
        );
    }

    for (prediction, label) in zip(predictions, labels) {
        if prediction == label {
            match correct_by_class.get(label) {
                Some(x) => correct_by_class.insert(label, x + 1),
                _ => correct_by_class.insert(label, 1),
            };
        }
    }
    let correct_sum: u32 = correct_by_class.values().sum();
    println!("Got {} correct in {} examples", correct_sum, n_examples);
    println!("{correct_by_class:#?}");
    return correct_sum as f64 / n_examples;
}
