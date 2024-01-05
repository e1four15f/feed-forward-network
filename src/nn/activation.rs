/// Represents the ReLU (Rectified Linear Unit) activation function.
#[derive(Debug)]
pub struct ReLU;

impl ReLU {
    pub fn forward(matrix: &mut Vec<Vec<f32>>) {
        for i in 0..matrix.len() {
            for j in 0..matrix[0].len() {
                matrix[i][j] = matrix[i][j].max(0.0);
            }
        }
    }

    pub fn backward(matrix: &mut Vec<Vec<f32>>, gradients: &Vec<Vec<f32>>) {
        for i in 0..matrix.len() {
            for j in 0..matrix[0].len() {
                if matrix[i][j] > 0.0 {
                    matrix[i][j] = gradients[i][j];
                } else {
                    matrix[i][j] = 0.0;
                }
            }
        }
    }
}

/// Represents the Softmax activation function.
#[derive(Debug)]
pub struct Softmax;

impl Softmax {
    pub fn forward(matrix: &mut Vec<Vec<f32>>) {
        for i in 0..matrix.len() {
            // Ensuring numerical stability by shifting logits distribution by max_value
            let max_value = matrix[i].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            for j in 0..matrix[0].len() {
                let exponent = (matrix[i][j] - max_value).exp();
                matrix[i][j] = exponent;
            }

            let sum_exp: f32 = matrix[i].iter().sum();
            for j in 0..matrix[0].len() {
                matrix[i][j] /= sum_exp;
            }
        }
    }
}

/// Standalone Softmax function
pub fn softmax(matrix: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut matrix = matrix.clone();
    Softmax::forward(&mut matrix);
    matrix
}
