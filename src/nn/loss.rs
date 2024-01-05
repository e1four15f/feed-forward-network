use crate::nn::activation::softmax;

/// Represents the cross-entropy loss function combined with a softmax layer.
#[derive(Debug)]
pub struct CrossEntropyLossWithSoftmax {
    // context for backward pass
    probabilities: Vec<Vec<f32>>,
    labels: Vec<u8>,
}

impl CrossEntropyLossWithSoftmax {
    pub fn new() -> Self {
        Self {
            probabilities: Default::default(),
            labels: Default::default(),
        }
    }

    pub fn forward(&mut self, logits: &Vec<Vec<f32>>, labels: &Vec<u8>) -> f32 {
        // Apply Softmax here
        let probabilities = softmax(&logits);

        self.probabilities = probabilities.clone();
        self.labels = labels.clone();

        let batch_size = labels.len();
        let mut loss: f32 = 0.0;

        for i in 0..batch_size {
            let class_probabilities = &probabilities[i];
            let label = labels[i] as usize;
            // Adding epsilon to protect against computational instability
            let predicted_probability = class_probabilities[label] + f32::EPSILON;
            loss -= predicted_probability.ln();
        }

        loss / batch_size as f32 // normalize by batch_size
    }

    pub fn backward(&self) -> Vec<Vec<f32>> {
        let (batch_size, num_classes) = (self.probabilities.len(), self.probabilities[0].len());

        // Create a matrix to store the gradients.
        let mut gradients = vec![vec![0.0; num_classes]; batch_size];

        for i in 0..batch_size {
            for c in 0..num_classes {
                if c == self.labels[i] as usize {
                    gradients[i][c] = self.probabilities[i][c] - 1.0;
                } else {
                    gradients[i][c] = self.probabilities[i][c];
                }
                // Normalize gradients by the batch size.
                gradients[i][c] /= batch_size as f32;
            }
        }

        gradients
    }
}
