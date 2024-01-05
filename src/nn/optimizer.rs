use crate::nn::linear::Linear;
use std::iter::zip;

/// ADAM optimizer for neural networks.
pub struct ADAM {
    learning_rate: f32,
    decay: f32,
    beta1: f32,
    beta2: f32,
    // Keep previous updates
    updates: Vec<LayerUpdate>,
    step: i32,
}

impl ADAM {
    pub fn new(learning_rate: f32, decay: f32, beta1: f32, beta2: f32, params: &[Linear]) -> Self {
        let updates = params
            .iter()
            .map(|layer| {
                let [output_dim, input_dim] = layer.weights.dims;
                LayerUpdate::new(input_dim, output_dim)
            })
            .collect();

        Self {
            learning_rate,
            decay,
            beta1,
            beta2,
            updates,
            step: 0,
        }
    }

    /// Performs a single optimization step.
    pub fn update(&mut self, params: &mut [Linear]) {
        // Adjust the learning rate using exponential decay
        self.step += 1;
        let learning_rate = self.get_step_learning_rate();

        for (layer, layer_update) in zip(params.iter_mut(), self.updates.iter_mut()) {
            Self::update_layer(
                layer,
                layer_update,
                learning_rate,
                self.beta1,
                self.beta2,
                self.step,
            );
        }
    }

    /// Calculates the learning rate at the current step with exponential decay.
    pub fn get_step_learning_rate(&self) -> f32 {
        self.learning_rate / (1.0 + self.decay * self.step as f32)
    }

    /// Method to update a single linear layer parameters.
    fn update_layer(
        layer: &mut Linear,
        layer_update: &mut LayerUpdate,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        t: i32, // timestep for bias correction
    ) {
        let [d1, d2] = layer.weights.dims;
        for i in 0..d1 {
            // Update weights
            for j in 0..d2 {
                // Update moving averages
                layer_update.m_weights[i][j] =
                    beta1 * layer_update.m_weights[i][j] + (1.0 - beta1) * layer.weights.grad[i][j];
                layer_update.v_weights[i][j] = beta2 * layer_update.v_weights[i][j]
                    + (1.0 - beta2) * layer.weights.grad[i][j].powi(2);

                // Bias correction
                let m_hat = layer_update.m_weights[i][j] / (1.0 - beta1.powi(t));
                let v_hat = layer_update.v_weights[i][j] / (1.0 - beta2.powi(t));

                // Update weights
                layer.weights.data[i][j] -= learning_rate * m_hat / (v_hat.sqrt() + f32::EPSILON);
            }

            // Update biases
            // Update moving averages
            layer_update.m_biases[i] =
                beta1 * layer_update.m_biases[i] + (1.0 - beta1) * layer.biases.grad[i];
            layer_update.v_biases[i] =
                beta2 * layer_update.v_biases[i] + (1.0 - beta2) * layer.biases.grad[i].powi(2);

            // Bias correction
            let m_hat = layer_update.m_biases[i] / (1.0 - beta1.powi(t));
            let v_hat = layer_update.v_biases[i] / (1.0 - beta2.powi(t));

            // Update weights
            layer.biases.data[i] -= learning_rate * m_hat / (v_hat.sqrt() + f32::EPSILON);
        }
    }
}

/// Maintains moving averages of gradients for a single layer.
#[derive(Debug)]
struct LayerUpdate {
    m_weights: Vec<Vec<f32>>, // Moving average of gradients (weights)
    v_weights: Vec<Vec<f32>>, // Moving average of squared gradients (weights)
    m_biases: Vec<f32>,       // Moving average of gradients (biases)
    v_biases: Vec<f32>,       // Moving average of squared gradients (biases)
}

impl LayerUpdate {
    fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            m_weights: vec![vec![0.0; input_dim]; output_dim],
            v_weights: vec![vec![0.0; input_dim]; output_dim],
            m_biases: vec![0.0; output_dim],
            v_biases: vec![0.0; output_dim],
        }
    }
}
