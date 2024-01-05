use crate::config::RANDOM;
use crate::data::ndarray::{Array, Matrix};
use crate::nn::activation::{ReLU, Softmax};
use crate::nn::ops::{matmul, transpose_matrix};
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use rand_distr::Normal;

/// Types of parameters initialization.
#[derive(Debug)]
pub enum Initialization {
    Xavier,
    Kaiming,
}

/// Types of layer activations.
#[derive(Debug)]
pub enum Activation {
    None,
    ReLU,
    #[allow(dead_code)]
    Softmax,
}

#[derive(Debug)]
pub struct Linear {
    pub weights: Matrix,
    pub biases: Array,
    activation: Activation,

    // context for backward pass
    input_data: Vec<Vec<f32>>,
    output_data: Vec<Vec<f32>>,
}

impl Linear {
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        initialization: Initialization,
        activation: Activation,
    ) -> Self {
        let mut weights = Matrix::with_zeros(output_dim, input_dim);
        let mut biases = Array::with_zeros(output_dim);

        for i in 0..output_dim {
            for j in 0..input_dim {
                weights.data[i][j] = match initialization {
                    // Normalized Xavier Weight Initialization
                    Initialization::Xavier => {
                        let bound: f64 = 6.0_f64.sqrt() / ((input_dim + output_dim) as f64).sqrt();
                        let distribution = Uniform::try_from(-bound..bound).unwrap();
                        unsafe { distribution.sample(&mut *RANDOM) as f32 }
                    }
                    // Kaiming Weight Initialization
                    Initialization::Kaiming => {
                        let std: f64 = (2.0 / input_dim as f64).sqrt();
                        let distribution = Normal::new(0.0, std).unwrap();
                        unsafe { distribution.sample(&mut *RANDOM) as f32 }
                    }
                }
            }

            biases.data[i] = match initialization {
                // Since we're using ReLU neurons, it is also good practice to initialize them with a slightly positive initial bias to avoid "dead neurons".
                Initialization::Xavier => unsafe { RANDOM.gen::<f32>() * 0.1 },
                // In Kaiming Initialization biases are initialized at zero
                Initialization::Kaiming => 0.0,
            }
        }
        Self {
            weights,
            biases,
            activation,
            input_data: Default::default(),
            output_data: Default::default(),
        }
    }
}

impl Linear {
    pub fn forward(&mut self, input: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        self.input_data = input.clone();

        let mut output = matmul(&input, &transpose_matrix(&self.weights.data));

        // Add biases
        for j in 0..output[0].len() {
            for i in 0..output.len() {
                output[i][j] += self.biases.data[j];
            }
        }

        match self.activation {
            Activation::ReLU => ReLU::forward(&mut output),
            Activation::Softmax => Softmax::forward(&mut output),
            Activation::None => {}
        }

        self.output_data = output.clone();

        output
    }

    pub fn backward(&mut self, gradient: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let activation_gradient = match self.activation {
            Activation::ReLU => {
                ReLU::backward(&mut self.output_data, &gradient);
                &self.output_data
            }
            Activation::Softmax => todo!("Backward pass for Softmax is not implemented rn"),
            Activation::None => gradient,
        };
        let transposed_activation_gradient = transpose_matrix(&activation_gradient);
        // Store computed gradients
        self.weights.grad = matmul(&transposed_activation_gradient, &self.input_data);
        self.biases.grad = transposed_activation_gradient
            .iter()
            .map(|x| x.iter().sum())
            .collect();

        let input_gradient = matmul(&activation_gradient, &self.weights.data);
        input_gradient
    }
}
