use once_cell::sync::Lazy;
use rand::prelude::*;
use rand::rngs::StdRng;

// Random
pub const SEED: [u8; 32] = [137; 32];
pub static mut RANDOM: Lazy<StdRng> = Lazy::new(|| StdRng::from_seed(SEED));

// Dataset params
pub const TRAIN_VECTORS_PATH: &str = "data/fashion_mnist_train_vectors.csv";
pub const TRAIN_LABELS_PATH: &str = "data/fashion_mnist_train_labels.csv";
pub const TEST_VECTORS_PATH: &str = "data/fashion_mnist_test_vectors.csv";
pub const TEST_LABELS_PATH: &str = "data/fashion_mnist_test_labels.csv";

// Model params
pub const LINEAR_DIMS: &[usize] = &[784, 512, 256];
pub const NUM_CLASSES: usize = 10;

// Training params
pub const NUM_EPOCHS: usize = 5; // or 10
pub const VALID_SIZE: f32 = 0.15;
pub const BATCH_SIZE: usize = 256;

// Optimizer params
pub const LEARNING_RATE: f32 = 1e-3;
pub const DECAY: f32 = 1e-3;
pub const BETA_1: f32 = 0.9;
pub const BETA_2: f32 = 0.999;

// Logging
pub const LOG_STEPS: usize = 20;
pub const TEST_PREDICTIONS_PATH: &str = "test_predictions.csv";
pub const TRAIN_PREDICTIONS_PATH: &str = "train_predictions.csv";
