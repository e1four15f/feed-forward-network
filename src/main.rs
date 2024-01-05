mod config;
mod data;
mod nn;
use crate::config::{
    BATCH_SIZE, BETA_1, BETA_2, DECAY, LEARNING_RATE, LINEAR_DIMS, LOG_STEPS, NUM_CLASSES,
    NUM_EPOCHS, TEST_LABELS_PATH, TEST_PREDICTIONS_PATH, TEST_VECTORS_PATH, TRAIN_LABELS_PATH,
    TRAIN_PREDICTIONS_PATH, TRAIN_VECTORS_PATH, VALID_SIZE,
};
use crate::data::dataset::DataLoader;
use crate::nn::linear::{Activation, Initialization, Linear};
use crate::nn::loss::CrossEntropyLossWithSoftmax;
use crate::nn::network::FeedForwardNetwork;
use crate::nn::ops::{accuracy, argmax};
use crate::nn::optimizer::ADAM;
use std::fs::File;
use std::io::Write;
use std::iter::{once, zip};
use std::time::SystemTime;

fn main() {
    let main_time = SystemTime::now();

    // Load dataset into memory
    let mut data_loader = DataLoader::new(
        TRAIN_VECTORS_PATH,
        TRAIN_LABELS_PATH,
        TEST_VECTORS_PATH,
        TEST_LABELS_PATH,
        VALID_SIZE,
        BATCH_SIZE,
    );
    // Initialize the network
    let mut network = FeedForwardNetwork {
        layers: {
            LINEAR_DIMS
                .windows(2)
                // Feature extraction linear layers
                .map(|dims| Linear::new(dims[0], dims[1], Initialization::Xavier, Activation::ReLU))
                // Last classification layer to n_classes
                .chain(once(Linear::new(
                    *LINEAR_DIMS.last().unwrap(),
                    NUM_CLASSES,
                    Initialization::Kaiming,
                    Activation::None,
                )))
                .collect()
        },
        // We use Softmax within Loss functions instead of adding it to last layer
        // This simplify the computation in backward pass
        loss: CrossEntropyLossWithSoftmax::new(),
    };
    // Initialize the optimizer
    let mut optimizer = ADAM::new(LEARNING_RATE, DECAY, BETA_1, BETA_2, &network.layers);

    // Train loop
    let num_batches: i32 = ((60_000. * (1. - VALID_SIZE)) / BATCH_SIZE as f32) as i32;
    for epoch in 0..NUM_EPOCHS {
        let epoch_time = SystemTime::now();
        for (step, (data, labels)) in data_loader.train_iterator().enumerate() {
            let step_time = SystemTime::now();

            // Forward pass
            let logits = network.forward(&data);
            // Calculate loss
            let loss = network.loss(&logits, &labels);
            // Backward pass
            network.backward();
            // Update parameters using the optimizer
            optimizer.update(&mut network.layers);

            // Logging
            if step % LOG_STEPS == 0 || step == (num_batches - 1) as usize {
                let current_learning_rate = optimizer.get_step_learning_rate();
                let step_time = step_time.elapsed().unwrap().as_millis();
                let step = step + 1;
                println!(
                    "Epoch #{epoch:>2}/{NUM_EPOCHS:<2} | \
                     Batch #{step:>3}/{num_batches:<3} | \
                     Loss {loss:.4} | \
                     Learning rate {current_learning_rate:.8} | \
                     Step time {step_time:3} ms",
                );
            }
        }

        println!("############");
        println!("Validation metrics for epoch #{}/{NUM_EPOCHS}", epoch + 1);
        evaluate(
            &mut network,
            &mut data_loader.valid_iterator(),
            "Valid",
            None,
        );
        println!(
            "Epoch time: {} sec",
            epoch_time.elapsed().unwrap().as_secs()
        );
        println!("############");
    }
    evaluate(
        &mut network,
        &mut data_loader.original_train_iterator(),
        "Original train",
        Some(TRAIN_PREDICTIONS_PATH),
    );

    println!("############");
    evaluate(
        &mut network,
        &mut data_loader.test_iterator(),
        "Test",
        Some(TEST_PREDICTIONS_PATH),
    );
    println!(
        "Main time: {} min",
        main_time.elapsed().unwrap().as_secs() / 60
    );
}

/// Evaluates network on give data. Computes accuracy and write predictions on disk.
fn evaluate(
    network: &mut FeedForwardNetwork,
    data_iterator: impl Iterator<Item = (Vec<Vec<f32>>, Vec<u8>)>,
    split: &str,
    file_path: Option<&str>,
) {
    println!("Evaluating on {} split", split);
    let (predictions, labels): (Vec<u8>, Vec<u8>) = data_iterator
        .flat_map(|(x, y)| zip(argmax(&network.forward(&x)), y))
        .unzip();
    let accuracy = accuracy(&predictions, &labels);
    println!("{} accuracy: {:.4}", split, accuracy);

    match file_path {
        Some(path) => {
            write_predictions_to_file(&predictions, path);
            println!("Predictions saved to {}", path);
        }
        _ => {}
    }
}

fn write_predictions_to_file(predictions: &Vec<u8>, file_path: &str) {
    let mut file = File::create(file_path).unwrap();
    for prediction in predictions {
        writeln!(file, "{}", prediction).unwrap();
    }
}
