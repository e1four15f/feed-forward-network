use crate::nn::linear::Linear;
use crate::nn::loss::CrossEntropyLossWithSoftmax;

/// Feed-Forward network with N linear layers for multiclass classification.
#[derive(Debug)]
pub struct FeedForwardNetwork {
    pub layers: Vec<Linear>,
    pub loss: CrossEntropyLossWithSoftmax,
}

impl FeedForwardNetwork {
    pub fn forward(&mut self, input: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let mut x = input.clone();
        for layer in &mut self.layers {
            x = layer.forward(&x);
        }
        x
    }

    pub fn loss(&mut self, logits: &Vec<Vec<f32>>, labels: &Vec<u8>) -> f32 {
        self.loss.forward(logits, labels)
    }

    pub fn backward(&mut self) {
        let mut activation_gradients = self.loss.backward();
        for layer in self.layers.iter_mut().rev() {
            activation_gradients = layer.backward(&activation_gradients);
        }
    }
}
