/// Represents a 1-dimensional array with data and gradient vectors.
#[derive(Debug)]
pub struct Array {
    pub data: Vec<f32>,
    pub grad: Vec<f32>,
    pub dims: [usize; 1],
}

impl Array {
    pub fn with_zeros(d1: usize) -> Self {
        Self {
            data: vec![0.0; d1],
            grad: vec![0.0; d1],
            dims: [d1],
        }
    }
}

/// Represents a 2-dimensional matrix with data and gradient matrices.
#[derive(Debug)]
pub struct Matrix {
    pub data: Vec<Vec<f32>>,
    pub grad: Vec<Vec<f32>>,
    pub dims: [usize; 2],
}

impl Matrix {
    pub fn with_zeros(d1: usize, d2: usize) -> Self {
        Self {
            data: vec![vec![0.0; d2]; d1],
            grad: vec![vec![0.0; d2]; d1],
            dims: [d1, d2],
        }
    }
}
