use serde::Serialize;

pub struct AttractorEngine {
    history: Vec<f32>,
    capacity: usize,
    tau: usize,
    dim: usize,
}

impl AttractorEngine {
    pub fn new(capacity: usize, tau: usize, dim: usize) -> Self {
        Self {
            history: Vec::with_capacity(capacity),
            capacity,
            tau,
            dim,
        }
    }

    pub fn update(&mut self, price: f32) {
        if self.history.len() >= self.capacity {
            self.history.remove(0);
        }
        self.history.push(price);
    }

    pub fn get_trajectory(&self) -> Vec<Vec<f32>> {
        if self.history.len() < self.tau * self.dim {
            return vec![];
        }

        let mut trajectory = Vec::new();
        
        // Normalize
        let min_p = self.history.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_p = self.history.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let rng = if max_p > min_p { max_p - min_p } else { 1.0 };

        let norm_prices: Vec<f32> = self.history.iter().map(|&p| (p - min_p) / rng).collect();

        // Compute Embedding
        let end = norm_prices.len() - (self.tau * (self.dim - 1));
        for i in 0..end {
            let mut point = Vec::with_capacity(self.dim);
            for d in 0..self.dim {
                let idx = i + (d * self.tau);
                point.push(norm_prices[idx]);
            }
            trajectory.push(point);
        }

        // Return last 100 points
        let len = trajectory.len();
        if len > 100 {
            trajectory.into_iter().skip(len - 100).collect()
        } else {
            trajectory
        }
    }
}
