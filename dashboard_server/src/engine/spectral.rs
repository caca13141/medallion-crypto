use rustfft::{FftPlanner, num_complex::Complex};
use std::sync::Arc;

pub struct SpectralEngine {
    planner: Arc<Mutex<FftPlanner<f32>>>,
    history: Vec<f32>,
    window_size: usize,
}

use std::sync::Mutex;

impl SpectralEngine {
    pub fn new(window_size: usize) -> Self {
        Self {
            planner: Arc::new(Mutex::new(FftPlanner::new())),
            history: Vec::with_capacity(window_size),
            window_size,
        }
    }

    pub fn update(&mut self, inter_arrival_time: f32) {
        if self.history.len() >= self.window_size {
            self.history.remove(0);
        }
        self.history.push(inter_arrival_time);
    }

    pub fn compute_spectrum(&self) -> Vec<(f32, f32)> {
        if self.history.len() < self.window_size {
            return vec![];
        }

        let mut planner = self.planner.lock().unwrap();
        let fft = planner.plan_fft_forward(self.window_size);

        let mut buffer: Vec<Complex<f32>> = self.history
            .iter()
            .map(|&x| Complex { re: x, im: 0.0 })
            .collect();

        fft.process(&mut buffer);

        // Calculate power spectrum
        let mut spectrum = Vec::new();
        for (i, val) in buffer.iter().enumerate().take(self.window_size / 2) {
            let power = val.norm_sqr();
            let freq = i as f32; // Simplified frequency mapping
            if i > 0 { // Skip DC component
                spectrum.push((freq, power));
            }
        }
        
        // Sort by power (descending) and take top 20
        spectrum.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        spectrum.into_iter().take(20).collect()
    }
}
