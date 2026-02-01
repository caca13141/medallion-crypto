use serde::Serialize;
use std::collections::HashMap;

#[derive(Serialize, Clone)]
pub struct GhostCluster {
    pub price: f32,
    pub volume: f32,
    pub count: usize,
}

pub struct GhostEngine {
    // We keep state if needed
}

impl GhostEngine {
    pub fn new() -> Self {
        Self {}
    }

    pub fn detect(&self, bids: &Vec<(f32, f32)>, asks: &Vec<(f32, f32)>) -> Vec<GhostCluster> {
        // Simplified "Grid Clustering" for High Performance
        // Instead of full DBSCAN, we bucket orders into price grids and find density peaks.
        // This is O(N) and much faster for 1D/2D order book data.

        let mut grid: HashMap<i64, f32> = HashMap::new();
        let grid_size = 10.0; // $10 buckets for BTC

        // 1. Map to Grid
        for (p, s) in bids.iter().chain(asks.iter()) {
            let bucket = (*p / grid_size).floor() as i64;
            *grid.entry(bucket).or_insert(0.0) += *s;
        }

        // 2. Find Peaks (Ghost Clusters)
        let mut clusters: Vec<GhostCluster> = grid.into_iter()
            .map(|(bucket, volume)| GhostCluster {
                price: (bucket as f32) * grid_size + (grid_size / 2.0),
                volume,
                count: 1 // Simplified
            })
            .filter(|c| c.volume > 5.0) // Filter noise
            .collect();

        // 3. Sort by Volume (Descending)
        clusters.sort_by(|a, b| b.volume.partial_cmp(&a.volume).unwrap());
        
        clusters.into_iter().take(10).collect()
    }
}
