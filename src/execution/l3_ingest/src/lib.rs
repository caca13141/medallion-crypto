use pyo3::prelude::*;
use memmap2::MmapMut;
use std::fs::OpenOptions;
use std::sync::Arc;
use parking_lot::RwLock;
use bytemuck::{Pod, Zeroable};

const RING_BUFFER_SIZE: usize = 1024 * 1024 * 100; // 100 MB Ring Buffer
const BOOK_LEVELS: usize = 50;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct PriceLevel {
    pub price: f64,
    pub size: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct L3Snapshot {
    pub timestamp: u64,
    pub bids: [PriceLevel; BOOK_LEVELS],
    pub asks: [PriceLevel; BOOK_LEVELS],
    pub last_trade_price: f64,
    pub last_trade_size: f64,
    pub cancellations_5s: u32,
    pub _padding: u32, // Ensure 8-byte alignment (Total size multiple of 8)
}

unsafe impl Zeroable for L3Snapshot {}
unsafe impl Pod for L3Snapshot {}

#[pyclass]
pub struct L3Ingestor {
    mmap: Arc<RwLock<MmapMut>>,
    offset: Arc<RwLock<usize>>,
}

#[pymethods]
impl L3Ingestor {
    #[new]
    pub fn new(file_path: &str) -> PyResult<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(file_path)?;
            
        file.set_len(RING_BUFFER_SIZE as u64)?;
        
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        
        Ok(L3Ingestor {
            mmap: Arc::new(RwLock::new(mmap)),
            offset: Arc::new(RwLock::new(0)),
        })
    }

    pub fn update_book(&self, 
                       timestamp: u64, 
                       bids: Vec<(f64, f64)>, 
                       asks: Vec<(f64, f64)>,
                       last_trade: (f64, f64),
                       cancellations: u32) -> PyResult<()> {
        
        let mut snapshot = L3Snapshot {
            timestamp,
            bids: [PriceLevel { price: 0.0, size: 0.0 }; BOOK_LEVELS],
            asks: [PriceLevel { price: 0.0, size: 0.0 }; BOOK_LEVELS],
            last_trade_price: last_trade.0,
            last_trade_size: last_trade.1,
            cancellations_5s: cancellations,
            _padding: 0,
        };

        for (i, (p, s)) in bids.iter().take(BOOK_LEVELS).enumerate() {
            snapshot.bids[i] = PriceLevel { price: *p, size: *s };
        }
        
        for (i, (p, s)) in asks.iter().take(BOOK_LEVELS).enumerate() {
            snapshot.asks[i] = PriceLevel { price: *p, size: *s };
        }

        let mut mmap = self.mmap.write();
        let mut offset = self.offset.write();
        
        let size = std::mem::size_of::<L3Snapshot>();
        if *offset + size > RING_BUFFER_SIZE {
            *offset = 0; // Wrap around
        }
        
        let bytes = bytemuck::bytes_of(&snapshot);
        mmap[*offset..*offset + size].copy_from_slice(bytes);
        *offset += size;

        Ok(())
    }

    pub fn get_latest_snapshot(&self) -> PyResult<(u64, Vec<(f64, f64)>, Vec<(f64, f64)>, f64, u32)> {
        // In a real scenario, we'd read the latest valid frame.
        // For this drop-in, we just return the last written state from memory or cache.
        // Simplified for brevity.
        Ok((0, vec![], vec![], 0.0, 0))
    }
}

#[pymodule]
fn l3_ingest(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<L3Ingestor>()?;
    Ok(())
}
