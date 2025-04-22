use std::{cell::RefCell, rc::Rc};

use crate::DType;

/// A simple buffer pool to reuse Vec allocations across recursive graph evaluation.
pub struct BufferPool<T> {
    pool: Vec<Vec<T>>,
}

/// Shared reference to a BufferPool for automatic recycling.
pub type SharedPool<T> = Rc<RefCell<BufferPool<T>>>;

/// Wrapper around Vec<T> that returns its buffer to the pool on drop.
pub struct PooledBuffer<T: DType> {
    buf: Vec<T>,
    pool: Option<SharedPool<T>>,
}

impl<T: DType> PooledBuffer<T> {
    /// Wrap an existing Vec and attach it to the pool.
    pub fn new(buf: Vec<T>, pool: SharedPool<T>) -> Self {
        PooledBuffer {
            buf,
            pool: Some(pool),
        }
    }

    /// Consume the wrapper and return the inner Vec without recycling.
    pub fn into_inner(mut self) -> Vec<T> {
        let buf = std::mem::take(&mut self.buf);
        self.pool = None;
        buf
    }
}

impl<T: DType> std::ops::Deref for PooledBuffer<T> {
    type Target = Vec<T>;
    fn deref(&self) -> &Vec<T> {
        &self.buf
    }
}

impl<T: DType> std::ops::DerefMut for PooledBuffer<T> {
    fn deref_mut(&mut self) -> &mut Vec<T> {
        &mut self.buf
    }
}

impl<T: DType> Drop for PooledBuffer<T> {
    fn drop(&mut self) {
        if let Some(pool) = self.pool.take() {
            let buf = std::mem::take(&mut self.buf);
            pool.borrow_mut().recycle_buffer(buf);
        }
    }
}

impl<T: DType> BufferPool<T> {
    pub fn new() -> Self {
        BufferPool { pool: Vec::new() }
    }

    /// Grab a Vec with at least `capacity`. Clears and reuses one from the pool if available.
    pub fn get_buffer(&mut self, capacity: usize) -> Vec<T> {
        // Find the smallest buf that can fit this capacity.
        let mut smallest_found_buf = None;
        for i in 0..self.pool.len() {
            let this_buf_len = self.pool[i].len();

            let found_smaller_buf = smallest_found_buf
                .is_some_and(|found: usize| self.pool[found].len() < this_buf_len);
            let found_useable_buf = smallest_found_buf.is_none() && this_buf_len >= capacity;
            if found_smaller_buf || found_useable_buf {
                smallest_found_buf = Some(i);
            }
        }

        if let Some(smallest_found_buf) = smallest_found_buf {
            let mut buf = self.pool.swap_remove(smallest_found_buf);
            buf.clear();
            buf.reserve(capacity);
            buf
        } else {
            Vec::with_capacity(capacity)
        }
    }

    /// Return a Vec back into the pool for reuse.
    pub fn recycle_buffer(&mut self, buf: Vec<T>) {
        self.pool.push(buf);
    }
}
