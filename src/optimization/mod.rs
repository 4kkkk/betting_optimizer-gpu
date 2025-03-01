pub mod params;
pub mod strategy;
pub mod results;
mod cuda;



pub use params::Params;
pub use results::OptimizationResult;
pub use strategy::optimize_parameters;
