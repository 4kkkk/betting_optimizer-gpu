mod gui;
mod models;
mod optimization;
mod utils;

use crate::gui::OptimizationApp;
use eframe::egui::ViewportBuilder;

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: ViewportBuilder::default()
            .with_inner_size([800.0, 600.0])
            .with_min_inner_size([600.0, 400.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Оптимизация ставок",
        options,
        Box::new(|_cc| Ok(Box::new(OptimizationApp::new()))),
    )
}
