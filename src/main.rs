
use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

mod gui;
mod models;
mod optimization;
mod utils;
mod debug_optimizer;

use crate::gui::OptimizationApp;
use eframe::egui::ViewportBuilder;

fn main() -> Result<(), eframe::Error> {
    // Повышаем приоритет процесса на Windows
    #[cfg(target_os = "windows")]
    unsafe {
        use windows_sys::Win32::System::Threading::{
            GetCurrentProcess, SetPriorityClass, HIGH_PRIORITY_CLASS,
        };
        let handle = GetCurrentProcess();
        SetPriorityClass(handle, HIGH_PRIORITY_CLASS);
    }

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