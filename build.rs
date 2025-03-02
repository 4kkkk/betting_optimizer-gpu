use std::process::Command;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=src/cuda/kernel.cu");

    let kernel_path = Path::new("src/cuda/kernel.cu");
    let ptx_path = Path::new("src/cuda/kernel.ptx");

    Command::new("nvcc")
        .args(&[
            "--ptx",
            "-arch=sm_86",    // Правильно для RTX 3060
            "-O3",            // Включаем максимальную оптимизацию
            "--threads", "0", // Использовать все доступные потоки процессора для компиляции
            kernel_path.to_str().unwrap(),
            "-o",
            ptx_path.to_str().unwrap(),
        ])
        .status()
        .unwrap();
}
