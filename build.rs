use std::process::Command;
use std::path::Path;
use std::fs;
use std::env;

fn main() {
    // Указываем повторную компиляцию при изменении CUDA-файлов
    println!("cargo:rerun-if-changed=src/cuda/kernel.cu");

    let kernel_path = Path::new("src/cuda/kernel.cu");
    let ptx_path = Path::new("src/cuda/kernel.ptx");

    // Проверка наличия NVCC
    let nvcc_output = Command::new("nvcc")
        .arg("--version")
        .output();

    match nvcc_output {
        Ok(output) => {
            if output.status.success() {
                println!("NVCC найден, версия: {}", String::from_utf8_lossy(&output.stdout));

                // Получение информации о GPU для оптимальной компиляции
                let gpu_info = Command::new("nvidia-smi")
                    .args(&["--query-gpu=name,compute_cap", "--format=csv,noheader"])
                    .output();

                let mut compute_capability = "86"; // По умолчанию для RTX 3060

                if let Ok(info) = gpu_info {
                    if info.status.success() {
                        let gpu_info_str = String::from_utf8_lossy(&info.stdout);
                        println!("Обнаружены GPU: {}", gpu_info_str);

                        // Автоматическое определение compute capability
                        if gpu_info_str.contains("RTX 3060") || gpu_info_str.contains("8.6") {
                            compute_capability = "86";
                        } else if gpu_info_str.contains("RTX 40") || gpu_info_str.contains("8.9") {
                            compute_capability = "89";
                        } else if gpu_info_str.contains("RTX 20") || gpu_info_str.contains("7.5") {
                            compute_capability = "75";
                        }
                    }
                }

                // Создаем строку с архитектурой SM
                let arch_flag = format!("-arch=sm_{}", compute_capability);

                // Оптимизированные флаги компиляции для CUDA
                let mut nvcc_args = vec![
                    "--ptx",
                    &arch_flag, // Теперь берем ссылку на переменную с достаточным временем жизни
                    "-O3", // Максимальный уровень оптимизации
                    "--use_fast_math",  // Быстрая математика (может влиять на точность)
                    "-Xptxas=-v", // Вывод информации о регистрах и использовании памяти
                ];

                // Проверка режима отладки или релиза
                let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
                if profile == "release" {
                    // Дополнительные оптимизации для релизной сборки
                    nvcc_args.push("--ptxas-options=-O3");
                    nvcc_args.push("--generate-line-info");
                } else {
                    // Для отладки - меньше оптимизаций, больше информации
                    nvcc_args.push("-G");
                    nvcc_args.push("--device-debug");
                }

                // Добавляем пути к файлам
                nvcc_args.push(kernel_path.to_str().unwrap());
                nvcc_args.push("-o");
                nvcc_args.push(ptx_path.to_str().unwrap());

                println!("Компиляция CUDA с параметрами: {:?}", nvcc_args);

                let nvcc_build = Command::new("nvcc")
                    .args(&nvcc_args)
                    .status();

                match nvcc_build {
                    Ok(status) => {
                        if status.success() {
                            println!("CUDA скомпилирован успешно!");

                            // Проверяем размер PTX-файла
                            if let Ok(metadata) = fs::metadata(ptx_path) {
                                println!("Размер PTX: {} байт", metadata.len());
                            }
                        } else {
                            eprintln!("Ошибка компиляции CUDA: {:?}", status);
                        }
                    },
                    Err(e) => {
                        eprintln!("Не удалось запустить nvcc: {}", e);
                    }
                }
            } else {
                eprintln!("NVCC найден, но возникла ошибка при определении версии");
            }
        },
        Err(e) => {
            eprintln!("NVCC не найден: {}", e);
            eprintln!("Сборка продолжится без поддержки CUDA");

            // Создаем заглушку PTX-файла при отсутствии NVCC
            // чтобы избежать ошибок компиляции
            if !ptx_path.exists() {
                fs::write(ptx_path, "// Empty PTX file - NVCC not found")
                    .expect("Не удалось создать заглушку PTX-файла");
            }
        }
    }
}