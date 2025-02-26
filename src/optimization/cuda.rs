use rustacuda::device::DeviceAttribute;
use rustacuda::prelude::*;



pub fn check_cuda_availability() -> bool {
    match rustacuda::init(CudaFlags::empty()) {
        Ok(_) => {
            if let Ok(device) = Device::get_device(0) {
                println!("CUDA доступна:");
                println!("GPU: {}", device.name().unwrap_or_default());
                println!(
                    "Compute Capability: {}.{}",
                    device
                        .get_attribute(DeviceAttribute::ComputeCapabilityMajor)
                        .unwrap_or(0),
                    device
                        .get_attribute(DeviceAttribute::ComputeCapabilityMinor)
                        .unwrap_or(0)
                );
                true
            } else {
                println!("CUDA устройства не найдены");
                false
            }
        }
        Err(e) => {
            println!("CUDA недоступна: {}", e);
            false
        }
    }
}
