use crate::models::settings::Settings;
use std::fs::File;
use std::io::{self, Read, Write};

pub fn save_settings(settings: &Settings, path: &str) -> io::Result<()> {
    let json = serde_json::to_string_pretty(settings)?;
    let mut file = File::create(path)?;
    file.write_all(json.as_bytes())?;
    Ok(())
}

pub fn load_settings(path: &str) -> io::Result<Settings> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let settings = serde_json::from_str(&contents)?;
    Ok(settings)
}
