use std::io::Cursor;
use std::path::Path;

#[cfg(not(target_os = "android"))]
pub fn load<P: AsRef<Path>>(path: P) -> Cursor<Vec<u8>> {
    use std::fs::File;
    use std::io::Read;

    let mut buf = Vec::new();
    let mut file = File::open(path).unwrap();
    file.read_to_end(&mut buf).unwrap();
    Cursor::new(buf)
}

#[cfg(target_os = "android")]
pub fn load<P: AsRef<Path>>(path: P) -> Cursor<Vec<u8>> {
    let mut filename: String = path
        .as_ref()
        .to_str()
        .expect("Can`t convert Path to &str")
        .into();
    if filename.starts_with("assets/") {
        filename = filename.replace("assets/", "");
    }

    match android_glue::load_asset(&filename) {
        Ok(buf) => Cursor::new(buf),
        Err(_) => panic!("Can`t load asset '{}'", filename),
    }
}
