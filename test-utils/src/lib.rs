use gguf::{ext::Mmap, map_files};
use std::path::Path;

pub fn map_gguf_files() -> Option<Box<[Mmap]>> {
    let Some(path) = std::env::var_os("TEST_MODEL") else {
        println!("TEST_MODEL not set");
        return None;
    };
    let path = Path::new(&path);
    if !path.is_file() {
        println!("{path:?} not found");
        return None;
    }
    Some(map_files(path))
}

#[macro_export]
macro_rules! print_now {
    ($($arg:tt)*) => {{
        use std::io::Write;

        print!($($arg)*);
        std::io::stdout().flush().unwrap();
    }};
}
