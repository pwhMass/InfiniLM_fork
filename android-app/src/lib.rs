use std::{
    ffi::CString,
    fs::File,
    path::{self, PathBuf},
};

// use crate::{print_now, InferenceArgs, Task};
use causal_lm::CausalLM;
use log::debug;
use ndk::asset::AssetManager;
use service::Service;

fn init_logger() {
    android_logger::init_once(
        android_logger::Config::default()
            .with_max_level(log::LevelFilter::Trace)
            .with_filter(
                android_logger::FilterBuilder::new()
                    .filter_level(log::LevelFilter::Debug)
                    .build(),
            ),
    );
}

#[no_mangle]
fn android_main(app: slint::android::AndroidApp) {
    init_logger();
    debug!("1111111111111111111111111111111111111111111111111");
    let mut asset_manager = app.asset_manager();
    test_func2(&mut asset_manager);
    debug!("22222222222222222222222222222222222222222222222");
    print!("3333333333333333");
    test_func();
    slint::android::init(app).unwrap();

    // ... rest of your code ...
    slint::slint! {
        export component MainWindow inherits Window {
            Text { text: "Hello World"; }
        }
    }
    MainWindow::new().unwrap().run().unwrap();
}

fn test_func() {
    use colored::{Color, Colorize};
    use std::{io::Write, iter::zip};
    use tokio::{runtime::Builder, task::JoinSet};

    // let Some(model_dir) = common::test_model::find() else {
    //     return;
    // };

    let model_dir = PathBuf::from("/sdcard/minillama_F16");

    debug!("model_dir: {}", model_dir.display());

    let config = File::open(model_dir.join(PathBuf::from("config.json")));
    debug!("config: {:?}", config);

    let runtime = Builder::new_current_thread().build().unwrap();
    let _rt = runtime.enter();

    let (service, _handle) = Service::<llama_cpu::Transformer>::load(model_dir, ());
    debug!("load finished");
    let mut set = JoinSet::new();
    let tasks = vec![
        ("Say \"Hi\" to me.", Color::Yellow),
        ("Hi", Color::Red),
        ("Where is the capital of France?", Color::Green),
    ];

    let sessions = tasks.iter().map(|_| service.launch()).collect::<Vec<_>>();

    for ((prompt, color), mut session) in zip(tasks, sessions) {
        set.spawn(async move {
            session.extend([prompt]);
            let mut busy = session.chat();
            while let Some(s) = busy.decode().await {
                debug!("{}", s.color(color));
                std::io::stdout().flush().unwrap();
            }
        });
    }

    runtime.block_on(async { while set.join_next().await.is_some() {} });
    runtime.shutdown_background();
}

fn test_func2(asset_manager: &mut AssetManager) {
    use std::io::Read;
    let mut asset = asset_manager
        .open(&CString::new("test.txt").unwrap())
        .unwrap();
    // let mut buffer = Vec::new();
    let test_txt = asset
        .bytes()
        .map(|x| char::from(x.unwrap()))
        .collect::<Vec<_>>();
    debug!("{:?}", test_txt);

    // let mut my_dir = asset_manager
    //     .open_dir(&CString::new("my_dir").unwrap())
    //     .expect("Could not open directory");

    // // Use it as an iterator
    // let all_files = my_dir.collect::<Vec<CString>>();

    // // Reset the iterator
    // my_dir.rewind();

    // // Use .with_next() to iterate without allocating `CString`s
    // while let Some(asset) = my_dir.with_next(|cstr| asset_manager.open(cstr).unwrap()) {
    //     let mut text = String::new();
    //     asset.read_to_string(&mut text);
    //     // ...
    // }
}
