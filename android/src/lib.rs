use android_logger::Config;
use jni::{
    objects::{JClass, JString},
    sys::jstring,
    JNIEnv,
};
use log::LevelFilter;
use service::Message;
use std::{
    path::PathBuf,
    sync::{
        mpsc::{Receiver, RecvError, Sender, TryRecvError},
        Mutex, Once, OnceLock,
    },
    time::Duration,
};
use tokio::sync::mpsc::{self, UnboundedSender};

type Service = service::Service<llama_cpu::Transformer>;
type Chat = (String, Sender<String>);

static COMMAND: OnceLock<UnboundedSender<Chat>> = OnceLock::new();
static DIALOG: OnceLock<Mutex<Option<Receiver<String>>>> = OnceLock::new();

#[no_mangle]
pub extern "system" fn Java_org_infinitensor_lm_Native_init(
    mut env: JNIEnv,
    _: JClass,
    model_path: JString,
) {
    android_logger::init_once(
        Config::default()
            .with_max_level(LevelFilter::Trace)
            .with_tag("Rust"),
    );

    static ONCE: Once = Once::new();
    if ONCE.is_completed() {
        panic!("Native library already initialized");
    }

    let model_dir: String = env
        .get_string(&model_path)
        .expect("Couldn't get java string!")
        .into();
    let model_dir = PathBuf::from(model_dir);

    if model_dir.is_dir() {
        ONCE.call_once(move || {
            std::thread::spawn(move || dispatch(model_dir));
        });
    } else {
        panic!("Model directory not found");
    }
}

#[no_mangle]
pub extern "system" fn Java_org_infinitensor_lm_Native_start(
    mut env: JNIEnv,
    _class: JClass,
    prompt: JString,
) {
    let prompt: String = env
        .get_string(&prompt)
        .expect("Couldn't get java string!")
        .into();
    let (sender, receiver) = std::sync::mpsc::channel();
    while COMMAND.get().is_none() {
        std::thread::sleep(Duration::from_millis(100));
    }
    COMMAND
        .get()
        .expect("Sender not initialized")
        .send((prompt, sender))
        .unwrap();
    DIALOG
        .get_or_init(Default::default)
        .lock()
        .unwrap()
        .replace(receiver);
}

#[no_mangle]
pub extern "system" fn Java_org_infinitensor_lm_Native_decode(
    env: JNIEnv,
    _class: JClass,
) -> jstring {
    let mut ans = String::new();
    let mut lock = DIALOG.get_or_init(Default::default).lock().unwrap();
    if let Some(receiver) = lock.as_mut() {
        loop {
            match receiver.try_recv() {
                Ok(s) => ans.push_str(&s),
                Err(TryRecvError::Empty) => match receiver.recv() {
                    Ok(s) => {
                        ans.push_str(&s);
                        break;
                    }
                    Err(RecvError) => {
                        log::error!("Receive disconnected");
                        lock.take();
                        break;
                    }
                },
                Err(TryRecvError::Disconnected) => {
                    log::error!("Try receive disconnected");
                    lock.take();
                    break;
                }
            }
        }
    }
    env.new_string(&ans)
        .expect("Couldn't create java string!")
        .into_raw()
}

fn dispatch(model_dir: PathBuf) {
    // 启动 tokio 运行时
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async move {
        let (service, _handle) = Service::load(model_dir, ());
        let (sender, mut receiver) = mpsc::unbounded_channel();
        COMMAND.get_or_init(move || sender);

        let mut session = service.launch();
        log::info!("session launched");
        while let Some((content, answer)) = receiver.recv().await {
            log::info!("chat: {content}");
            session.extend(&[Message {
                role: "user",
                content: &content,
            }]);
            let mut chat = session.chat();
            while let Some(piece) = chat.decode().await {
                log::info!("piece = {piece}");
                if answer.send(piece).is_err() {
                    log::warn!("send error");
                    break;
                }
            }
            log::info!("chat finished");
        }
    });
    // 关闭 tokio 运行时
    runtime.shutdown_background();
}
