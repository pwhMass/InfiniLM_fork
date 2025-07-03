use crate::{BaseArgs, macros::print_now, progress_bar};
use llama_cu::{Message, Received, Service, Session, SessionId, TextBuf};
use log::info;
use std::time::{Duration, Instant};

#[derive(Args)]
pub struct GenerateArgs {
    #[clap(flatten)]
    base: BaseArgs,
    #[clap(short, long)]
    prompt: Option<String>,
    #[clap(short = 't', long)]
    use_template: bool,
}

impl GenerateArgs {
    pub fn generate(self) {
        let Self {
            base,
            prompt,
            use_template,
        } = self;
        let gpus = base.gpus();
        let max_steps = base.max_steps();
        let sample_args = base.sample_args();
        let mut prompt = prompt.unwrap_or("Once upon a time,".into());

        let mut service = Service::new(base.model, &gpus, !base.no_cuda_graph);
        progress_bar(&mut service);

        let term = service.terminal();

        if use_template {
            prompt = term.render(&[Message::user(&prompt)])
        }
        print_now!("{prompt}");

        let session = Session {
            id: SessionId(0),
            sample_args,
            cache: term.new_cache(),
        };
        let tokens = term.tokenize(&prompt);

        term.start(session, &tokens, max_steps);

        let mut prefill = Duration::ZERO;
        let mut decode = Duration::ZERO;
        let mut ntoks = 0;
        let mut buf = TextBuf::new();
        loop {
            let time = Instant::now();
            let Received { sessions, outputs } = service.recv(Duration::from_millis(50));
            if prefill.is_zero() {
                prefill = time.elapsed()
            } else {
                decode += time.elapsed()
            }

            for (_, tokens) in outputs {
                ntoks += tokens.len();
                print_now!("{}", service.terminal().decode(&tokens, &mut buf))
            }
            if !sessions.is_empty() {
                break;
            }
        }
        println!();
        info!("prefill = {prefill:?}, decode = {decode:?}");
        info!(
            "n toks = {ntoks}, perf: {:?}/tok, {}tok/s",
            decode / ntoks as _,
            ntoks as f64 / decode.as_secs_f64()
        )
    }
}
