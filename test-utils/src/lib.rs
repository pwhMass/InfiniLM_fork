use gguf::{
    ext::{utok, Mmap},
    map_files, Tokenizer,
};
use std::{
    fmt,
    path::Path,
    time::{Duration, Instant},
    vec,
};

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

pub trait CausalLM<B> {
    fn infer(&mut self, tokens: &[utok], cache: &mut [B], pos: usize) -> utok;
}

pub fn test_infer<B>(
    mut lm: impl CausalLM<B>,
    cache: &mut [B],
    eos: utok,
    tokenizer: Tokenizer,
    prompt: &str,
) {
    use cli_table::{format::Justify, print_stdout, Cell, CellStruct, Style, Table};
    macro_rules! print_now {
        ($($arg:tt)*) => {{
            use std::io::Write;

            print!($($arg)*);
            std::io::stdout().flush().unwrap();
        }};
    }

    print_now!("{prompt}");

    let mut tokens = tokenizer.encode(prompt);
    let num_prompt_tokens = tokens.len();

    let mut prefill = Duration::ZERO;
    let mut decode = Duration::ZERO;

    let mut pos = 0;
    loop {
        let time = Instant::now();
        let next = lm.infer(&tokens, cache, pos);
        let time = time.elapsed();

        if prefill.is_zero() {
            prefill = time;
        } else {
            decode += time;
        }

        pos += tokens.len();
        if next == eos {
            break;
        }

        let piece = tokenizer.decode(next);
        print_now!("{piece}");
        tokens = vec![next];
    }

    let table = [
        row("total", prefill + decode, pos),
        row("prefill", prefill, num_prompt_tokens),
        row("decode", decode, pos - num_prompt_tokens),
    ]
    .table()
    .title(
        ["\\", "num tokens", "elapse", "time per token"]
            .into_iter()
            .map(|s| cell(s).bold(true)),
    )
    .bold(true);

    println!();
    println!();
    assert!(print_stdout(table).is_ok());

    fn cell(x: impl fmt::Display) -> CellStruct {
        x.cell().justify(Justify::Center)
    }
    fn row(name: &str, time: Duration, n: usize) -> [CellStruct; 4] {
        [
            cell(name).bold(true),
            cell(n),
            cell(format!("{:.3?}", time)),
            cell(format!("{:.3?}", time.div_f64(n as _))),
        ]
    }
}
