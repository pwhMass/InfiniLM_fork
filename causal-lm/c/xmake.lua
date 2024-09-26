add_rules("mode.debug", "mode.release")
set_encodings("utf-8")
set_warnings("all")
set_languages("c11")

target("causal-lm")
    set_kind("static")
    add_files("causal-lm.c")
