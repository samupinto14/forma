fn main() {
    println!("cargo:rerun-if-changed=assets/forma.ico");
    if std::env::var("CARGO_CFG_TARGET_OS").map(|t| t == "windows").unwrap_or(false) {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/forma.ico");
        res.set("ProductName", "Forma");
        res.set("FileDescription", "Forma asset reviewer");
        if let Err(e) = res.compile() {
            println!("cargo:warning=could not embed icon: {e}");
        }
    }
}
