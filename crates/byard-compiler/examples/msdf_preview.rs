//! Generates an MSDF field from an SVG icon and saves it as a PNG, so the
//! field can be inspected visually (RFC-0009 §2, §5).
//!
//! Run with the bundled gear icon:
//!
//! ```sh
//! cargo run -p byard-compiler --example msdf_preview
//! ```
//!
//! Or point it at any other flat, monochrome SVG:
//!
//! ```sh
//! cargo run -p byard-compiler --example msdf_preview -- path/to/icon.svg out.png
//! ```

use byard_compiler::vector::{GRID_SIZE, PX_RANGE, generate};
use byard_compiler::{CompileError, Span};

fn main() {
    let mut args = std::env::args().skip(1);
    let svg_path = args.next().unwrap_or_else(|| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/svg/gear.svg").to_string()
    });
    let out_path = args
        .next()
        .unwrap_or_else(|| "msdf_preview.png".to_string());

    let svg_bytes = std::fs::read(&svg_path).unwrap_or_else(|e| {
        eprintln!("failed to read {svg_path}: {e}");
        std::process::exit(1);
    });

    let glyph = match generate(&svg_bytes, GRID_SIZE, PX_RANGE, Span::new(0, 0)) {
        Ok(glyph) => glyph,
        Err(err) => {
            print_error(&err, &svg_path);
            std::process::exit(1);
        }
    };

    let image = image::RgbaImage::from_raw(glyph.width, glyph.height, glyph.bitmap)
        .expect("generator returns a well-formed RGBA8 buffer");
    image.save(&out_path).expect("failed to write PNG");

    println!(
        "wrote a {}x{} MSDF field (px_range {}) from {svg_path} to {out_path}",
        glyph.width, glyph.height, glyph.px_range
    );
}

fn print_error(err: &CompileError, svg_path: &str) {
    eprintln!(
        "could not generate an MSDF field for {svg_path}: {}",
        err.headline()
    );
}
