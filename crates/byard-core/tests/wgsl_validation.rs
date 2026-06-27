//! Headless validation of every WGSL shader module (RFC-0009 M44).
//!
//! `naga` is the same shader front-end/validator `wgpu` uses internally, so a
//! green validation here predicts a clean `Device::create_shader_module` on the
//! GPU — caught in CI without a graphics device. M44 adds the fifth shader
//! (`vector_msdf.wgsl`); this test keeps all five honest from now on.

use naga::valid::{Capabilities, ValidationFlags, Validator};

/// The five pipeline shaders (RFC-0001 §3.1 + RFC-0009 §1's MSDF addition).
const SHADERS: &[(&str, &str)] = &[
    ("solid_box", include_str!("../src/encoder/solid_box.wgsl")),
    (
        "decorated_box",
        include_str!("../src/encoder/decorated_box.wgsl"),
    ),
    (
        "texture_sampler",
        include_str!("../src/encoder/texture_sampler.wgsl"),
    ),
    (
        "vector_msdf",
        include_str!("../src/encoder/vector_msdf.wgsl"),
    ),
];

#[test]
fn every_wgsl_module_parses_and_validates() {
    for (name, src) in SHADERS {
        let module = naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|e| panic!("`{name}.wgsl` failed to parse: {e:?}"));
        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("`{name}.wgsl` failed to validate: {e:?}"));
    }
}
