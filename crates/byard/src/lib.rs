//! The `byard` façade crate: the single dependency an application adds to write
//! the Rust half of a two-layer Byard app (RFC-0001 §1, RFC-0028 §2).
//!
//! It re-exports the controller boundary from `byard-core` and the
//! `#[byard_controller]` attribute from `byard-macro`, so the macro's generated
//! code — which references `::byard::bridge::*` — resolves against the *app's*
//! dependency graph (`app → byard → byard-core`, a forward edge; INV-1 holds:
//! `byard-macro` still has no dependency on core/compiler).
//!
//! ```ignore
//! use byard::bridge::HostValue;
//!
//! #[byard::byard_controller]
//! #[derive(Clone)]
//! struct WeatherApi { base_url: String }
//!
//! #[byard::byard_controller]
//! impl WeatherApi {
//!     async fn forecast(&self, city: String) -> Result<HostValue, HostValue> {
//!         Ok(HostValue::Record(vec![("city".into(), HostValue::Str(city))]))
//!     }
//! }
//! ```

/// The controller boundary: [`HostValue`](bridge::HostValue), the
/// [`Controller`](bridge::Controller) trait, [`ControllerRegistry`](bridge::ControllerRegistry),
/// the `From`/`IntoHostValue` conversions, and the reply/timer wire types.
pub use byard_core::bridge;

/// Marks a Rust struct as a byard controller (field metadata) and, on its `impl`
/// block, generates the async-method dispatch shim (`impl Controller`).
pub use byard_macro::byard_controller;
