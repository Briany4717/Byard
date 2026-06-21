//! Interned identifiers (RFC-0002 §"Data structures").
//!
//! A [`Symbol`] is an `Arc<str>` drawn from a single process-global,
//! content-addressed interner (an append-only `HashSet<Arc<str>>`). Interning
//! gives two properties the rest of the compiler relies on:
//!
//! - **Pointer-equality comparison.** Equal text always interns to the same
//!   `Arc`, so [`Symbol`] equality is an `Arc::ptr_eq` — the fast path the
//!   hot-reload structural diff (RFC-0002 §"Hot-reload boundary") needs.
//! - **`Send`.** `CompiledView` must cross the file-watcher → logic-thread
//!   channel (RFC-0002 §"Integration with Engine", INV-6), and every `Symbol`
//!   in its AST has to satisfy that bound. `Arc<str>` is `Send`; `Rc<str>`
//!   would not be.
//!
//! Identity is stable across reparses *by construction*: the interner is keyed
//! on content, not encounter order, so the same source text always yields a
//! `Symbol` that compares equal to one interned in a previous parse.

use std::collections::HashSet;
use std::sync::{Arc, LazyLock, RwLock};

/// A content-addressed, interned identifier.
///
/// Cheap to clone (an `Arc` bump) and cheap to compare (`Arc::ptr_eq`).
#[derive(Clone, Debug, Eq)]
pub struct Symbol(Arc<str>);

/// The single process-global interner. Append-only: entries are never removed,
/// so a `Symbol`'s backing `Arc` stays valid (and its identity stable) for the
/// life of the process. The lock is only taken to *insert* new text; repeated
/// interning of already-seen text takes the shared read path.
static INTERNER: LazyLock<RwLock<HashSet<Arc<str>>>> =
    LazyLock::new(|| RwLock::new(HashSet::new()));

impl Symbol {
    /// Interns `s`, returning the canonical [`Symbol`] for that text.
    ///
    /// Two calls with equal text return [`Symbol`]s that share one `Arc`, so
    /// they compare equal by pointer.
    #[must_use]
    pub fn intern(s: &str) -> Self {
        // Fast path: the text is almost always already interned.
        {
            let read = INTERNER.read().expect("symbol interner poisoned");
            if let Some(arc) = read.get(s) {
                return Self(Arc::clone(arc));
            }
        }

        // Slow path: insert under the write lock, re-checking in case another
        // thread inserted the same text between the two locks.
        let mut write = INTERNER.write().expect("symbol interner poisoned");
        if let Some(arc) = write.get(s) {
            Self(Arc::clone(arc))
        } else {
            let arc: Arc<str> = Arc::from(s);
            write.insert(Arc::clone(&arc));
            Self(arc)
        }
    }

    /// Returns the interned text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl PartialEq for Symbol {
    /// Pointer equality — sound because interning guarantees one `Arc` per
    /// distinct text.
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl std::hash::Hash for Symbol {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Hash the identity (pointer), consistent with pointer equality.
        (Arc::as_ptr(&self.0).cast::<()>() as usize).hash(state);
    }
}

impl std::fmt::Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// A `Symbol` is `Arc<str>`-sized (two pointers, fat-pointer to the str slice).
const _: () = {
    assert!(
        std::mem::size_of::<Symbol>() <= 16,
        "Symbol exceeded its 16-byte budget"
    );
};

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-time proof that `Symbol` is `Send` (INV-6): if it were not, this
    /// would fail to type-check.
    #[test]
    fn symbol_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<Symbol>();
    }

    #[test]
    fn same_text_interns_to_same_arc() {
        let a = Symbol::intern("Column");
        let b = Symbol::intern("Column");
        assert!(Arc::ptr_eq(&a.0, &b.0), "equal text must share one Arc");
        assert_eq!(a, b);
    }

    #[test]
    fn identity_stable_across_reinterning() {
        // Interning unrelated text in between must not disturb identity — this
        // is the property hot-reload diffing depends on.
        let first = Symbol::intern("clicks");
        let _ = Symbol::intern("unrelated");
        let _ = Symbol::intern("another");
        let again = Symbol::intern("clicks");
        assert_eq!(first, again);
        assert!(Arc::ptr_eq(&first.0, &again.0));
    }

    #[test]
    fn different_text_differs() {
        let a = Symbol::intern("Row");
        let b = Symbol::intern("Column");
        assert_ne!(a, b);
        assert!(!Arc::ptr_eq(&a.0, &b.0));
    }

    #[test]
    fn as_str_round_trips() {
        assert_eq!(Symbol::intern("inject").as_str(), "inject");
    }
}
