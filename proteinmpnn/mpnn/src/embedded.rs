//! The four published "vanilla" ProteinMPNN checkpoints, baked into the binary.
//!
//! Each is ~6.7 MB, so all four together add ~27 MB to the executable and remove
//! the companion-data-file problem entirely: `mpnn` and `mpnn_gui` run with no
//! download, no install, and no `--weights` argument.
//!
//! The name encodes the backbone-noise level the model was trained with:
//! `v_48_020` = 48 neighbours, 0.20 A Gaussian noise. More noise gives sequences
//! that are more tolerant of an imperfect backbone; `v_48_020` is the default in
//! the reference implementation and here.

/// `(name, bytes, training backbone noise in Angstrom)`.
pub const MODELS: &[(&str, &[u8], f32)] = &[
    ("v_48_002", include_bytes!("../../weights/v_48_002.pt"), 0.02),
    ("v_48_010", include_bytes!("../../weights/v_48_010.pt"), 0.10),
    ("v_48_020", include_bytes!("../../weights/v_48_020.pt"), 0.20),
    ("v_48_030", include_bytes!("../../weights/v_48_030.pt"), 0.30),
];

pub const DEFAULT_MODEL: &str = "v_48_020";

pub fn by_name(name: &str) -> Option<&'static [u8]> {
    MODELS.iter().find(|(n, _, _)| *n == name).map(|(_, b, _)| *b)
}

pub fn names() -> Vec<&'static str> {
    MODELS.iter().map(|(n, _, _)| *n).collect()
}

pub fn noise_of(name: &str) -> Option<f32> {
    MODELS.iter().find(|(n, _, _)| *n == name).map(|(_, _, v)| *v)
}
