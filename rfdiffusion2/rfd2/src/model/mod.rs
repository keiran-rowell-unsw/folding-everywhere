//! The RFdiffusion2 network — `rf2aa`'s `LegacyRoseTTAFoldModule` and every
//! layer under it.
//!
//! Structure mirrors upstream so the two can be read side by side:
//!
//! | Rust | reference |
//! |---|---|
//! | `attention` | `rf2aa/model/layers/Attention_module.py` |
//! | `embeddings` | `rf2aa/model/layers/Embeddings.py` (+ `PositionalEncoding2D`) |
//! | `track` | `rf2aa/model/Track_module.py` |
//! | `aux` | `rf2aa/model/layers/AuxiliaryPredictor.py` |
//! | `se3` | `rf2aa/SE3Transformer/` + the e3nn bases |

pub mod attention;
pub mod aux;
pub mod embeddings;
pub mod iterblock;
pub mod rf;
pub mod se3;
pub mod str2str;
pub mod track;
pub mod xyzconv;
