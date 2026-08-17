//! ESM-2 33-token alphabet. Sequence -> token ids with <cls> prepended and
//! <eos> appended (matches the HF esm tokenizer for standard sequences).

pub const VOCAB: [&str; 33] = [
    "<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K",
    "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O", ".", "-", "<null_1>", "<mask>",
];

pub const CLS: i64 = 0;
pub const PAD: i64 = 1;
pub const EOS: i64 = 2;
pub const UNK: i64 = 3;
pub const MASK: i64 = 32;

pub fn aa_to_idx(c: char) -> i64 {
    match c.to_ascii_uppercase() {
        'L' => 4, 'A' => 5, 'G' => 6, 'V' => 7, 'S' => 8, 'E' => 9, 'R' => 10, 'T' => 11,
        'I' => 12, 'D' => 13, 'P' => 14, 'K' => 15, 'Q' => 16, 'N' => 17, 'F' => 18, 'Y' => 19,
        'M' => 20, 'H' => 21, 'W' => 22, 'C' => 23, 'X' => 24, 'B' => 25, 'U' => 26, 'Z' => 27,
        'O' => 28, '.' => 29, '-' => 30, _ => UNK,
    }
}

/// Tokenize: [<cls>] + residues + [<eos>].
pub fn tokenize(seq: &str) -> Vec<i64> {
    let mut ids = Vec::with_capacity(seq.len() + 2);
    ids.push(CLS);
    for c in seq.chars() {
        ids.push(aa_to_idx(c));
    }
    ids.push(EOS);
    ids
}
