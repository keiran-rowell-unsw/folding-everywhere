//! Reader for PyTorch `.pt` / `pytorch_model.bin` files (a ZIP64 archive of
//! uncompressed raw tensor storages + a pickle index). Lets the app use the
//! official ProteinMPNN `vanilla_model_weights/v_48_020.pt` checkpoints directly
//! — no Python, no conversion step.
//!
//! ProteinMPNN checkpoints are a *nested* dict:
//!   `{'num_edges': 48, 'noise_level': 0.2, 'model_state_dict': OrderedDict(...)}`
//! The pickle machine below walks the whole structure and simply records every
//! (str -> tensor) pair it sees, so the inner state-dict entries come out under
//! their plain parameter names.
//!
//! Returns, for each parameter, its name, dtype, shape, and the absolute byte
//! range of its raw little-endian storage within the file (mmap-friendly).
//! Parsing is pure computation, so behaviour is identical on every OS.

#[derive(Debug, Clone)]
pub struct PthEntry {
    pub name: String,
    pub dtype: String, // "F32" | "F16" | "I64"
    pub shape: Vec<usize>,
    pub start: usize,
    pub end: usize,
}

// ---- little-endian readers ------------------------------------------------
fn u16le(b: &[u8], o: usize) -> usize {
    u16::from_le_bytes([b[o], b[o + 1]]) as usize
}
fn u32le(b: &[u8], o: usize) -> u64 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) as u64
}
fn u64le(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

/// Parse the (ZIP64) central directory -> entries. Returns map name -> data byte offset & size.
fn parse_zip(b: &[u8]) -> std::collections::HashMap<String, (usize, usize)> {
    let n = b.len();
    // find End-Of-Central-Directory (sig 0x06054b50), scanning backwards
    let mut eocd = None;
    let lo = n.saturating_sub(65557); // max comment 65535 + 22
    for i in (lo..=n - 22).rev() {
        if u32le(b, i) == 0x0605_4b50 {
            eocd = Some(i);
            break;
        }
    }
    let eocd = eocd.expect("zip: no EOCD");
    let mut cd_off = u32le(b, eocd + 16);
    let mut total = u16le(b, eocd + 10) as u64;

    // ZIP64?
    if cd_off == 0xFFFF_FFFF || total == 0xFFFF {
        // ZIP64 EOCD locator sits 20 bytes before EOCD
        let loc = eocd - 20;
        assert_eq!(u32le(b, loc), 0x0706_4b50, "zip64 locator");
        let z64 = u64le(b, loc + 8) as usize; // offset of ZIP64 EOCD record
        assert_eq!(u32le(b, z64), 0x0606_4b50, "zip64 eocd");
        total = u64le(b, z64 + 32);
        cd_off = u64le(b, z64 + 48);
    }

    let mut map = std::collections::HashMap::new();
    let mut p = cd_off as usize;
    for _ in 0..total {
        assert_eq!(u32le(b, p), 0x0201_4b50, "central dir sig");
        let method = u16::from_le_bytes([b[p + 10], b[p + 11]]);
        let mut size = u32le(b, p + 24); // uncompressed size
        let name_len = u16le(b, p + 28);
        let extra_len = u16le(b, p + 30);
        let comment_len = u16le(b, p + 32);
        let mut local_off = u32le(b, p + 42);
        let name = String::from_utf8_lossy(&b[p + 46..p + 46 + name_len]).into_owned();
        // ZIP64 extra field (0x0001) supplies real 64-bit values for 0xFFFFFFFF markers
        let extra = &b[p + 46 + name_len..p + 46 + name_len + extra_len];
        let mut e = 0;
        while e + 4 <= extra.len() {
            let id = u16le(extra, e);
            let dsz = u16le(extra, e + 2);
            if id == 0x0001 {
                let mut q = e + 4;
                if size == 0xFFFF_FFFF {
                    size = u64le(extra, q);
                    q += 8;
                }
                if u32le(b, p + 20) == 0xFFFF_FFFF {
                    q += 8; // compressed size (skip)
                }
                if local_off == 0xFFFF_FFFF {
                    local_off = u64le(extra, q);
                }
            }
            e += 4 + dsz;
        }
        // read local header to compute data offset (its name/extra lens can differ)
        let lo = local_off as usize;
        assert_eq!(u32le(b, lo), 0x0403_4b50, "local header sig");
        let lname = u16le(b, lo + 26);
        let lextra = u16le(b, lo + 28);
        let data_off = lo + 30 + lname + lextra;
        assert_eq!(method, 0, "zip entry {name} is compressed (expected STORED)");
        map.insert(name, (data_off, size as usize));
        p += 46 + name_len + extra_len + comment_len;
    }
    map
}

// ---- minimal pickle machine (subset used by torch state_dicts) ------------
#[derive(Clone)]
enum V {
    Int(i64),
    Float,
    Str(String),
    Global(String),
    Bool,
    None,
    Tup(Vec<V>),
    Mark,
    Pers { stype: String, key: String },
    Tensor { key: String, stype: String, offset: usize, shape: Vec<usize> },
}

/// Returns (name, storage_key, storage_type, storage_offset, shape) per tensor.
fn parse_pickle(b: &[u8]) -> Vec<(String, String, String, usize, Vec<usize>)> {
    let mut st: Vec<V> = Vec::new();
    // Memo table. Protocol 2 addresses it explicitly (BINPUT/LONG_BINPUT); protocol
    // 4's MEMOIZE appends at the next free slot, so this must start empty.
    let mut memo: Vec<V> = Vec::new();
    let mut out: Vec<(String, String, String, usize, Vec<usize>)> = Vec::new();
    let mut i = 0;
    let put = |memo: &mut Vec<V>, id: usize, v: V| {
        if id >= memo.len() {
            memo.resize(id + 1, V::None);
        }
        memo[id] = v;
    };
    macro_rules! pop { () => { st.pop().unwrap() } }
    while i < b.len() {
        let op = b[i];
        i += 1;
        match op {
            0x80 => i += 1,                       // PROTO
            b'}' => st.push(V::Tup(vec![])),      // EMPTY_DICT (treated as generic)
            b']' => st.push(V::Tup(vec![])),      // EMPTY_LIST
            b'q' => { let id = b[i] as usize; i += 1; put(&mut memo, id, st.last().unwrap().clone()); }
            b'r' => { let id = u32le(b, i) as usize; i += 4; put(&mut memo, id, st.last().unwrap().clone()); }
            b'h' => { let id = b[i] as usize; i += 1; st.push(memo[id].clone()); }
            b'j' => { let id = u32le(b, i) as usize; i += 4; st.push(memo[id].clone()); }
            b'(' => st.push(V::Mark),             // MARK
            b'X' => { let l = u32le(b, i) as usize; i += 4; let s = String::from_utf8_lossy(&b[i..i + l]).into_owned(); i += l; st.push(V::Str(s)); }
            0x8c => { let l = b[i] as usize; i += 1; let s = String::from_utf8_lossy(&b[i..i + l]).into_owned(); i += l; st.push(V::Str(s)); } // SHORT_BINUNICODE
            0x8d => { let l = u64le(b, i) as usize; i += 8; let s = String::from_utf8_lossy(&b[i..i + l]).into_owned(); i += l; st.push(V::Str(s)); } // BINUNICODE8
            b'G' => { let v = f64::from_be_bytes(b[i..i + 8].try_into().unwrap()); i += 8; let _ = v; st.push(V::Float); } // BINFLOAT
            0x94 => { let v = st.last().unwrap().clone(); memo.push(v); }  // MEMOIZE (proto 4)
            b'd' => { // DICT: pop pairs back to MARK
                let mut items = Vec::new();
                loop { match st.pop().unwrap() { V::Mark => break, x => items.push(x) } }
                items.reverse();
                let mut k = 0;
                while k + 1 < items.len() {
                    if let (V::Str(name), V::Tensor { key, stype, offset, shape }) = (&items[k], &items[k + 1]) {
                        out.push((name.clone(), key.clone(), stype.clone(), *offset, shape.clone()));
                    }
                    k += 2;
                }
                st.push(V::Tup(vec![]));
            }
            b'l' => { // LIST: pop back to MARK
                loop { match st.pop().unwrap() { V::Mark => break, _ => {} } }
                st.push(V::Tup(vec![]));
            }
            b'b' => { let _state = pop!(); }  // BUILD: object stays on the stack
            0x81 => { let _args = pop!(); let _cls = pop!(); st.push(V::None); } // NEWOBJ
            b'c' => { // GLOBAL: module\nname\n
                let s0 = i; while b[i] != b'\n' { i += 1; } let m = String::from_utf8_lossy(&b[s0..i]).into_owned(); i += 1;
                let s1 = i; while b[i] != b'\n' { i += 1; } let nme = String::from_utf8_lossy(&b[s1..i]).into_owned(); i += 1;
                st.push(V::Global(format!("{m} {nme}")));
            }
            b'K' => { st.push(V::Int(b[i] as i64)); i += 1; }               // BININT1
            b'M' => { st.push(V::Int(u16le(b, i) as i64)); i += 2; }        // BININT2
            b'J' => { st.push(V::Int(i32::from_le_bytes(b[i..i+4].try_into().unwrap()) as i64)); i += 4; } // BININT
            0x8a => { let l = b[i] as usize; i += 1; let mut v = 0i64; for k in 0..l { v |= (b[i+k] as i64) << (8*k); } i += l; st.push(V::Int(v)); } // LONG1
            0x88 => st.push(V::Bool),
            0x89 => st.push(V::Bool),
            b'N' => st.push(V::None),
            b')' => st.push(V::Tup(vec![])),       // EMPTY_TUPLE
            0x85 => { let a = pop!(); st.push(V::Tup(vec![a])); }
            0x86 => { let b2 = pop!(); let a = pop!(); st.push(V::Tup(vec![a, b2])); }
            0x87 => { let c = pop!(); let b2 = pop!(); let a = pop!(); st.push(V::Tup(vec![a, b2, c])); }
            b't' => { // TUPLE: pop to MARK
                let mut v = Vec::new();
                loop { match st.pop().unwrap() { V::Mark => break, x => v.push(x) } }
                v.reverse(); st.push(V::Tup(v));
            }
            b'Q' => { // BINPERSID: top is ('storage', Global, key, location, numel)
                if let V::Tup(t) = pop!() {
                    let stype = if let V::Global(g) = &t[1] { g.split(' ').last().unwrap().to_string() } else { String::new() };
                    let key = if let V::Str(s) = &t[2] { s.clone() } else { String::new() };
                    st.push(V::Pers { stype, key });
                }
            }
            b'R' => { // REDUCE: callable + args
                let args = pop!();
                let callable = pop!();
                let cname = if let V::Global(g) = &callable { g.clone() } else { String::new() };
                if cname.contains("_rebuild_tensor") {
                    if let V::Tup(a) = args {
                        // a = (storage(Pers), storage_offset(Int), size(Tup), stride(Tup), ...)
                        let (mut key, mut stype) = (String::new(), String::new());
                        if let V::Pers { stype: s, key: k } = &a[0] { key = k.clone(); stype = s.clone(); }
                        let offset = if let V::Int(n) = &a[1] { *n as usize } else { 0 };
                        let shape = if let V::Tup(sz) = &a[2] { sz.iter().map(|x| if let V::Int(n) = x { *n as usize } else { 0 }).collect() } else { vec![] };
                        st.push(V::Tensor { key, stype, offset, shape });
                    } else { st.push(V::None); }
                } else {
                    st.push(V::None); // e.g. OrderedDict() backward hooks
                }
            }
            b'u' => { // SETITEMS: pop pairs back to MARK into out
                let mut items = Vec::new();
                loop { match st.pop().unwrap() { V::Mark => break, x => items.push(x) } }
                items.reverse();
                let mut k = 0;
                while k + 1 < items.len() {
                    if let (V::Str(name), V::Tensor { key, stype, offset, shape }) = (&items[k], &items[k + 1]) {
                        out.push((name.clone(), key.clone(), stype.clone(), *offset, shape.clone()));
                    }
                    k += 2;
                }
            }
            b's' => { // SETITEM: val,key under dict
                let val = pop!(); let keyv = pop!();
                if let (V::Str(name), V::Tensor { key, stype, offset, shape }) = (&keyv, &val) {
                    out.push((name.clone(), key.clone(), stype.clone(), *offset, shape.clone()));
                }
            }
            b'a' => { let _ = pop!(); }   // APPEND (ignore)
            b'e' => { loop { match st.pop().unwrap() { V::Mark => break, _ => {} } } } // APPENDS
            b'.' => break,                // STOP
            other => panic!("pickle: unsupported opcode 0x{other:02x} at {}", i - 1),
        }
    }
    out
}

fn dtype_of(stype: &str) -> &'static str {
    match stype {
        "FloatStorage" => "F32",
        "HalfStorage" => "F16",
        "DoubleStorage" => "F64",
        "LongStorage" => "I64",
        "IntStorage" => "I32",
        "BFloat16Storage" => "BF16",
        other => panic!("pth: unsupported storage {other}"),
    }
}

fn elem_size(dtype: &str) -> usize {
    match dtype {
        "F64" | "I64" => 8,
        "F32" | "I32" => 4,
        "F16" | "BF16" => 2,
        _ => panic!("pth: size {dtype}"),
    }
}

/// Build the tensor index from a memory-mapped pytorch_model.bin.
pub fn index_pth(b: &[u8]) -> Vec<PthEntry> {
    let zip = parse_zip(b);
    // locate the pickle (archive/data.pkl); the archive prefix may vary
    let pkl_name = zip.keys().find(|k| k.ends_with("data.pkl")).expect("no data.pkl").clone();
    let (po, ps) = zip[&pkl_name];
    let prefix = &pkl_name[..pkl_name.len() - "data.pkl".len()]; // e.g. "archive/"
    let meta = parse_pickle(&b[po..po + ps]);
    let mut out = Vec::with_capacity(meta.len());
    for (name, key, stype, offset, shape) in meta {
        let dtype = dtype_of(&stype);
        let esz = elem_size(dtype);
        let storage_name = format!("{prefix}data/{key}");
        let (doff, _dsz) = zip[&storage_name];
        let numel: usize = shape.iter().product::<usize>().max(if shape.is_empty() { 1 } else { 0 });
        let start = doff + offset * esz;
        let end = start + numel * esz;
        out.push(PthEntry { name, dtype: dtype.to_string(), shape, start, end });
    }
    out
}

