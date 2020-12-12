use std::ops::Range;

use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use rayon::prelude::*;

macro_rules! bits {
    ($e:expr) => {
        !((!0) << $e)
    };
}

fn arm_crc32x(data: u64) -> u32 {
    const POLY: u32 = 0xEDB88320;
    let mut crc: u32 = 0;
    for i in 0..64 {
        crc ^= (data >> i) as u32 & 1;
        crc = (crc >> 1) ^ (POLY * (crc & 1));
    }

    crc
}

fn signature(addr: u64, key: u64) -> u64 {
    (arm_crc32x(addr.wrapping_mul(key)) & bits!(24)) as u64
}

/// Computes the inverse of an odd number modulo 2^64
fn mod_inv(u: u64) -> u64 {
    debug_assert_eq!(u & 1, 1);

    let mut v = u;
    for _ in 0..5 {
        v = v.wrapping_mul(2u64.wrapping_sub(v.wrapping_mul(u)));
    }

    v
}

#[derive(Copy, Clone, Debug)]
struct Target {
    addr: u64,
    signature: u64,
    inv: u64,
    inv_shift: u32,
}

impl Target {
    pub fn from_signed(signed: u64) -> Target {
        let addr = signed & bits!(40);
        assert_ne!(addr, 0);
        let signature = signed >> 40;

        // We shift the address right just as much to make it odd and then compute the modular
        // inverse of that. This allows us to quickly find one or all inverses of a multiplication
        // with addr. If inverse is only unique if addr is odd, as otherwise MSBs of the other
        // factor are lost when multiplying with addr.
        let inv_shift = addr.trailing_zeros();
        let inv = mod_inv(addr >> inv_shift);
        Target {
            addr,
            signature,
            inv,
            inv_shift,
        }
    }

    pub fn matches_key(&self, key: u64) -> bool {
        signature(self.addr, key) == self.signature
    }
}

struct CrcFixup {
    input_tab: [u64; 64],
    output_tab: [u64; 64],
}

impl Default for CrcFixup {
    fn default() -> Self {
        let mut rv = CrcFixup {
            input_tab: [0u64; 64],
            output_tab: [0u64; 64],
        };

        // Create a block matrix where the left block has a row for each one-hot 64-bit input and
        // the right block has a corresponding row of the truncated CRC output.
        // Every input-output pair is a linear combination of these.

        for i in 0..64 {
            rv.input_tab[i] = 1 << i;
            rv.output_tab[i] = (arm_crc32x(rv.input_tab[i]) & bits!(24)) as u64;
        }

        // Next we perform Gauss-Jordan elimination, so all output columns and the 40 LSBs input
        // columns have a single bit set. If we have an input-output pair, this allows us to easily
        // flip any of those bits, adjusting the 24 MSB of the input. We can use this to a) change
        // the 24 MSBs of the input to fit a given truncated crc or b) change exactly one of the 40
        // LSBs and some of the 24 MSBs without changing the resulting CRC.

        macro_rules! tab {
            ($r:expr, $c:expr) => {{
                let r = $r;
                let c = $c;
                if c < 24 {
                    rv.output_tab[r] & (1 << c) != 0
                } else {
                    rv.input_tab[r] & (1 << (c - 24)) != 0
                }
            }};
        }

        for i in 0..64 {
            for j in i..64 {
                if tab![j, i] {
                    rv.input_tab.swap(i, j);
                    rv.output_tab.swap(i, j);
                    break;
                }
            }

            assert!(tab![i, i]);

            for j in 0..64 {
                if i != j && tab![j, i] {
                    rv.input_tab[j] ^= rv.input_tab[i];
                    rv.output_tab[j] ^= rv.output_tab[i];
                }
            }
        }

        rv
    }
}

impl CrcFixup {
    #[cfg(test)]
    pub fn print_table(&self) {
        for i in 0..64 {
            println!(
                "{:2} {:064b} {:024b}",
                i, self.input_tab[i], self.output_tab[i]
            );
        }
    }

    pub fn fixup_key(&self, key: u64, target: Target) -> u64 {
        // Compute the key-addr product whose CRC has to match the signature
        let mut product = key.wrapping_mul(target.addr);

        // Compute the delta between the current CRC and the signature
        let syndrome = (target.signature ^ arm_crc32x(product) as u64) & bits!(24);

        // For each wrong bit, flip some of the 24 MSBs of the product such that the corresponding
        // CRC bit is flipped.
        for i in 0..24 {
            if syndrome & (1 << i) != 0 {
                product ^= self.input_tab[i];
            }
        }

        // To get back to the key (or one of them when addr is even), we need to undo the
        // multiplication with the address, see Target::from_signed above.
        let fixed_key = (product >> target.inv_shift).wrapping_mul(target.inv);

        assert!(target.matches_key(fixed_key));

        fixed_key
    }

    pub fn scan_key_range(
        &self,
        key: u64,
        target: Target,
        bits: Range<u32>,
        mut action: impl FnMut(u64),
    ) {
        // First compute the product
        let mut product = key.wrapping_mul(target.addr);

        debug_assert!(target.matches_key(key));
        action(key);
        for i in 1..1u64 << bits.len() {
            // By using a Gray code, we need just one xor per step, where trailing_zeros gives the
            // bit to flip to get to the next value in the Gray code.
            product ^=
                self.input_tab[(24 + bits.start + target.inv_shift + i.trailing_zeros()) as usize];
            let key = (product >> target.inv_shift).wrapping_mul(target.inv);
            debug_assert!(target.matches_key(key));
            action(key);
        }
    }
}

const BLOCK_BITS: u32 = 16;
static BAR_TEMPLATE: &str =
    "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {percent:>3}% ({eta_precise})";

struct Bruteforcer {
    targets: Vec<Target>,
    fixup: CrcFixup,
    min_shift: u32,
}

impl Bruteforcer {
    pub fn new(mut targets: Vec<Target>) -> Bruteforcer {
        assert!(!targets.is_empty());
        assert!(targets.len() < 256);

        let min_shift = targets.iter().map(|t| t.inv_shift).min().unwrap();
        targets.sort_by_key(|t| t.inv_shift);
        Bruteforcer {
            targets,
            fixup: CrcFixup::default(),
            min_shift,
        }
    }

    pub fn scan_key_range(&self, key: u64, bits: Range<u32>) -> Option<u64> {
        // We scan through all combinations we get by changing key's bits in the given range. For
        // each such candidate, we then fix up the 24 MSBs so it matches a target addr. We do this
        // for all candidate addresses and use a table to store 16 of the 24 MSBs to compare the
        // candidates for the different target addresses. If the candidate matches for all targets
        // we have found the key. As we only store 16 out of 24 MSBs we need to double check at the
        // end to discard false positives.
        //
        // This is slightly simplified as trailing zero bits of addresses move down the 24 bits that
        // are fixed up. As long as all addresses have the same number of trailing zeros, this is
        // essentially equivalent to making the key shorter and shifting everything accordingly.
        //
        // When the number of trailing zeros differs, we start with the fewest trailing zeros and
        // when comparing the candidates from later target addresses, do some masking to adjust for
        // this. This is explained a bit more inline.
        let mut table = [0u16; 1 << BLOCK_BITS];
        let table_mask = bits!(bits.len());
        assert!(table_mask <= table.len());

        for &target in &self.targets {
            // If an address has a lot of trailing zeros, fixing a key can change lower bits that
            // are outside of the given range. This means for that target address the whole range is
            // empty.
            let target_key = self.fixup.fixup_key(key, target);
            if (target_key & bits!(bits.start)) != (key & bits!(bits.start)) {
                return None;
            }
        }

        {
            // First we fill the table using a target with the fewest trailing zeros in the addr,
            // i.e. a target with the least number of equivalent keys.
            let target = self.targets[0];
            // We fix a first key
            let target_key = self.fixup.fixup_key(key, target);
            // And then scan all keys we can get from target_key by flipping bits in the given range
            // (and the 24 MSBs) and that are compatible with the target.

            self.fixup
                .scan_key_range(target_key, target, bits.clone(), |candidate| {
                    table[((candidate >> bits.start) as usize) & table_mask] =
                        (candidate >> bits.end) as u16;
                });
        }

        let mut check_table = [1u8; 1 << BLOCK_BITS];

        for (pass, &target) in self.targets.iter().enumerate().skip(1) {
            let target_key = self.fixup.fixup_key(key, target);

            // If this target addr has a larger number of trailing zeros, there are more redundant
            // keys, which means there are fewer bits for us to brute force through.
            let extra_shift = target.inv_shift - self.min_shift;
            let target_bits = bits.start..bits.end - extra_shift;

            // In that case we also need to mask our comparison, as the MSBs of the candidate are
            // arbitrary and changing them gives us redundant keys (but only for this target, the
            // first target fixes them!).
            let full_mask: u64 = bits!(24);
            let entry_mask = if extra_shift >= 24 {
                0
            } else {
                (full_mask >> extra_shift) as u16
            };

            // We scan through the keys in the same way as we did for the first target addr, but
            // this time we check if the candidate matches the corresponding candidate of the
            // previous targets. We do this by incrementing a match counter for each entry.
            self.fixup
                .scan_key_range(target_key, target, target_bits, |candidate| {
                    let index = ((candidate >> bits.start) as usize) & table_mask;
                    if check_table[index] == pass as u8 {
                        let entry = &mut table[index];
                        let expected = (candidate >> bits.end) as u16;
                        if *entry & entry_mask == expected & entry_mask {
                            check_table[index] += 1;
                        }
                    }
                });
        }

        let all_passes = self.targets.len() as u8;

        for (i, &entry) in check_table[0..1 << bits.len()].iter().enumerate() {
            if entry == all_passes {
                let candidate_lsbs = (key & bits!(bits.start)) | ((i as u64) << bits.start);
                let candidate = self.fixup.fixup_key(candidate_lsbs, self.targets[0]);
                if self
                    .targets
                    .iter()
                    .all(|target| target.matches_key(candidate))
                {
                    return Some(candidate);
                }
            }
        }
        None
    }

    pub fn parallel_scan(&self) -> Option<u64> {
        let chunk_bits = 24;
        let suffix_len = 24 + chunk_bits + self.min_shift;
        assert!(suffix_len <= 64);
        let prefix_len = 64 - suffix_len;
        let mid_bits = chunk_bits - BLOCK_BITS;

        let bar = ProgressBar::new(1 << prefix_len);
        bar.set_style(ProgressStyle::default_bar().template(BAR_TEMPLATE));

        // One bit fewer, as the key is always odd

        let rv = (0..1u64 << (prefix_len - 1))
            .into_par_iter()
            .progress_with(bar.clone())
            .find_map_any(|key_prefix| {
                for key_mid in 0..1 << mid_bits {
                    if let Some(res) = self.scan_key_range(
                        1 | (key_prefix << 1) | (key_mid << prefix_len),
                        prefix_len + mid_bits..prefix_len + chunk_bits,
                    ) {
                        return Some(res);
                    }
                }
                None
            });

        bar.finish_at_current_pos();
        rv
    }
}

fn main() {
    let targets: Vec<_> = std::env::args()
        .skip(1)
        .map(|arg| {
            Target::from_signed(
                u64::from_str_radix(&arg, 16).expect("could not parse signed pointer"),
            )
        })
        .collect();

    if targets.is_empty() {
        println!("usage: pac_brute ADDR1 ADDR2 ADDR3 ...");
        println!();

        println!("Each signed address is given in hex without prefix. Finding the key requires at");
        println!("least 3 addresses. If all given addresses have a certain number of trailing");
        println!("zeros, the same number of MSBs of the key are undetermined. They are arbitrary");
        println!("in the returned key, and they key will likely not work for addresses with fewer");
        println!("trailing zeros. Specifying more than 3 addresses makes things slower and is not");
        println!("needed unless the addresses line up in a bad way.");
        return;
    }

    let bruteforcer = Bruteforcer::new(targets);

    if let Some(key) = bruteforcer.parallel_scan() {
        println!("Found key {:016x}", key);
    } else {
        println!("Did not find a key :(");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test via inline assertions above

    #[test]
    fn generate_crc_fixup_table() {
        let table = CrcFixup::default();
        table.print_table();
    }

    #[test]
    fn fixup_key() {
        let table = CrcFixup::default();
        table.fixup_key(0xdeafbeef32531337, Target::from_signed(0xabcdef0123456789));
    }
}
