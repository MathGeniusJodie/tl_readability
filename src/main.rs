use log::{debug, info};
use tl::{Node};

/// This is your parallel data structure. 
/// It stays aligned with tl's node vector by index.
#[derive(Default, Clone, Debug)]
struct NodeStats {
    score: f32,
    content_len: u32,
    link_text_len: u32,
}

use std::arch::x86_64::*;

#[inline(always)]
fn is_whitespace(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r')
}

// Checks for "&nbsp;" at curr_ptr + idx
// Safe to read because we check bounds against `len` (logical length)
// and the underlying buffer is guaranteed to be at least 64 bytes (either data or stack buffer).
#[inline(always)]
unsafe fn is_nbsp_at(ptr: *const u8, len: usize, idx: usize) -> bool {
    // We need 6 bytes: "&nbsp;"
    if idx + 6 > len {
        return false;
    }
    // Optimization: Read "nbsp" as u32
    // and the trailing ';' byte.
    let start = ptr.add(idx + 1);
    let sequence = (start as *const u32).read_unaligned();
    let semi = *start.add(4);
    
    sequence == u32::from_le_bytes(*b"nbsp") && semi == b';'
}

pub fn count_redundant_whitespace_simd(data: &[u8]) -> u32 {
    unsafe {
        // --- 1. Setup LUTs ---
        let lut_128 = _mm_setr_epi8(
            0x20, -1, -1, -1, -1, -1, -1, -1,
            -1, 0x09, 0x0A, -1, -1, 0x0D, -1, -1,
        );

        #[cfg(target_feature = "avx512bw")]
        let lut_vec = _mm512_broadcast_i32x4(lut_128);
        #[cfg(not(target_feature = "avx512bw"))]
        let lut_vec = _mm256_broadcastsi128_si256(lut_128);

        // --- 2. State & Buffers ---
        let mut count = 0u32;
        let mut prev_msb = 0u64;    // MSB of whitespace mask from previous iter
        let mut carry_semis = 0u64; // Semicolons from &nbsp; spilling into next chunk

        // Temp buffer for the final remainder chunk. 
        // We do not need to zero-init it because we mask the results based on length.
        let mut buffer = [0u8; 64]; 
        
        let len = data.len();
        let ptr = data.as_ptr();
        let mut offset = 0;

        // --- 3. Unified Loop ---
        while offset < len {
            let remaining = len - offset;
            
            // Determine source pointer and valid length
            // If full chunk: point to data, length is effectively infinite for our 64-byte window
            // If remainder: copy to buffer, point to buffer, length is exact
            let (curr_ptr, chunk_len) = if remaining >= 64 {
                (ptr.add(offset), remaining)
            } else {
                // Copy remainder to stack buffer
                std::ptr::copy_nonoverlapping(ptr.add(offset), buffer.as_mut_ptr(), remaining);
                (buffer.as_ptr(), remaining)
            };

            // --- 4. SIMD Load & Mask Generation ---
            // Calculate mask_ws (Standard Whitespace) and mask_amp (Ampersands)
            let (mask_ws, mask_amp) = {
                #[cfg(target_feature = "avx512bw")]
                {
                    let v = _mm512_loadu_si512(curr_ptr as *const _);
                    let expected = _mm512_shuffle_epi8(lut_vec, v);
                    let ws = _mm512_cmpeq_epi8_mask(v, expected);
                    let amp = _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b'&' as i8));
                    (ws, amp)
                }
                #[cfg(not(target_feature = "avx512bw"))]
                {
                    // Load 2x 256-bit vectors
                    let v0 = _mm256_loadu_si256(curr_ptr as *const __m256i);
                    let v1 = _mm256_loadu_si256(curr_ptr.add(32) as *const __m256i);
                    let amp_vec = _mm256_set1_epi8(b'&' as i8);

                    // Generate WS mask
                    let ws0 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v0, _mm256_shuffle_epi8(lut_vec, v0))) as u32;
                    let ws1 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, _mm256_shuffle_epi8(lut_vec, v1))) as u32;
                    
                    // Generate Ampersand mask
                    let amp0 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v0, amp_vec)) as u32;
                    let amp1 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, amp_vec)) as u32;

                    let ws = (ws0 as u64) | ((ws1 as u64) << 32);
                    let amp = (amp0 as u64) | ((amp1 as u64) << 32);
                    (ws, amp)
                }
            };

            // 1. Identify start of entity (&)
            // mask_amp tells us where '&' is. We filter for only valid "&nbsp;" sequences.
            let mut nbsp_start_mask = 0u64;
            let mut pending_amps = mask_amp;
            while pending_amps != 0 {
                let idx = pending_amps.trailing_zeros();
                pending_amps &= !(1 << idx);
                if is_nbsp_at(curr_ptr, chunk_len, idx as usize) {
                    nbsp_start_mask |= 1 << idx;
                }
            }

            // 2. Identify end of entity (;)
            // If '&' is at bit N, the ';' is at bit N+5.
            // We combine ends calculated from current chunk with those spilling over from previous.
            let nbsp_end_local = nbsp_start_mask << 5; 
            let nbsp_end_mask  = nbsp_end_local | carry_semis;

            // 3. Construct the "All Whitespace" view
            // A byte behaves as whitespace if it is:
            // - A standard space/tab/newline
            // - The start of an &nbsp; (the '&')
            // - The end of an &nbsp; (the ';')
            let any_ws_mask = mask_ws | nbsp_start_mask | nbsp_end_mask;

            // 4. Calculate Redundancy
            // A character is redundant if it is whitespace AND the previous character was whitespace.
            // (prev_msb handles the boundary case between chunks)
            let prev_was_ws = (any_ws_mask << 1) | prev_msb;
            let is_redundant = any_ws_mask & prev_was_ws;

            // 5. Accumulate Counts
            // Case A: [WS] followed by [Standard WS] or [End of &nbsp;] -> 1 redundant char
            count += (is_redundant & mask_ws).count_ones(); 
            
            // Case B: [WS] followed by [&nbsp;] -> 6 redundant chars
            // We check this by seeing if the 'redundant' bit aligns with an '&' start.
            count += (is_redundant & nbsp_start_mask).count_ones() * 6;

            // 6. Prepare state for next iteration
            // The MSB (bit 63) of the current mask becomes the "previous" bit for the next chunk.
            prev_msb = any_ws_mask >> 63;
            
            // If an &nbsp; starts near the end (bits 59-63), its ';' falls into the next chunk.
            // We save these overflow bits to apply them as 'carry_semis' next time.
            carry_semis = nbsp_start_mask >> 59; // 64 - 5 = 59

            offset += 64;
        }

        count
    }
}

use std::arch::x86_64::*;
use std::ptr;

#[target_feature(enable = "avx2")]
pub unsafe fn find_fast_avx2_extended(text: &[u8]) -> Option<usize> {
    let ptr = text.as_ptr();
    let len = text.len();
    let mut i = 0;

    // --- Setup LUTs ---
    // Maps low nibble to expected byte. 
    // '&' (0x26) -> Index 6 is set to 0, so the LUT IGNORES ampersands.
    let lut_128 = _mm_setr_epi8(
        0x20, 0, 0, 0, 0, 0, 0, 0, // 0->Space
        0, 0x09, 0x0A, 0, 0, 0x0D, 0, 0  // 9->Tab, 10->LF, 13->CR
    );
    let lut = _mm256_set_m128i(lut_128, lut_128);
    let amp_char = _mm256_set1_epi8(b'&' as i8);

    let mut prev_char_is_ws = false;

    loop {
        let remaining = len - i;

        // 1. Load Data (Full chunk or Tail buffer)
        let (chunk, valid_count) = if remaining >= 32 {
            (_mm256_loadu_si256(ptr.add(i) as *const __m256i), 32)
        } else {
            if remaining == 0 { break; }
            let mut buffer = [0u8; 32];
            ptr::copy_nonoverlapping(ptr.add(i), buffer.as_mut_ptr(), remaining);
            (_mm256_loadu_si256(buffer.as_ptr() as *const __m256i), remaining)
        };

        // 2. SIMD Identification
        let expected = _mm256_shuffle_epi8(lut, chunk);
        let ws_vec = _mm256_cmpeq_epi8(chunk, expected);
        let ws_mask = _mm256_movemask_epi8(ws_vec) as u32;

        let amp_vec = _mm256_cmpeq_epi8(chunk, amp_char);
        let amp_mask = _mm256_movemask_epi8(amp_vec) as u32;

        // 3. Bitwise Logic
        // Combine masks to see if any interesting characters exist
        let all_interesting = ws_mask | amp_mask;
        
        if all_interesting != 0 {
            // MATCH LOGIC:
            // 1. Current char is Whitespace...
            // 2. ...AND Next char is (Whitespace OR Ampersand)
            let next_is_interesting = all_interesting >> 1;
            let mut sequence_match = ws_mask & next_is_interesting;

            // Handle Boundary: 
            // If prev chunk ended in WS, and this chunk starts with (WS or &)
            if prev_char_is_ws && (all_interesting & 1) == 1 {
                // Return index i (which is the start of this chunk). 
                // Technically the match started at i-1, but since i-1 was in the 
                // previous chunk (where we didn't return), returning i is standard.
                sequence_match |= 1;
            }

            // Combine with standalone Ampersand matches
            let matches = sequence_match | amp_mask;

            if matches != 0 {
                let match_index = matches.trailing_zeros() as usize;
                if match_index < valid_count {
                    return Some(i + match_index);
                }
            }
        }

        if valid_count < 32 { break; }

        prev_char_is_ws = (ws_mask & (1 << 31)) != 0;
        i += 32;
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_redundant_whitespace_simd_basic() {
        // Example: "a  b" has 1 redundant whitespace (second space)
        let data = b"a  b";
        assert_eq!(count_redundant_whitespace_simd(data), 1);
    }

    #[test]
    fn test_count_redundant_whitespace_simd_nbsp() {
        // Example: "a  b" has 1 redundant whitespace (second space)
        let data = b"a  &nbsp;&nbsp; &nbsp;b";
        assert_eq!(count_redundant_whitespace_simd(data), 20);
    }

    #[test]
    fn test_count_redundant_whitespace_simd_multiple_types() {
        let data = b"a \t b";
        assert_eq!(count_redundant_whitespace_simd(data), 2);

        let data = b"a  \t\t b";
        assert_eq!(count_redundant_whitespace_simd(data), 4);

        let data = b"a \n\n\n b";
        assert_eq!(count_redundant_whitespace_simd(data), 4);
    }

    #[test]
    fn test_count_redundant_whitespace_simd_edge_cases() {
        // Empty input
        let data = b"";
        assert_eq!(count_redundant_whitespace_simd(data), 0);

        // Single whitespace
        let data = b" ";
        assert_eq!(count_redundant_whitespace_simd(data), 0);

        // All whitespace
        let data = b"    ";
        assert_eq!(count_redundant_whitespace_simd(data), 3);

        let data = b"\t\t\t";
        assert_eq!(count_redundant_whitespace_simd(data), 2);
    }

    // More tests will be added for edge cases and different whitespace types
    #[test]
    fn test_count_redundant_whitespace_simd_long_spaces() {
        // 1000 spaces: should have 999 redundant whitespace
        let data = vec![b' '; 1000];
        assert_eq!(count_redundant_whitespace_simd(&data), 999);
    }

    #[test]
    fn test_count_redundant_whitespace_simd_long_mixed() {
        // 500 spaces, 500 tabs: every whitespace after the first is redundant
        let mut data = vec![b' '; 500];
        data.extend(vec![b'\t'; 500]);
        // 999 redundant whitespace (every whitespace after the first)
        assert_eq!(count_redundant_whitespace_simd(&data), 999);
    }

    #[test]
    fn test_count_redundant_whitespace_simd_long_pattern() {
        // Alternate space and tab, 1000 bytes: no redundant whitespace
        let mut data = Vec::with_capacity(1000);
        for i in 0..1000 {
            data.push(if i % 2 == 0 { b' ' } else { b'\t' });
        }
        assert_eq!(count_redundant_whitespace_simd(&data), 999);
    }
}

/*
todo: case insensitive tag name comparisons
todo: fast whitespace collapsed charcount
fast comma charcount
*/

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Setup Logging (Run with `RUST_LOG=debug cargo run`)
    env_logger::init();

    // 2. Load and Parse
    let html = std::fs::read_to_string("/home/jodie/Downloads/test.html")
        .expect("Place a 'test.html' file in the root directory");
    
    let dom = tl::parse(&html, tl::ParserOptions::default())?;
    let parser = dom.parser();
    let nodes = dom.nodes();
    
    // 3. Initialize parallel stats vector
    let mut stats = vec![NodeStats::default(); nodes.len()];

    info!("Processing {} nodes...", nodes.len());

    // 4. THE BACKWARDS PASS (Reverse Document Order)
    // Children are processed BEFORE parents.
    for id in (0..nodes.len()).rev() {
        let node = &nodes[id];

        match node {
            Node::Tag(tag) => {
                let mut current = NodeStats::default();

                // PULL logic: Look at children (already processed in the loop)
                for child_handle in tag.children().top().iter() {
                    let child_id = child_handle.get_inner() as usize;

                    match child_handle.get(parser){
                        Some(Node::Raw(bytes)) => {
                            // SIMD-ready text scanning logic goes here
                            let raw_bytes = bytes.as_bytes();
                            current.content_len += raw_bytes.len() as u32;
                        }
                        Some(Node::Tag(_)) => {
                            // Pull accumulated values from the child we already scored
                            let child_stat = &stats[child_id];
                            current.content_len += child_stat.content_len;
                            current.link_text_len += child_stat.link_text_len;
                            current.score += child_stat.score;
                        }
                        _ => {}
                    }
                }

                // LOGIC PLACEHOLDER: 
                // If tag == "a", current.link_text_len = current.content_len;
                // If tag == "p", current.score += some_calculation;
                
                if tag.name().as_utf8_str() == "a" {
                    current.link_text_len = current.content_len;
                }

                // Store the result
                stats[id] = current;
            }
            Node::Raw(_) => {
                // We handle raw nodes inside the Tag match (the Parent) 
                // but you could also store raw text scores here if needed.
            }
            _ => {}
        }
    }

    // 5. DEBUG OUTPUT
    // Iterate forwards to show the relationship between nodes and your data
    for (id, node) in nodes.iter().enumerate() {
        let stat = &stats[id];
        println!(
            "Node [{:?}] | Score: {:.2} | Text: {} | LinkText: {}",
            node.as_tag().map(|t| t.name().as_utf8_str()).unwrap_or("text-or-other".into()),
            stat.score,
            stat.content_len, 
            stat.link_text_len
        );
    }

    Ok(())
}