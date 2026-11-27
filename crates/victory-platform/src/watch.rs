//! A write watch for diagnostics (`VICTORY_TRACE_DESCRIPTORS=sites`, #43): which title
//! instructions store into a set of watched ranges, and how often.
//!
//! [`watch`] publishes the ranges and makes their pages read-only. The loader's signal handler
//! hands every write fault to [`on_write_fault`] first, whoever wrote: a fault in a watched page
//! makes that page writable again, and the store runs again when the handler returns. It is
//! counted as a site only when the writing instruction lies in the title image ([`set_image`]) and
//! the address in a watched range itself, so host writes (the device's view calls) and title
//! stores that only share the page go on unseen. The next [`watch`] closes the pages again, so
//! what is counted is the first write to each page per interval, not every store.
//!
//! Signal safety: the fault path reads atomics, calls `mprotect` and adds to a fixed table
//! ([`Sites`], open addressing, no allocation or lock); a full table counts the sites it could not
//! hold.
//!
//! Updates race with faults: a title store can fault while [`watch`] replaces the ranges. So the
//! ranges live in two tables, each behind a sequence count. [`watch`] writes the table not in use,
//! publishes it, protects its pages, and only then opens the pages neither table holds; the
//! handler classifies against both tables (a table read while it is being written is skipped).
//! Every page read-only by the watch is then in a table the handler reads whole, whenever the
//! fault is handled: during the write, the protected pages are all in the published table; after
//! publication, they are in one of the two.
//!
//! Pages are opened to read-write, not to what they were before. A descriptor heap's records are
//! read-write memory (the device's view calls write them, as the title does), and the page around
//! a record is the record's page.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering, fence};

unsafe extern "C" {
    fn mprotect(addr: *mut std::ffi::c_void, len: usize, prot: i32) -> i32;
}
const READ: i32 = 1;
const READ_WRITE: i32 = 3;

/// `mprotect`, which only returns a status (nothing to allocate in a signal handler).
///
/// # Safety
/// The pages are the caller's.
unsafe fn set(at: usize, len: usize, prot: i32) -> bool {
    // SAFETY: forwarded from the caller.
    unsafe { mprotect(at as *mut _, len, prot) == 0 }
}

/// The most ranges watched at once (more are left out).
pub const RANGES: usize = 64;
/// The sites the table holds.
pub const SITES: usize = 4096;

/// Store sites (image offsets, never 0 here: offset 0 is the image header) and their counts.
pub struct Sites<const N: usize> {
    keys: [AtomicU64; N],
    counts: [AtomicU64; N],
    overflow: AtomicU64,
}

impl<const N: usize> Default for Sites<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> Sites<N> {
    pub const fn new() -> Self {
        Sites {
            keys: [const { AtomicU64::new(0) }; N],
            counts: [const { AtomicU64::new(0) }; N],
            overflow: AtomicU64::new(0),
        }
    }

    /// One store from `site`. Signal safe.
    pub fn add(&self, site: u64) {
        if site == 0 {
            return;
        }
        let start = (site.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) as usize % N;
        for i in 0..N {
            let k = (start + i) % N;
            let key = &self.keys[k];
            let found = match key.compare_exchange(0, site, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => true,
                Err(other) => other == site,
            };
            if found {
                self.counts[k].fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
        self.overflow.fetch_add(1, Ordering::Relaxed);
    }

    /// Every site and its count, most first (then by site).
    pub fn sorted(&self) -> Vec<(u64, u64)> {
        let mut out: Vec<(u64, u64)> = (0..N)
            .filter_map(|k| {
                let site = self.keys[k].load(Ordering::Acquire);
                (site != 0).then(|| (site, self.counts[k].load(Ordering::Relaxed)))
            })
            .collect();
        out.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        out
    }

    /// Stores from sites the full table could not hold.
    pub fn overflow(&self) -> u64 {
        self.overflow.load(Ordering::Relaxed)
    }
}

/// What a write fault at `addr` from the instruction at `rip` is to the watch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    /// Not in a watched page: someone else's fault.
    NotOurs,
    /// In a watched page: open the page (its first byte given) and run the store again.
    Open { page: usize },
    /// The same, and a title store into a watched range: count `site` (an image offset).
    Count { page: usize, site: u64 },
}

/// The rule, for ranges `[start, end)`, the title image `[base, end)` and pages of `page` bytes.
pub fn classify(
    addr: usize,
    rip: usize,
    ranges: impl IntoIterator<Item = (usize, usize)>,
    (image, image_end): (usize, usize),
    page: usize,
) -> Fault {
    let first = addr & !(page - 1);
    let mut open = false;
    for (start, end) in ranges {
        if start >= end {
            continue;
        }
        let pages = (start & !(page - 1))..(end.saturating_add(page - 1) & !(page - 1));
        if !pages.contains(&addr) {
            continue;
        }
        if (start..end).contains(&addr) && (image..image_end).contains(&rip) {
            return Fault::Count {
                page: first,
                site: (rip - image) as u64,
            };
        }
        open = true;
    }
    if open {
        Fault::Open { page: first }
    } else {
        Fault::NotOurs
    }
}

/// One set of ranges, behind a sequence count (odd while it is being written).
struct Table {
    seq: AtomicUsize,
    len: AtomicUsize,
    starts: [AtomicUsize; RANGES],
    ends: [AtomicUsize; RANGES],
}

impl Table {
    const fn new() -> Table {
        Table {
            seq: AtomicUsize::new(0),
            len: AtomicUsize::new(0),
            starts: [const { AtomicUsize::new(0) }; RANGES],
            ends: [const { AtomicUsize::new(0) }; RANGES],
        }
    }

    /// Copies the table into `out`: how many ranges, or `None` when it is being written. Signal
    /// safe.
    fn read(&self, out: &mut [(usize, usize); RANGES]) -> Option<usize> {
        let before = self.seq.load(Ordering::Acquire);
        if before % 2 == 1 {
            return None;
        }
        let n = self.len.load(Ordering::Relaxed).min(RANGES);
        for (i, r) in out.iter_mut().enumerate().take(n) {
            *r = (
                self.starts[i].load(Ordering::Relaxed),
                self.ends[i].load(Ordering::Relaxed),
            );
        }
        fence(Ordering::Acquire);
        (self.seq.load(Ordering::Relaxed) == before).then_some(n)
    }

    /// Replaces the ranges (one writer at a time).
    fn write(&self, ranges: &[(usize, usize)]) {
        let seq = self.seq.load(Ordering::Relaxed);
        self.seq.store(seq + 1, Ordering::Relaxed);
        fence(Ordering::Release);
        for (i, &(s, e)) in ranges.iter().enumerate() {
            self.starts[i].store(s, Ordering::Relaxed);
            self.ends[i].store(e, Ordering::Relaxed);
        }
        self.len.store(ranges.len(), Ordering::Relaxed);
        self.seq.store(seq + 2, Ordering::Release);
    }
}

/// The two tables, the one in use, and whether any page may be read-only.
static TABLES: [Table; 2] = [Table::new(), Table::new()];
static CURRENT: AtomicUsize = AtomicUsize::new(0);
static ACTIVE: AtomicBool = AtomicBool::new(false);
static IMAGE: [AtomicUsize; 2] = [AtomicUsize::new(0), AtomicUsize::new(0)];
static PAGE: AtomicUsize = AtomicUsize::new(0);
static SITES_SEEN: Sites<SITES> = Sites::new();

/// The title image, whose instructions count as store sites (from the loader, at the start).
pub fn set_image(base: usize, len: usize) {
    IMAGE[0].store(base, Ordering::Relaxed);
    IMAGE[1].store(base.saturating_add(len), Ordering::Release);
}

/// Whether anything is watched: the signal handler's one check when the watch is off.
pub fn active() -> bool {
    ACTIVE.load(Ordering::Relaxed)
}

/// A write fault at `addr` from the instruction at `rip`, from the signal handler: true when it
/// was the watch's (the page is writable again and the store is to run again). Signal safe.
pub fn on_write_fault(addr: usize, rip: usize) -> bool {
    let page = PAGE.load(Ordering::Relaxed);
    if page == 0 || !active() {
        return false;
    }
    let image = (
        IMAGE[0].load(Ordering::Relaxed),
        IMAGE[1].load(Ordering::Acquire),
    );
    // The table in use and the one before it (skipped if it is being written: the pages read-only
    // then are all in the one in use).
    let current = CURRENT.load(Ordering::Acquire);
    let (mut a, mut b) = ([(0, 0); RANGES], [(0, 0); RANGES]);
    let na = TABLES[current].read(&mut a).unwrap_or(0);
    let nb = TABLES[1 - current].read(&mut b).unwrap_or(0);
    let ranges = a[..na].iter().chain(&b[..nb]).copied();
    let at = match classify(addr, rip, ranges, image, page) {
        Fault::NotOurs => return false,
        Fault::Open { page } => page,
        Fault::Count { page, site } => {
            SITES_SEEN.add(site);
            page
        }
    };
    // SAFETY: a page of a range the caller of `watch` owns; it only becomes writable again.
    unsafe { set(at, page, READ_WRITE) }
}

/// Watches `ranges` (each `[start, end)`, read-write memory of the caller's) until the next call:
/// publishes them, then makes their pages read-only. Pages no range covers any more are made
/// writable again; an empty list ends the watch.
pub fn watch(ranges: &[(usize, usize)]) {
    static WRITER: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _one = WRITER.lock().unwrap_or_else(|e| e.into_inner());
    let page = crate::mem::page_size();
    PAGE.store(page, Ordering::Relaxed);
    let ranges = &ranges[..ranges.len().min(RANGES)];
    let pages =
        |(s, e): (usize, usize)| (s & !(page - 1))..(e.saturating_add(page - 1) & !(page - 1));
    let current = CURRENT.load(Ordering::Relaxed);
    let mut old = [(0, 0); RANGES];
    // Only this writer writes the tables, so the one in use reads whole.
    let n = TABLES[current].read(&mut old).unwrap_or(0);
    let old = &old[..n];
    // The table not in use: its pages are all in the one in use, whose pages the handler sees.
    TABLES[1 - current].write(ranges);
    if !ranges.is_empty() {
        ACTIVE.store(true, Ordering::Release);
    }
    CURRENT.store(1 - current, Ordering::Release);
    for &r in ranges {
        if r.0 >= r.1 {
            continue;
        }
        let p = pages(r);
        if p.end > p.start {
            // SAFETY: the caller's read-write memory; a write to it faults and reopens its page.
            unsafe { set(p.start, p.end - p.start, READ) };
        }
    }
    // Only now, with both tables readable, the pages the new ranges leave out.
    for &r in old.iter().filter(|r| r.0 < r.1 && !ranges.contains(r)) {
        for at in pages(r).step_by(page) {
            if !ranges.iter().any(|&n| n.0 < n.1 && pages(n).contains(&at)) {
                // SAFETY: a page this watch made read-only; it only becomes writable again.
                unsafe { set(at, page, READ_WRITE) };
            }
        }
    }
    if ranges.is_empty() {
        ACTIVE.store(false, Ordering::Release);
    }
}

/// Every store site counted so far (image offsets), most first, and the stores the full table
/// could not hold.
pub fn sites() -> (Vec<(u64, u64)>, u64) {
    (SITES_SEEN.sorted(), SITES_SEEN.overflow())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sites_count_and_overflow() {
        let s = Sites::<4>::new();
        for site in [0x10, 0x20, 0x10, 0x30, 0x10, 0x20, 0x40] {
            s.add(site);
        }
        assert_eq!(s.sorted(), [(0x10, 3), (0x20, 2), (0x30, 1), (0x40, 1)]);
        assert_eq!(s.overflow(), 0);
        // Full: a fifth site is counted as overflow, the four keep counting.
        s.add(0x50);
        s.add(0x50);
        s.add(0x30);
        assert_eq!(s.overflow(), 2);
        assert_eq!(s.sorted()[2], (0x30, 2));
    }

    #[test]
    fn only_title_stores_into_a_range_count() {
        let page = 0x1000;
        let image = (0x1_4000_0000, 0x1_5000_0000);
        let ranges = [(0x10_0100, 0x10_0200), (0x20_0000, 0x20_2000)];
        let title = 0x1_4000_1234;
        // In a range, from the title: counted, its page opened.
        assert_eq!(
            classify(0x10_0180, title, ranges, image, page),
            Fault::Count {
                page: 0x10_0000,
                site: 0x1234
            }
        );
        // The same page outside the range, or a host instruction: opened only.
        assert_eq!(
            classify(0x10_0080, title, ranges, image, page),
            Fault::Open { page: 0x10_0000 }
        );
        assert_eq!(
            classify(0x10_0180, 0x7000_0000, ranges, image, page),
            Fault::Open { page: 0x10_0000 }
        );
        // A range's last page, and pages no range touches.
        assert_eq!(
            classify(0x20_1ff8, title, ranges, image, page),
            Fault::Count {
                page: 0x20_1000,
                site: 0x1234
            }
        );
        assert_eq!(
            classify(0x20_2000, title, ranges, image, page),
            Fault::NotOurs
        );
        assert_eq!(
            classify(0x10_1000, title, ranges, image, page),
            Fault::NotOurs
        );
        assert_eq!(classify(0x10_0180, title, [], image, page), Fault::NotOurs);
    }

    /// An update moves the watch from page 0x10_0000 to 0x30_0000. Until the old page is opened,
    /// a store into it must still be the watch's: classified against the new table alone it was
    /// not (and became a title exception); against both tables, as the handler does, it is.
    #[test]
    fn a_store_into_the_old_tables_page_during_an_update_is_the_watchs() {
        let page = 0x1000;
        let image = (0x1_4000_0000, 0x1_5000_0000);
        let (old, new) = ([(0x10_0100, 0x10_0200)], [(0x30_0000, 0x30_0040)]);
        let title = 0x1_4000_0010;
        assert_eq!(classify(0x10_0180, title, new, image, page), Fault::NotOurs);
        let both = new.iter().chain(&old).copied();
        assert_eq!(
            classify(0x10_0180, title, both, image, page),
            Fault::Count {
                page: 0x10_0000,
                site: 0x10
            }
        );
    }

    #[test]
    fn a_table_being_written_is_not_read() {
        let t = Table::new();
        let mut out = [(0, 0); RANGES];
        t.write(&[(1, 2), (3, 4)]);
        assert_eq!(t.read(&mut out), Some(2));
        assert_eq!(out[..2], [(1, 2), (3, 4)]);
        // Mid-write (odd sequence): skipped, whatever its words hold.
        t.seq.fetch_add(1, Ordering::Relaxed);
        assert_eq!(t.read(&mut out), None);
        t.seq.fetch_add(1, Ordering::Relaxed);
        assert_eq!(t.read(&mut out), Some(2));
    }

    #[test]
    fn empty_and_inverted_ranges_are_ignored_by_classify() {
        let page = 0x1000;
        let image = (0x1_4000_0000, 0x1_5000_0000);
        let title = 0x1_4000_0010;
        assert_eq!(
            classify(0x10_0104, title, [(0x10_0104, 0x10_0104)], image, page),
            Fault::NotOurs
        );
        assert_eq!(
            classify(0x10_0104, title, [(0x10_2000, 0x10_0000)], image, page),
            Fault::NotOurs
        );
    }

    #[test]
    fn site_zero_does_not_corrupt_later_sites() {
        let s = Sites::<4>::new();
        s.add(0);
        s.add(0);
        assert!(s.sorted().is_empty());
        // Insert sites across every slot; none should inherit site 0's count.
        for site in [1u64, 2, 3, 4] {
            s.add(site);
        }
        assert!(s.sorted().iter().all(|&(_, count)| count == 1));
    }
}
