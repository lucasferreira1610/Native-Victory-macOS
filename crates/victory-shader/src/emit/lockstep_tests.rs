//! Pins for the lockstep rewrites of `lockstep.rs`, on hand-assembled
//! programs: `branches` (a mask branch becomes a 64-bit compare with 0 and
//! an SCC branch of the same distance) and `lds_barriers` (a group barrier
//! around each LDS write, none around reads).

use super::lockstep::{branches, lds_barriers};
use crate::ir::{self, Exit, Reg};

const S_ENDPGM: u32 = 0xBF81_0000;

fn build_words(words: &[u32]) -> ir::Program {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    ir::build(&code, true, 256).unwrap()
}

#[test]
fn an_execz_branch_becomes_a_compare_with_zero_and_an_scc_branch() {
    // s_cbranch_execz +1: while EXEC is zero, skip one instruction.
    let p = build_words(&[0xBF88_0001, 0xBF80_0000, S_ENDPGM]);
    let q = branches(&p);
    let block = q
        .blocks
        .iter()
        .find(|b| matches!(b.exit, Exit::Branch { .. }))
        .expect("a branch block");
    let names: Vec<_> = block.insts.iter().map(|i| i.name()).collect();
    assert_eq!(names, ["s_cmp_eq_u64", "s_cbranch_scc1"], "{names:?}");
    let (cmp, br) = (&block.insts[0], &block.insts[1]);
    // The compare reads EXEC as a pair and writes SCC; the branch reads that SCC
    // and keeps the original distance.
    assert_eq!(
        cmp.reads,
        [
            (Reg::S(ir::EXEC), cmp.reads[0].1),
            (Reg::S(ir::EXEC + 1), cmp.reads[1].1)
        ]
    );
    assert_eq!(cmp.writes.len(), 1);
    assert_eq!(cmp.writes[0].0, Reg::Scc);
    assert_eq!(br.reads, [(Reg::Scc, cmp.writes[0].1)]);
    assert_eq!(br.machine.words[0] & 0xFFFF, 1, "the branch's distance");
    assert_eq!(br.machine.words[0] >> 16 & 0x7F, 0x05, "s_cbranch_scc1");
}

#[test]
fn execnz_vccz_and_vccnz_use_the_matching_compare() {
    // s_cbranch_execnz -> s_cmp_lg_u64 exec, 0; s_cbranch_vccz -> s_cmp_eq_u64 vcc, 0;
    // s_cbranch_vccnz -> s_cmp_lg_u64 vcc, 0. (SOPP opcodes 0x009, 0x006, 0x007.)
    for (word, cmp_op, reg) in [
        (0xBF89_0000u32, 0x13u32, ir::EXEC),
        (0xBF86_0000, 0x12, ir::VCC),
        (0xBF87_0000, 0x13, ir::VCC),
    ] {
        let p = build_words(&[word | 1, 0xBF80_0000, S_ENDPGM]); // branch over the nop
        let q = branches(&p);
        let block = q
            .blocks
            .iter()
            .find(|b| matches!(b.exit, Exit::Branch { .. }))
            .expect("a branch block");
        assert_eq!(
            block.insts[0].name(),
            if cmp_op == 0x12 {
                "s_cmp_eq_u64"
            } else {
                "s_cmp_lg_u64"
            }
        );
        assert_eq!((block.insts[0].machine.words[0] >> 16) & 0x7F, cmp_op);
        assert_eq!((block.insts[0].machine.words[0]) & 0xFF, u32::from(reg));
        assert_eq!(block.insts[1].name(), "s_cbranch_scc1");
    }
}

#[test]
fn a_branch_on_scc_or_a_non_mask_branch_is_untouched() {
    // s_cbranch_scc1 branches on SCC, not on a mask: kept as it is.
    let p = build_words(&[0xBF85_0001, 0xBF80_0000, S_ENDPGM]);
    let q = branches(&p);
    let block = q
        .blocks
        .iter()
        .find(|b| matches!(b.exit, Exit::Branch { .. }))
        .expect("a branch block");
    assert_eq!(block.insts.len(), 1);
    assert_eq!(block.insts[0].name(), "s_cbranch_scc1");
    assert_eq!(block.insts[0].machine.words[0], 0xBF85_0001);
}

#[test]
fn lds_barriers_wrap_each_write_and_no_read() {
    // ds_write_b32 v1, v2 offset 0x40; ds_read_b32 v3, v1 offset 0x10.
    // DS encoding: 0xD8.. | op<<18 | gds<<17 | off1<<8 | off0; w1 = addr | data<<8 | dst<<24.
    let write = [0xD800_0000u32 | 0x00D << 18 | 0x40, 1 | 2 << 8];
    let read = [0xD800_0000 | 0x036 << 18 | 0x10, 1 | 3 << 24];
    let p = build_words(&[write[0], write[1], read[0], read[1], S_ENDPGM]);
    let at = lds_barriers(&p);
    let block = 0;
    assert!(at.contains(&(block, 0)), "a barrier before the write");
    assert!(at.contains(&(block, 1)), "a barrier after the write");
    assert!(!at.contains(&(block, 2)), "none around the read");
    assert!(!at.contains(&(block, 3)), "none around the read");
    assert_eq!(at.len(), 2);
}

fn emit_wave64_multi(words: &[u32], lanes: u32, lds_bytes: u32) -> Result<super::HostShader, crate::ShaderError> {
    use crate::dxbc::Container;
    use crate::native::{ComputeIo, HwStage, NativeProgram, RootParam, RootType, StageIo};
    use crate::{io, resource, structure};

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 256,
        layout: vec![RootParam {
            root: RootType::Descriptor,
            table_offset_dwords: 0,
            sgprs: 2,
            base_register: 0,
            user_data: 2,
            slot: 0,
        }],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2: (lds_bytes.div_ceil(512)) << 15 | 2 << 1,
            threads: [lanes, 1, 1],
        }),
        code,
    };
    let c = Container {
        digest: [0; 16],
        version: (1, 0),
        parts: vec![],
    };
    let p = ir::build(&native.code, true, 256)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, HwStage::Cs, Some(&st))?;
    let io = io::raise(&p, &native)?;
    super::emit_with(
        &c,
        &native,
        &p,
        &st,
        &io,
        &res,
        super::Options {
            wave32_host: true,
            ..super::Options::default()
        },
    )
}

/// #527 shape 1: behind an earlier vote (`s_cbranch_scc0`) and an EXEC skip, `s_bcnt1_i32_b64`
/// followed by `s_cbranch_scc0` in a 128- and 256-thread wave64 compute shader. Refused on the
/// fast path (`DIVERGENT_EXCHANGE`, even after `keep_lanes_to_end`), translated via multi-wave
/// if-conversion (`lower/predicated.rs`) with SSA and barrier uniformity verified.
#[test]
fn multi_wave_bcnt1_vote_branch_under_exec_skip_translates() {
    let w = [
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0x7E04_0280, // v_mov_b32 v2, 0
        // Outer vote (console-wave-wide) that skips the inner block when no lane < 50:
        0x7D88_00B2, // v_cmp_gt_u32 vcc, 50, v0
        0xBE84_106A, // s_bcnt1_i32_b64 s4, vcc
        0xBF84_0008, // s_cbranch_scc0 AFTER_OUTER (skip 8 words)
        // Inside the outer vote branch (not a `once` block): second vote on upper half lanes.
        0x7D82_00A0, // v_cmp_lt_u32 vcc, 32, v0
        0xBE86_106A, // s_bcnt1_i32_b64 s6, vcc (SCC = any lane > 32)
        0xBF84_0004, // s_cbranch_scc0 AFTER_INNER (skip 4 words)
        0x7E04_0206, // v_mov_b32 v2, s6
        0xDC70_8000,
        0x0002_0201, // global_store_dword v1, v2, s[2:3]
        0xBF80_0000, // s_nop 0
        // AFTER_INNER / AFTER_OUTER: post-branch store so `keep_lanes_to_end` cannot flatten it.
        0x4A04_0481, // v_add_nc_u32 v2, 1, v2
        0xDC70_8100,
        0x0002_0201, // global_store_dword v1, v2, s[2:3] offset:256
        S_ENDPGM,
    ];
    for lanes in [128, 256] {
        if let Err(e) = emit_wave64_multi(&w, lanes, 0) {
            panic!("lanes={lanes}: {e}");
        }
    }
    // Above the 256-thread cost bound, stays refused with DIVERGENT_EXCHANGE.
    let err = emit_wave64_multi(&w, 512, 0).err();
    assert!(
        matches!(
            err,
            Some(crate::ShaderError::Unsupported {
                what: super::lower::DIVERGENT_EXCHANGE,
                ..
            })
        ),
        "expected DIVERGENT_EXCHANGE at 512 threads, got {err:?}",
    );
}

/// #527 shape 2: under an EXEC skip (with post-merge work), `s_and_b64 vcc, exec, mask` followed
/// by `s_cbranch_vccz` guarding a cross-half read in a 128- and 256-thread compute shader.
#[test]
fn multi_wave_and_vcc_branch_under_exec_skip_translates() {
    let w = [
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0x4A06_00FF,
        0x0000_03E8, // v_add_nc_u32 v3, 1000, v0
        0x7E04_0280, // v_mov_b32 v2, 0
        0xD4C4_0008,
        0x0202_00A8, // v_cmp_gt_u32 s[8:9], 40, v0
        0x7D88_00B2, // v_cmp_gt_u32 vcc, 50, v0
        0xBE84_246A, // s_and_saveexec_b64 s[4:5], vcc
        0xBF88_0008, // s_cbranch_execz DONE
        0x87EA_087E, // s_and_b64 vcc, exec, s[8:9]
        0xBF86_0005, // s_cbranch_vccz AFTER
        0xD760_0006,
        0x0001_4703, // v_readlane_b32 s6, v3, 35
        0x7E04_0206, // v_mov_b32 v2, s6
        0xDC70_8000,
        0x0002_0201, // global_store_dword v1, v2, s[2:3]
        // AFTER / DONE:
        0x88FE_047E, // s_or_b64 exec, exec, s[4:5]
        0xDC70_8100,
        0x0002_0201, // global_store_dword v1, v2, s[2:3] offset:256
        S_ENDPGM,
    ];
    for lanes in [128, 256] {
        if let Err(e) = emit_wave64_multi(&w, lanes, 0) {
            panic!("lanes={lanes}: {e}");
        }
    }
    let err = emit_wave64_multi(&w, 512, 0).err();
    assert!(
        matches!(
            err,
            Some(crate::ShaderError::Unsupported {
                what: super::lower::DIVERGENT_EXCHANGE,
                ..
            })
        ),
        "expected DIVERGENT_EXCHANGE at 512 threads, got {err:?}",
    );
}

/// Multi-wave loop with per-wave trip counts, cross-half LDS exchange, and live-out VGPR + SGPR
/// values used after the loop (tests `group_any`, loop exit accumulation, and LCSSA snapshotting).
#[test]
fn multi_wave_loop_with_lds_and_live_outs_translates() {
    let w = [
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0x3A06_02FF,
        0x0000_0080, // v_xor_b32 v3, 0x80, v1
        0x3608_0083, // v_and_b32 v4, 3, v0
        0x7E0A_0300, // v_mov_b32 v5, v0
        0xD834_0000,
        0x0000_0501, // ds_write_b32 v1, v5
        0xBE86_047E, // s_mov_b64 s[6:7], exec
        // loop:
        0x7D8A_0880, // v_cmp_ne_u32 vcc, 0, v4
        0x87FE_6A7E, // s_and_b64 exec, exec, vcc
        0xBF88_0009, // s_cbranch_execz EXIT
        0xD8D8_0000,
        0x0200_0003, // ds_read_b32 v2, v3
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xD760_0008,
        0x0001_4702, // v_readlane_b32 s8, v2, 35
        0x4A0A_0A08, // v_add_nc_u32 v5, s8, v5
        0xD834_0000,
        0x0000_0501, // ds_write_b32 v1, v5
        0x4A08_08C1, // v_add_nc_u32 v4, -1, v4
        0xBF82_FFF3, // s_branch loop
        // EXIT:
        0xBEFE_0406, // s_mov_b64 exec, s[6:7]
        0xDC70_8000,
        0x0002_0501, // global_store_dword v1, v5, s[2:3]
        S_ENDPGM,
    ];
    for lanes in [128, 256] {
        if let Err(e) = emit_wave64_multi(&w, lanes, 1024) {
            panic!("lanes={lanes}: {e}");
        }
    }
    assert!(emit_wave64_multi(&w, 512, 1024).is_err());
}

/// Multi-wave if-conversion with an early `s_endpgm` in one arm: waves that exit early stay in
/// the linearized stream with `active = false` so they still meet live waves at later barriers.
#[test]
fn multi_wave_early_endpgm_keeps_later_exchanges_uniform() {
    let w = [
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
        0xBE86_106A, // s_bcnt1_i32_b64 s6, vcc
        0xBF85_0001, // s_cbranch_scc1 LIVE
        S_ENDPGM,    // waves with no lane < 40 exit early
        // LIVE:
        0xD760_0007,
        0x0001_4700, // v_readlane_b32 s7, v0, 35
        0x7E04_0207, // v_mov_b32 v2, s7
        0xDC70_8000,
        0x0002_0201, // global_store_dword v1, v2, s[2:3]
        S_ENDPGM,
    ];
    if let Err(e) = emit_wave64_multi(&w, 128, 0) {
        panic!("early endpgm failed: {e}");
    }
    let err = emit_wave64_multi(&w, 512, 0).err();
    assert!(
        matches!(
            err,
            Some(crate::ShaderError::Unsupported {
                what: super::lower::DIVERGENT_EXCHANGE,
                ..
            })
        ),
        "expected DIVERGENT_EXCHANGE at 512 threads, got {err:?}",
    );
}

/// Multi-wave nested loops with cross-half exchange inside the inner loop.
#[test]
fn multi_wave_nested_loops_with_cross_half_exchange_translate() {
    let w = [
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0x7E08_0282, // v_mov_b32 v4, 2 (outer count)
        0x7E0A_0280, // v_mov_b32 v5, 0 (acc)
        0xBE86_047E, // s_mov_b64 s[6:7], exec
        // OUTER:
        0x7D8A_0880, // v_cmp_ne_u32 vcc, 0, v4
        0x87FE_6A7E, // s_and_b64 exec, exec, vcc
        0xBF88_000E, // s_cbranch_execz OUTER_DONE (skip 14 words)
        0x7E0C_0283, // v_mov_b32 v6, 3 (inner count)
        0xBE88_047E, // s_mov_b64 s[8:9], exec
        // INNER:
        0x7D8A_0C80, // v_cmp_ne_u32 vcc, 0, v6
        0x87FE_6A7E, // s_and_b64 exec, exec, vcc
        0xBF88_0005, // s_cbranch_execz INNER_DONE (skip 5 words)
        0xD760_000A,
        0x0001_4700, // v_readlane_b32 s10, v0, 35
        0x4A0A_0A0A, // v_add_nc_u32 v5, s10, v5
        0x4A0C_0CC1, // v_add_nc_u32 v6, -1, v6
        0xBF82_FFF8, // s_branch INNER
        // INNER_DONE:
        0xBEFE_0408, // s_mov_b64 exec, s[8:9]
        0x4A08_08C1, // v_add_nc_u32 v4, -1, v4
        0xBF82_FFEF, // s_branch OUTER
        // OUTER_DONE:
        0xBEFE_0406, // s_mov_b64 exec, s[6:7]
        0xDC70_8000,
        0x0002_0501, // global_store_dword v1, v5, s[2:3]
        S_ENDPGM,
    ];
    if let Err(e) = emit_wave64_multi(&w, 128, 0) {
        panic!("nested loops failed: {e}");
    }
    assert!(emit_wave64_multi(&w, 512, 0).is_err());
}

/// Multi-wave if-conversion still refuses a branch whose SCC comes from a per-lane VGPR
/// (via `v_readfirstlane_b32` in a waterfall, or any lane-written scalar): the two halves of
/// one console wave would disagree on the predicate.
#[test]
fn multi_wave_if_conversion_refuses_split_scalar_branch() {
    use crate::native::{ComputeIo, HwStage, NativeProgram, RootParam, RootType, StageIo};
    use crate::resource::tests::WATERFALL_FETCHED_WAVE64;
    let mut w = WATERFALL_FETCHED_WAVE64.to_vec();
    let end = w.len() - 1;
    w.splice(
        end..end,
        [
            0xBF06_800C, // s_cmp_eq_u32 s12, 0
            0xBF85_0003, // s_cbranch_scc1 3
            0xD760_000E,
            0x0001_4700, // v_readlane_b32 s14, v0, 35
            0x7E1A_020E, // v_mov_b32 v13, s14
        ],
    );
    let code: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
    let table = |user_data| RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot: user_data,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 256,
        layout: vec![table(0), table(1)],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1,
            threads: [128, 1, 1],
        }),
        code,
    };
    let c = crate::dxbc::Container {
        digest: [0; 16],
        version: (1, 0),
        parts: vec![],
    };
    let p = ir::build(&native.code, true, 256).unwrap();
    let st = crate::structure::structure(&p).unwrap();
    let res = crate::resource::track(&p, &native.layout, HwStage::Cs, Some(&st)).unwrap();
    let io = crate::io::raise(&p, &native).unwrap();
    let err = super::emit_with(
        &c,
        &native,
        &p,
        &st,
        &io,
        &res,
        super::Options {
            wave32_host: true,
            ..super::Options::default()
        },
    )
    .err();
    assert!(err.is_some(), "branch on waterfall scalar must stay refused");
}
