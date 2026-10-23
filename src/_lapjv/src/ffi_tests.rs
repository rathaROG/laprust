//! Exercise the actual C ABI, including error containment and reuse.
use super::*;
use crate::test_faults::{Plan, Scope, Site};

fn call(b: usize, r: usize, c: usize, threads: usize, f32_costs: bool) -> (i32, usize) {
    let input32 = vec![1.0f32; b * r * c];
    let input64 = vec![1.0f64; b * r * c];
    let input = if f32_costs {
        input32.as_ptr().cast()
    } else {
        input64.as_ptr().cast()
    };
    let mut output = vec![-1; 2 * b * r.min(c)];
    let mut counts = vec![0; b];
    let mut totals = vec![0.; b];
    let mut failed = 999;
    let status = unsafe {
        laprust_run(
            input,
            i32::from(f32_costs),
            input,
            i32::from(f32_costs),
            b,
            r,
            c,
            f64::INFINITY,
            threads,
            0,
            1,
            1,
            output.as_mut_ptr(),
            counts.as_mut_ptr(),
            totals.as_mut_ptr(),
            &mut failed,
        )
    };
    if status == 0 {
        assert_eq!(counts, vec![r.min(c); b]);
        assert_eq!(totals, vec![r.min(c) as f64; b]);
        for result in output.chunks_exact(2 * r.min(c)) {
            let mut columns = std::collections::HashSet::new();
            for pair in result.chunks_exact(2) {
                assert!((0..r as i64).contains(&pair[0]));
                assert!((0..c as i64).contains(&pair[1]));
                assert!(columns.insert(pair[1]));
            }
        }
    }
    (status, failed)
}

#[test]
fn workspace_allocation_failure_at_first_and_later_reservations_recovers() {
    for f32_costs in [false, true] {
        for (r, c) in [(7, 7), (7, 9), (9, 7)] {
            for after in [0, 2] {
                F32_WORKER.with(|w| *w.borrow_mut() = Worker::default());
                F64_WORKER.with(|w| *w.borrow_mut() = Worker::default());
                {
                    let _inject = Scope::new(
                        Plan {
                            site: Site::Allocation,
                            after,
                            batch: 0,
                        },
                        0,
                    );
                    assert_eq!(call(3, r, c, 1, f32_costs), (Error::Allocation as i32, 0));
                }
                assert_eq!(call(3, r, c, 1, f32_costs).0, 0);
            }
        }
    }
}

#[test]
fn worker_start_failure_and_partial_startup_recovers() {
    for after in [0, 2] {
        let threads = 7 + after; // These counts are not used by other pool tests.
        {
            let _inject = Scope::new(
                Plan {
                    site: Site::Worker,
                    after,
                    batch: 0,
                },
                0,
            );
            assert_eq!(
                call(threads, 4, 4, threads, false),
                (Error::Allocation as i32, 0)
            );
        }
        assert_eq!(call(threads, 4, 4, threads, false).0, 0);
    }
}

#[test]
fn panics_never_unwind_across_c_boundary_and_workers_recover() {
    for f32_costs in [false, true] {
        for threads in [1, 3] {
            {
                let _inject = Scope::new(
                    Plan {
                        site: Site::Panic,
                        after: 0,
                        batch: 2,
                    },
                    0,
                );
                assert_eq!(
                    call(5, 7, 9, threads, f32_costs),
                    (Error::Internal as i32, 0)
                );
            }
            assert_eq!(call(5, 7, 9, threads, f32_costs).0, 0);
        }
    }
}

#[test]
fn parallel_failures_report_the_lowest_index_and_recover() {
    let mut input = vec![0.0f64; 5 * 3 * 3];
    input[9..18].fill(f64::INFINITY);
    input[36..45].fill(f64::INFINITY);
    for threads in [1, 3] {
        let mut output = [0; 30];
        let mut counts = [0; 5];
        let mut totals = [0.; 5];
        let mut failed = 999;
        let status = unsafe {
            laprust_run(
                input.as_ptr().cast(),
                0,
                input.as_ptr().cast(),
                0,
                5,
                3,
                3,
                f64::INFINITY,
                threads,
                0,
                1,
                1,
                output.as_mut_ptr(),
                counts.as_mut_ptr(),
                totals.as_mut_ptr(),
                &mut failed,
            )
        };
        assert_eq!((status, failed), (Error::Infeasible as i32, 1));
        assert_eq!(call(5, 3, 3, threads, false).0, 0);
    }
}

#[test]
fn overflow_is_rejected_before_ffi_pointer_access() {
    let mut failed = 999;
    let status = unsafe {
        laprust_run(
            std::ptr::null(),
            0,
            std::ptr::null(),
            0,
            usize::MAX,
            2,
            2,
            f64::INFINITY,
            1,
            0,
            1,
            1,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut failed,
        )
    };
    assert_eq!((status, failed), (Error::Dimensions as i32, 0));
}
