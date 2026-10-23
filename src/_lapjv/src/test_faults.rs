//! Fault injection compiled only by `cargo test`, never into Python wheels.
use crate::Error;
use std::cell::Cell;

#[derive(Clone, Copy, PartialEq)]
pub enum Site {
    None,
    Allocation,
    Worker,
    Panic,
}
#[derive(Clone, Copy)]
pub struct Plan {
    pub site: Site,
    pub after: usize,
    pub batch: usize,
}
thread_local! {
    static PLAN: Cell<Plan> = const { Cell::new(Plan { site: Site::None, after: 0, batch: 0 }) };
    static BATCH: Cell<usize> = const { Cell::new(0) };
}
pub fn current() -> Plan {
    PLAN.with(Cell::get)
}
pub struct Scope(Plan, usize);
impl Scope {
    pub fn new(plan: Plan, batch: usize) -> Self {
        Self(
            PLAN.with(|p| p.replace(plan)),
            BATCH.with(|b| b.replace(batch)),
        )
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        PLAN.with(|p| p.set(self.0));
        BATCH.with(|b| b.set(self.1));
    }
}
pub fn hit(site: Site) -> Result<(), Error> {
    PLAN.with(|cell| {
        let mut plan = cell.get();
        if plan.site != site || plan.batch != BATCH.with(Cell::get) {
            return Ok(());
        }
        if plan.after > 0 {
            plan.after -= 1;
            cell.set(plan);
            return Ok(());
        }
        plan.site = Site::None;
        cell.set(plan);
        assert!(site != Site::Panic, "injected native panic");
        Err(Error::Allocation)
    })
}
