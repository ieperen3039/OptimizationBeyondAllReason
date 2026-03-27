use std::fmt::{Debug, Formatter};

#[derive(PartialEq)]
pub struct BuildOption {
    pub name: &'static str,
    pub cost_metal: u32,
    pub cost_energy: u32,
    pub cost_bp: u32,
    pub energy_generation: u32,
    pub build_power: u32,
    pub conversion_drain: f32,
    pub conversion_result: f32,
    pub energy_storage: u32,
}

impl BuildOption {}

impl Debug for BuildOption {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name)
    }
}