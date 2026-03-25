pub struct BuildOption {
    pub name: &'static str,
    pub cost_metal: u32,
    pub cost_energy: u32,
    pub cost_bp: u32,
    pub make_energy: u32,
    pub build_power: u32,
    pub conversion_drain: f32,
    pub conversion_result: f32,
    pub storage_energy: u32,
}

impl BuildOption {
    pub const fn new_converter(
        name: &'static str,
        cost_metal: u32,
        cost_energy: u32,
        cost_bp: u32,
        drain: f32,
        result: f32,
    ) -> BuildOption {
        BuildOption {
            name,
            cost_metal,
            cost_energy,
            cost_bp,
            make_energy: 0,
            build_power: 0,
            conversion_drain: drain,
            conversion_result: result,
            storage_energy: 0,
        }
    }

    pub const fn new_generator(
        name: &'static str,
        cost_metal: u32,
        cost_energy: u32,
        cost_bp: u32,
        make_energy: u32,
        storage_energy: u32,
    ) -> BuildOption {
        BuildOption {
            name,
            cost_metal,
            cost_energy,
            cost_bp,
            make_energy,
            build_power: 0,
            conversion_drain: 0.0,
            conversion_result: 0.0,
            storage_energy,
        }
    }
}