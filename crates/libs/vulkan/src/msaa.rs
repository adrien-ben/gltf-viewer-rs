#[derive(Copy, Clone, Debug, PartialOrd, PartialEq)]
pub enum MsaaSamples {
    S1,
    S2,
    S4,
    S8,
    S16,
    S32,
    S64,
}

impl Default for MsaaSamples {
    fn default() -> Self {
        MsaaSamples::S1
    }
}
