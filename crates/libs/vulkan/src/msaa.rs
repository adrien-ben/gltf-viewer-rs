#[derive(Copy, Clone, Debug, PartialOrd, PartialEq, Eq, Default)]
pub enum MsaaSamples {
    #[default]
    S1,
    S2,
    S4,
    S8,
    S16,
    S32,
    S64,
}
