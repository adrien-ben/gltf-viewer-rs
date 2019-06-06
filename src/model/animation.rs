use super::node::Nodes;
use crate::math::slerp;
use cgmath::{Quaternion, Vector3, VectorSpace};
use gltf::{
    animation::{
        iter::Channels,
        util::{ReadOutputs, Reader, Rotations},
        Channel as GltfChannel, Interpolation as GltfInterpolation, Property,
    },
    buffer::{Buffer, Data},
    iter::Animations,
    Animation as GltfAnimation,
};
use std::cmp::Ordering;

trait LinearInterpolation: Copy + Clone {
    fn interpolate(self, other: Self, amount: f32) -> Self;
}

impl LinearInterpolation for Vector3<f32> {
    fn interpolate(self, other: Self, amount: f32) -> Self {
        self.lerp(other, amount)
    }
}

impl LinearInterpolation for Quaternion<f32> {
    fn interpolate(self, other: Self, amount: f32) -> Self {
        slerp(self, other, amount)
    }
}

#[derive(Copy, Clone, Debug)]
enum Interpolation {
    Linear,
    Step,
}

#[derive(Debug)]
struct Sampler<T> {
    interpolation: Interpolation,
    times: Vec<f32>,
    values: Vec<T>,
}

impl<T> Sampler<T> {
    fn get_max_time(&self) -> f32 {
        self.times.last().copied().unwrap_or(0.0)
    }
}

impl<T: LinearInterpolation> Sampler<T> {
    fn sample(&self, t: f32) -> Option<T> {
        let index = {
            let mut index = None;
            for i in 0..(self.times.len() - 1) {
                let previous = self.times[i];
                let next = self.times[i + 1];
                if t >= previous && t < next {
                    index = Some(i);
                    break;
                }
            }
            index
        };

        index.map(|i| {
            let previous_time = self.times[i];
            let next_time = self.times[i + 1];
            let delta = next_time - previous_time;
            let from_start = t - previous_time;

            let previous_value = self.values[i];
            let next_value = self.values[i + 1];

            match self.interpolation {
                Interpolation::Step => previous_value,
                Interpolation::Linear => {
                    let factor = from_start / delta;
                    previous_value.interpolate(next_value, factor)
                }
            }
        })
    }
}

#[derive(Debug)]
struct Channel<T> {
    sampler: Sampler<T>,
    node_index: usize,
}

impl<T> Channel<T> {
    fn get_max_time(&self) -> f32 {
        self.sampler.get_max_time()
    }
}

impl<T: LinearInterpolation> Channel<T> {
    fn sample(&self, t: f32) -> Option<(usize, T)> {
        self.sampler.sample(t).map(|s| (self.node_index, s))
    }
}

struct NodesKeyFrame(
    Vec<(usize, Vector3<f32>)>,
    Vec<(usize, Quaternion<f32>)>,
    Vec<(usize, Vector3<f32>)>,
);

#[derive(Debug)]
pub struct Animation {
    time: f32,
    total_time: f32,
    translation_channels: Vec<Channel<Vector3<f32>>>,
    rotation_channels: Vec<Channel<Quaternion<f32>>>,
    scale_channels: Vec<Channel<Vector3<f32>>>,
}

impl Animation {
    /// Update nodes' transforms from animation data.
    ///
    /// Returns true if any nodes was updated.
    pub fn animate(&mut self, nodes: &mut Nodes, delta_time: f32) -> bool {
        let time = (self.time + delta_time) % self.total_time;
        self.time = time;

        let NodesKeyFrame(translations, rotations, scale) = self.sample(time);
        translations.iter().for_each(|(node_index, translation)| {
            nodes.nodes_mut()[*node_index].set_translation(*translation);
        });
        rotations.iter().for_each(|(node_index, rotation)| {
            nodes.nodes_mut()[*node_index].set_rotation(*rotation);
        });
        scale.iter().for_each(|(node_index, scale)| {
            nodes.nodes_mut()[*node_index].set_scale(*scale);
        });

        !translations.is_empty() || !rotations.is_empty() || !scale.is_empty()
    }

    fn sample(&self, t: f32) -> NodesKeyFrame {
        NodesKeyFrame(
            self.translation_channels
                .iter()
                .filter_map(|tc| tc.sample(t))
                .collect::<Vec<_>>(),
            self.rotation_channels
                .iter()
                .filter_map(|tc| tc.sample(t))
                .collect::<Vec<_>>(),
            self.scale_channels
                .iter()
                .filter_map(|tc| tc.sample(t))
                .collect::<Vec<_>>(),
        )
    }
}

pub fn load_animations(gltf_animations: Animations, data: &[Data]) -> Vec<Animation> {
    gltf_animations
        .map(|a| map_animation(&a, data))
        .collect::<Vec<_>>()
}

fn map_animation(gltf_animation: &GltfAnimation, data: &[Data]) -> Animation {
    let translation_channels = map_translation_channels(gltf_animation.channels(), data);
    let rotation_channels = map_rotation_channels(gltf_animation.channels(), data);
    let scale_channels = map_scale_channels(gltf_animation.channels(), data);

    let max_translation_time = translation_channels
        .iter()
        .map(Channel::get_max_time)
        .max_by(|c0, c1| c0.partial_cmp(&c1).unwrap_or(Ordering::Equal))
        .unwrap_or(0.0);
    let max_rotation_time = rotation_channels
        .iter()
        .map(Channel::get_max_time)
        .max_by(|c0, c1| c0.partial_cmp(&c1).unwrap_or(Ordering::Equal))
        .unwrap_or(0.0);
    let max_scale_time = scale_channels
        .iter()
        .map(Channel::get_max_time)
        .max_by(|c0, c1| c0.partial_cmp(&c1).unwrap_or(Ordering::Equal))
        .unwrap_or(0.0);

    let total_time = *[max_translation_time, max_rotation_time, max_scale_time]
        .iter()
        .max_by(|c0, c1| c0.partial_cmp(&c1).unwrap_or(Ordering::Equal))
        .unwrap_or(&0.0);

    Animation {
        time: 0.0,
        total_time,
        translation_channels,
        rotation_channels,
        scale_channels,
    }
}

fn map_translation_channels(gltf_channels: Channels, data: &[Data]) -> Vec<Channel<Vector3<f32>>> {
    gltf_channels
        .filter(|c| c.target().property() == Property::Translation)
        .filter_map(|c| map_translation_channel(&c, data))
        .collect::<Vec<_>>()
}

fn map_translation_channel(
    gltf_channel: &GltfChannel,
    data: &[Data],
) -> Option<Channel<Vector3<f32>>> {
    let gltf_sampler = gltf_channel.sampler();
    if let Property::Translation = gltf_channel.target().property() {
        map_interpolation(gltf_sampler.interpolation()).map(|i| {
            let reader = gltf_channel.reader(|buffer| Some(&data[buffer.index()]));
            let times = read_times(&reader);
            let output = read_translations(&reader);
            Channel {
                sampler: Sampler {
                    interpolation: i,
                    times,
                    values: output,
                },
                node_index: gltf_channel.target().node().index(),
            }
        })
    } else {
        None
    }
}

fn map_rotation_channels(gltf_channels: Channels, data: &[Data]) -> Vec<Channel<Quaternion<f32>>> {
    gltf_channels
        .filter(|c| c.target().property() == Property::Rotation)
        .filter_map(|c| map_rotation_channel(&c, data))
        .collect::<Vec<_>>()
}

fn map_rotation_channel(
    gltf_channel: &GltfChannel,
    data: &[Data],
) -> Option<Channel<Quaternion<f32>>> {
    let gltf_sampler = gltf_channel.sampler();
    if let Property::Rotation = gltf_channel.target().property() {
        map_interpolation(gltf_sampler.interpolation()).map(|i| {
            let reader = gltf_channel.reader(|buffer| Some(&data[buffer.index()]));
            let times = read_times(&reader);
            let output = read_rotations(&reader);
            Channel {
                sampler: Sampler {
                    interpolation: i,
                    times,
                    values: output,
                },
                node_index: gltf_channel.target().node().index(),
            }
        })
    } else {
        None
    }
}

fn map_scale_channels(gltf_channels: Channels, data: &[Data]) -> Vec<Channel<Vector3<f32>>> {
    gltf_channels
        .filter(|c| c.target().property() == Property::Scale)
        .filter_map(|c| map_scale_channel(&c, data))
        .collect::<Vec<_>>()
}

fn map_scale_channel(gltf_channel: &GltfChannel, data: &[Data]) -> Option<Channel<Vector3<f32>>> {
    let gltf_sampler = gltf_channel.sampler();
    if let Property::Scale = gltf_channel.target().property() {
        map_interpolation(gltf_sampler.interpolation()).map(|i| {
            let reader = gltf_channel.reader(|buffer| Some(&data[buffer.index()]));
            let times = read_times(&reader);
            let output = read_scales(&reader);
            Channel {
                sampler: Sampler {
                    interpolation: i,
                    times,
                    values: output,
                },
                node_index: gltf_channel.target().node().index(),
            }
        })
    } else {
        None
    }
}

fn map_interpolation(gltf_interpolation: GltfInterpolation) -> Option<Interpolation> {
    match gltf_interpolation {
        GltfInterpolation::Linear => Some(Interpolation::Linear),
        GltfInterpolation::Step => Some(Interpolation::Step),
        _ => None,
    }
}

fn read_times<'a, 's, F>(reader: &Reader<'a, 's, F>) -> Vec<f32>
where
    F: Clone + Fn(Buffer<'a>) -> Option<&'s [u8]>,
{
    reader.read_inputs().map_or(vec![], |times| times.collect())
}

fn read_translations<'a, 's, F>(reader: &Reader<'a, 's, F>) -> Vec<Vector3<f32>>
where
    F: Clone + Fn(Buffer<'a>) -> Option<&'s [u8]>,
{
    reader
        .read_outputs()
        .map_or(vec![], |outputs| match outputs {
            ReadOutputs::Translations(translations) => translations.map(Vector3::from).collect(),
            _ => vec![],
        })
}

fn read_scales<'a, 's, F>(reader: &Reader<'a, 's, F>) -> Vec<Vector3<f32>>
where
    F: Clone + Fn(Buffer<'a>) -> Option<&'s [u8]>,
{
    reader
        .read_outputs()
        .map_or(vec![], |outputs| match outputs {
            ReadOutputs::Scales(scales) => scales.map(Vector3::from).collect(),
            _ => vec![],
        })
}

fn read_rotations<'a, 's, F>(reader: &Reader<'a, 's, F>) -> Vec<Quaternion<f32>>
where
    F: Clone + Fn(Buffer<'a>) -> Option<&'s [u8]>,
{
    reader
        .read_outputs()
        .map_or(vec![], |outputs| match outputs {
            ReadOutputs::Rotations(scales) => {
                let scales = scales.into_f32().unwrap();
                match scales {
                    Rotations::F32(rotations) => rotations
                        .map(|r| Quaternion::new(r[3], r[0], r[1], r[2]))
                        .collect(),
                    _ => {
                        log::warn!("Failed to cast rotations to F32");
                        vec![]
                    }
                }
            }
            _ => vec![],
        })
}
