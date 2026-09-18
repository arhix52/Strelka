use std::collections::{HashMap, HashSet};
use std::io::{Cursor, Seek, SeekFrom};

use anyhow::{Context, Result, bail};
use byteorder::{LittleEndian, ReadBytesExt};
use glam::{Mat4, Quat, Vec3};
use wow_m2::chunks::animation::M2Animation;
use wow_m2::chunks::bone::M2Bone;
use wow_m2::chunks::m2_track::{M2TrackQuat, M2TrackVec3};
use wow_m2::common::{C3Vector, M2Array};
use wow_m2::header::M2Header;

use crate::ir::{
    AnimationChannel, AnimationClip, AnimationPath, AttachmentTemplate, BoneTemplate, SkinTemplate,
};

const MAX_BONES: u32 = 4096;
const MAX_ANIMATIONS: u32 = 4096;
type QuatFrames = (Vec<f32>, Vec<[f32; 4]>);

pub fn build_skin(
    data: &[u8],
    header: &M2Header,
    joints: Vec<[u16; 4]>,
    weights: Vec<[f32; 4]>,
    external_animations: &HashMap<(u16, u16), Vec<u8>>,
) -> Result<Option<SkinTemplate>> {
    if header.bones.count == 0 {
        return Ok(None);
    }
    if header.bones.count > MAX_BONES || header.animations.count > MAX_ANIMATIONS {
        bail!("unreasonable M2 skeleton dimensions");
    }
    let mut cursor = Cursor::new(data);
    cursor.seek(SeekFrom::Start(header.bones.offset as u64))?;
    let mut bones = Vec::with_capacity(header.bones.count as usize);
    for index in 0..header.bones.count {
        bones.push(
            M2Bone::parse(&mut cursor, header.version)
                .with_context(|| format!("failed to parse bone {index}"))?,
        );
    }
    cursor.seek(SeekFrom::Start(header.animations.offset as u64))?;
    let mut animations = Vec::with_capacity(header.animations.count as usize);
    for index in 0..header.animations.count {
        animations.push(
            M2Animation::parse(&mut cursor, header.version)
                .with_context(|| format!("failed to parse animation {index}"))?,
        );
    }
    cursor.seek(SeekFrom::Start(header.attachments.offset as u64))?;
    let mut attachments = Vec::new();
    for index in 0..header.attachments.count.min(1024) {
        let id = cursor
            .read_u32::<LittleEndian>()
            .with_context(|| format!("failed to parse attachment {index} ID"))?;
        let bone = cursor.read_u16::<LittleEndian>()? as usize;
        cursor.read_u16::<LittleEndian>()?;
        let x = cursor.read_f32::<LittleEndian>()?;
        let y = cursor.read_f32::<LittleEndian>()?;
        let z = cursor.read_f32::<LittleEndian>()?;
        cursor.seek(SeekFrom::Current(20))?;
        if bone < bones.len() {
            let pivot = &bones[bone].pivot;
            attachments.push(AttachmentTemplate {
                id,
                bone,
                translation: [x - pivot.x, z - pivot.z, -y + pivot.y],
            });
        }
    }
    let pivots: Vec<Vec3> = bones
        .iter()
        .map(|bone| Vec3::new(bone.pivot.x, bone.pivot.z, -bone.pivot.y))
        .collect();
    let bone_templates: Vec<BoneTemplate> = bones
        .iter()
        .enumerate()
        .map(|(index, bone)| {
            let parent = usize::try_from(bone.parent_bone)
                .ok()
                .filter(|parent| *parent < bones.len());
            let translation = parent.map_or(pivots[index], |parent| pivots[index] - pivots[parent]);
            BoneTemplate {
                parent,
                translation: translation.to_array(),
                inverse_bind: Mat4::from_translation(-pivots[index]).to_cols_array(),
            }
        })
        .collect();
    let clips = select_clips(data, &bones, &animations, external_animations)?;
    Ok(Some(SkinTemplate {
        joints,
        weights,
        bones: bone_templates,
        clips,
        attachments,
    }))
}

pub fn build_skeleton_clips(
    data: &[u8],
    external_animations: &HashMap<(u16, u16), Vec<u8>>,
) -> Result<Vec<AnimationClip>> {
    let skb1 = chunk_payload(data, b"SKB1").context("SKEL has no SKB1 chunk")?;
    let sks1 = chunk_payload(data, b"SKS1").context("SKEL has no SKS1 chunk")?;
    let mut cursor = Cursor::new(skb1);
    let bone_count = cursor.read_u32::<LittleEndian>()?;
    let bone_offset = cursor.read_u32::<LittleEndian>()?;
    if bone_count > MAX_BONES {
        bail!("unreasonable SKEL bone count {bone_count}");
    }
    cursor.seek(SeekFrom::Start(bone_offset as u64))?;
    let mut bones = Vec::with_capacity(bone_count as usize);
    for index in 0..bone_count {
        bones.push(
            M2Bone::parse(&mut cursor, 264)
                .with_context(|| format!("failed to parse SKEL bone {index}"))?,
        );
    }
    let mut cursor = Cursor::new(sks1);
    cursor.seek(SeekFrom::Start(8))?;
    let animation_count = cursor.read_u32::<LittleEndian>()?;
    let animation_offset = cursor.read_u32::<LittleEndian>()?;
    if animation_count > MAX_ANIMATIONS {
        bail!("unreasonable SKEL animation count {animation_count}");
    }
    cursor.seek(SeekFrom::Start(animation_offset as u64))?;
    let mut animations = Vec::with_capacity(animation_count as usize);
    for index in 0..animation_count {
        animations.push(
            M2Animation::parse(&mut cursor, 264)
                .with_context(|| format!("failed to parse SKEL animation {index}"))?,
        );
    }
    select_clips(skb1, &bones, &animations, external_animations)
}

fn select_clips(
    data: &[u8],
    bones: &[M2Bone],
    animations: &[M2Animation],
    external_animations: &HashMap<(u16, u16), Vec<u8>>,
) -> Result<Vec<AnimationClip>> {
    let mut selected_ids = HashSet::new();
    let mut clips = Vec::new();
    for (index, animation) in animations.iter().enumerate() {
        if !matches!(animation.animation_id, 0 | 4 | 5)
            || selected_ids.contains(&animation.animation_id)
        {
            continue;
        }
        if let Some(clip) = build_clip(
            data,
            external_animations.get(&(animation.animation_id, animation.sub_animation_id)),
            bones,
            index,
            animation.animation_id,
            animation.sub_animation_id,
            animation.flags & 0x20 != 0,
        )? {
            selected_ids.insert(animation.animation_id);
            clips.push(clip);
        }
    }
    add_locomotion_fallbacks(&mut clips, bones);
    Ok(clips)
}

fn add_locomotion_fallbacks(clips: &mut Vec<AnimationClip>, bones: &[M2Bone]) {
    let root = bones
        .iter()
        .position(|bone| bone.parent_bone < 0)
        .unwrap_or(0);
    let left_arm = bones.iter().position(|bone| bone.bone_id == 0);
    let right_arm = bones.iter().position(|bone| bone.bone_id == 1);
    let spine = bones.iter().position(|bone| bone.bone_id == 4);
    let waist = bones.iter().position(|bone| bone.bone_id == 5);
    let head = bones.iter().position(|bone| bone.bone_id == 6);
    let left_leg = leg_root_from_foot(bones, 59);
    let right_leg = leg_root_from_foot(bones, 60);
    if !clips.iter().any(|clip| clip.name.starts_with("Stand")) {
        clips.push(AnimationClip {
            name: "Stand_fallback".to_owned(),
            channels: vec![AnimationChannel {
                bone: root,
                path: AnimationPath::Translation,
                times: vec![0.0, 0.5, 1.0],
                values: vec![
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.015, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
            }],
            fallback: true,
        });
    }
    let stand_channels = clips
        .iter()
        .find(|clip| clip.name.starts_with("Stand"))
        .map(|clip| clip.channels.clone())
        .unwrap_or_default();
    for (animation_id, name, amplitude, bob) in [
        (4u16, "Walk_fallback", 0.55f32, 0.10f32),
        (5u16, "Run_fallback", 0.75f32, 0.16f32),
    ] {
        if clips.iter().any(|clip| {
            clip.name
                .starts_with(if animation_id == 4 { "Walk" } else { "Run" })
        }) {
            continue;
        }
        let times = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let mut channels = stand_channels.clone();
        channels.push(AnimationChannel {
            bone: root,
            path: AnimationPath::Translation,
            times: times.clone(),
            values: vec![
                [0.0, 0.0, 0.0, 0.0],
                [0.0, bob, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, bob, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        });
        for (bone, sign) in [(left_arm, 1.0f32), (right_arm, -1.0f32)] {
            let Some(bone) = bone else {
                continue;
            };
            channels.push(AnimationChannel {
                bone,
                path: AnimationPath::Rotation,
                times: times.clone(),
                values: [0.0, amplitude, 0.0, -amplitude, 0.0]
                    .map(|angle| {
                        let half = angle * sign * 0.5;
                        [0.0, half.sin(), 0.0, half.cos()]
                    })
                    .to_vec(),
            });
        }
        for (bone, sign) in [(left_leg, -1.0f32), (right_leg, 1.0f32)] {
            let Some(bone) = bone else {
                continue;
            };
            channels.push(AnimationChannel {
                bone,
                path: AnimationPath::Rotation,
                times: times.clone(),
                values: [0.0, amplitude, 0.0, -amplitude, 0.0]
                    .map(|angle| {
                        let half = angle * sign * 0.5;
                        [0.0, 0.0, half.sin(), half.cos()]
                    })
                    .to_vec(),
            });
        }
        for (bone, axis, scale) in [
            (spine, 2usize, 0.14f32),
            (waist, 1usize, -0.10f32),
            (head, 2usize, -0.07f32),
        ] {
            let Some(bone) = bone else {
                continue;
            };
            channels.push(AnimationChannel {
                bone,
                path: AnimationPath::Rotation,
                times: times.clone(),
                values: [0.0, amplitude, 0.0, -amplitude, 0.0]
                    .map(|angle| axis_quaternion(axis, angle * scale))
                    .to_vec(),
            });
        }
        if left_arm.is_none() && right_arm.is_none() && left_leg.is_none() && right_leg.is_none() {
            channels.push(AnimationChannel {
                bone: root,
                path: AnimationPath::Rotation,
                times: times.clone(),
                values: [0.0, amplitude, 0.0, -amplitude, 0.0]
                    .map(|angle| axis_quaternion(2, angle * 0.18))
                    .to_vec(),
            });
        }
        let mut unique_channels: Vec<AnimationChannel> = Vec::new();
        for channel in channels {
            if let Some(existing) = unique_channels
                .iter_mut()
                .find(|existing| existing.bone == channel.bone && existing.path == channel.path)
            {
                *existing = channel;
            } else {
                unique_channels.push(channel);
            }
        }
        clips.push(AnimationClip {
            name: name.to_owned(),
            channels: unique_channels,
            fallback: true,
        });
    }
}

fn axis_quaternion(axis: usize, angle: f32) -> [f32; 4] {
    let half = angle * 0.5;
    let mut result = [0.0, 0.0, 0.0, half.cos()];
    if axis < 3 {
        result[axis] = half.sin();
    }
    result
}

fn leg_root_from_foot(bones: &[M2Bone], foot_id: i32) -> Option<usize> {
    let mut current = bones.iter().position(|bone| bone.bone_id == foot_id)?;
    loop {
        let parent = usize::try_from(bones[current].parent_bone).ok()?;
        if parent >= bones.len() {
            return None;
        }
        if bones[parent].bone_id == 5 {
            return Some(current);
        }
        current = parent;
    }
}

fn chunk_payload<'a>(data: &'a [u8], magic: &[u8; 4]) -> Option<&'a [u8]> {
    let mut offset = 0usize;
    while offset + 8 <= data.len() {
        let size = u32::from_le_bytes(data[offset + 4..offset + 8].try_into().ok()?) as usize;
        let end = offset.checked_add(8 + size)?;
        if end > data.len() {
            return None;
        }
        if &data[offset..offset + 4] == magic {
            return data.get(offset + 8..end);
        }
        offset = end;
    }
    None
}

fn build_clip(
    data: &[u8],
    external: Option<&Vec<u8>>,
    bones: &[M2Bone],
    animation_index: usize,
    animation_id: u16,
    sub_id: u16,
    embedded: bool,
) -> Result<Option<AnimationClip>> {
    let mut channels = Vec::new();
    for (bone_index, bone) in bones.iter().enumerate() {
        if let Some((times, values)) = resolve_vec3(
            data,
            external.map(Vec::as_slice),
            embedded,
            &bone.translation,
            animation_index,
        )? {
            channels.push(AnimationChannel {
                bone: bone_index,
                path: AnimationPath::Translation,
                times: normalized_times(times),
                values: values
                    .into_iter()
                    .map(|value| [value.x, value.z, -value.y, 0.0])
                    .collect(),
            });
        }
        if let Some((times, values)) = resolve_quat(
            data,
            external.map(Vec::as_slice),
            embedded,
            &bone.rotation,
            animation_index,
        )? {
            channels.push(AnimationChannel {
                bone: bone_index,
                path: AnimationPath::Rotation,
                times: normalized_times(times),
                values: values
                    .into_iter()
                    .map(|value| {
                        let [x, y, z, w] = value;
                        let quaternion = Quat::from_xyzw(x, z, -y, w);
                        if quaternion.length_squared() > 0.0 {
                            quaternion.normalize().to_array()
                        } else {
                            Quat::IDENTITY.to_array()
                        }
                    })
                    .collect(),
            });
        }
        if let Some((times, values)) = resolve_vec3(
            data,
            external.map(Vec::as_slice),
            embedded,
            &bone.scale,
            animation_index,
        )? {
            channels.push(AnimationChannel {
                bone: bone_index,
                path: AnimationPath::Scale,
                times: normalized_times(times),
                values: values
                    .into_iter()
                    .map(|value| [value.x, value.z, value.y, 0.0])
                    .collect(),
            });
        }
    }
    if channels.is_empty() {
        return Ok(None);
    }
    let name = match animation_id {
        0 => format!("Stand_{sub_id}"),
        4 => format!("Walk_{sub_id}"),
        5 => format!("Run_{sub_id}"),
        _ => unreachable!(),
    };
    Ok(Some(AnimationClip {
        name,
        channels,
        fallback: false,
    }))
}

fn normalized_times(mut times: Vec<f32>) -> Vec<f32> {
    let start = times.first().copied().unwrap_or(0.0);
    let duration = times.last().copied().unwrap_or(start) - start;
    if duration > 0.0 && duration.is_finite() {
        for time in &mut times {
            *time = (*time - start) / duration;
        }
    } else {
        times.fill(0.0);
    }
    times
}

fn nested<T>(data: &[u8], array: &M2Array<T>, index: usize) -> Result<Option<M2Array<T>>> {
    if index >= array.count as usize {
        return Ok(None);
    }
    let offset = array.offset as usize + index * 8;
    let Some(bytes) = data.get(offset..offset + 8) else {
        return Ok(None);
    };
    Ok(Some(M2Array::new(
        u32::from_le_bytes(bytes[..4].try_into()?),
        u32::from_le_bytes(bytes[4..].try_into()?),
    )))
}

fn resolve_vec3(
    data: &[u8],
    external: Option<&[u8]>,
    embedded: bool,
    track: &M2TrackVec3,
    index: usize,
) -> Result<Option<(Vec<f32>, Vec<C3Vector>)>> {
    let Some(timestamps) = nested(data, &track.timestamps, index)? else {
        return Ok(None);
    };
    let Some(values) = nested(data, &track.values, index)? else {
        return Ok(None);
    };
    let count = timestamps.count.min(values.count) as usize;
    if count == 0 || count > 1_000_000 {
        return Ok(None);
    }
    let Some(track_data) =
        animation_data(data, external, embedded, &timestamps, &values, count, 12)
    else {
        return Ok(None);
    };
    let mut cursor = Cursor::new(track_data);
    cursor.seek(SeekFrom::Start(timestamps.offset as u64))?;
    let mut times = Vec::with_capacity(count);
    for _ in 0..count {
        times.push(cursor.read_u32::<LittleEndian>()? as f32 / 1000.0);
    }
    cursor.seek(SeekFrom::Start(values.offset as u64))?;
    let mut output = Vec::with_capacity(count);
    for _ in 0..count {
        output.push(C3Vector::parse(&mut cursor)?);
    }
    Ok(Some((times, output)))
}

fn resolve_quat(
    data: &[u8],
    external: Option<&[u8]>,
    embedded: bool,
    track: &M2TrackQuat,
    index: usize,
) -> Result<Option<QuatFrames>> {
    let Some(timestamps) = nested(data, &track.timestamps, index)? else {
        return Ok(None);
    };
    let Some(values) = nested(data, &track.values, index)? else {
        return Ok(None);
    };
    let count = timestamps.count.min(values.count) as usize;
    if count == 0 || count > 1_000_000 {
        return Ok(None);
    }
    let Some(track_data) = animation_data(data, external, embedded, &timestamps, &values, count, 8)
    else {
        return Ok(None);
    };
    let mut cursor = Cursor::new(track_data);
    cursor.seek(SeekFrom::Start(timestamps.offset as u64))?;
    let mut times = Vec::with_capacity(count);
    for _ in 0..count {
        times.push(cursor.read_u32::<LittleEndian>()? as f32 / 1000.0);
    }
    cursor.seek(SeekFrom::Start(values.offset as u64))?;
    let mut output = Vec::with_capacity(count);
    for _ in 0..count {
        output.push([
            (cursor.read_u16::<LittleEndian>()? as f32 - 32767.0) / 32768.0,
            (cursor.read_u16::<LittleEndian>()? as f32 - 32767.0) / 32768.0,
            (cursor.read_u16::<LittleEndian>()? as f32 - 32767.0) / 32768.0,
            (cursor.read_u16::<LittleEndian>()? as f32 - 32767.0) / 32768.0,
        ]);
    }
    Ok(Some((times, output)))
}

fn animation_data<'a, T, U>(
    model: &'a [u8],
    external: Option<&'a [u8]>,
    embedded: bool,
    timestamps: &M2Array<T>,
    values: &M2Array<U>,
    count: usize,
    value_stride: usize,
) -> Option<&'a [u8]> {
    if embedded
        && array_fits(model, timestamps.offset, count, 4)
        && array_fits(model, values.offset, count, value_stride)
    {
        return Some(model);
    }
    external.filter(|data| {
        array_fits(data, timestamps.offset, count, 4)
            && array_fits(data, values.offset, count, value_stride)
    })
}

fn array_fits(data: &[u8], offset: u32, count: usize, stride: usize) -> bool {
    (offset as usize)
        .checked_add(count.saturating_mul(stride))
        .is_some_and(|end| end <= data.len())
}

#[cfg(test)]
mod tests {
    use super::{array_fits, leg_root_from_foot, nested};
    use wow_m2::chunks::bone::M2Bone;
    use wow_m2::common::M2Array;

    #[test]
    fn resolves_per_animation_array_descriptors() {
        let mut data = vec![0u8; 32];
        data[8..12].copy_from_slice(&3u32.to_le_bytes());
        data[12..16].copy_from_slice(&20u32.to_le_bytes());
        let outer = M2Array::<u32>::new(1, 8);
        let resolved = nested(&data, &outer, 0).unwrap().unwrap();
        assert_eq!(resolved.count, 3);
        assert_eq!(resolved.offset, 20);
    }

    #[test]
    fn rejects_external_or_truncated_track_ranges() {
        let data = vec![0u8; 16];
        assert!(array_fits(&data, 4, 3, 4));
        assert!(!array_fits(&data, 8, 3, 4));
    }

    #[test]
    fn resolves_classic_leg_root_from_foot_hierarchy() {
        let bones = vec![
            M2Bone::new(5, -1),
            M2Bone::new(-1, 0),
            M2Bone::new(-1, 1),
            M2Bone::new(59, 2),
        ];
        assert_eq!(leg_root_from_foot(&bones, 59), Some(1));
    }
}
