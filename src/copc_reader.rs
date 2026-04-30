//! Vendored COPC (Cloud Optimized Point Cloud) reader.
//!
//! Based on copc-rs 0.5.0 (MIT/Apache-2.0) by Pirmin Kalberer and Øyvind Hjermstad.
//! Only the reader portion is included — the writer is omitted to avoid a laz
//! version conflict (copc-rs pins laz 0.9.x, while las 0.9.11 uses laz 0.12.x).
//!
//! The decompressor uses laz 0.9.x directly, while the point types come from
//! las 0.9.x (same version tyler already uses).

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek, SeekFrom};
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt};
use las::raw;
use las::{Bounds, Builder, Header, Transform, Vector, Vlr};
use laz::laszip::LazVlr;
use laz::record::{LayeredPointRecordDecompressor, RecordDecompressor};

// ────────────────────────────── Error ──────────────────────────────

/// COPC reader errors.
#[derive(thiserror::Error, Debug)]
pub enum CopcError {
    #[error(transparent)]
    Las(#[from] las::Error),
    #[error(transparent)]
    LasZip(#[from] laz::LasZipError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("COPC info VLR not found")]
    CopcInfoVlrNotFound,
    #[error("EPT hierarchy VLR not found")]
    EptHierarchyVlrNotFound,
    #[error("LasZip VLR not found")]
    LasZipVlrNotFound,
    #[error("Invalid resolution: {0}")]
    InvalidResolution(f64),
}

pub type Result<T> = std::result::Result<T, CopcError>;

// ─────────────────────────── COPC VLR data ────────────────────────

/// COPC Info VLR data.
#[derive(Clone, Debug, Default)]
pub struct CopcInfo {
    pub center: Vector<f64>,
    pub halfsize: f64,
    pub spacing: f64,
    pub root_hier_offset: u64,
    pub root_hier_size: u64,
}

impl CopcInfo {
    fn read_from<R: Read>(mut read: R) -> Result<Self> {
        Ok(CopcInfo {
            center: Vector {
                x: read.read_f64::<LittleEndian>()?,
                y: read.read_f64::<LittleEndian>()?,
                z: read.read_f64::<LittleEndian>()?,
            },
            halfsize: read.read_f64::<LittleEndian>()?,
            spacing: read.read_f64::<LittleEndian>()?,
            root_hier_offset: read.read_u64::<LittleEndian>()?,
            root_hier_size: read.read_u64::<LittleEndian>()?,
            // gpstime_minimum, gpstime_maximum, _reserved — skip
        })
    }
}

// ───────────────────────── Octree structures ──────────────────────

#[derive(Hash, PartialEq, Eq, Clone, Debug)]
struct VoxelKey {
    level: i32,
    x: i32,
    y: i32,
    z: i32,
}

impl Default for VoxelKey {
    fn default() -> Self {
        VoxelKey { level: -1, x: 0, y: 0, z: 0 }
    }
}

impl VoxelKey {
    fn read_from<R: Read>(read: &mut R) -> Result<Self> {
        Ok(VoxelKey {
            level: read.read_i32::<LittleEndian>()?,
            x: read.read_i32::<LittleEndian>()?,
            y: read.read_i32::<LittleEndian>()?,
            z: read.read_i32::<LittleEndian>()?,
        })
    }

    fn children(&self) -> Vec<VoxelKey> {
        (0..8)
            .map(|dir| VoxelKey {
                level: self.level + 1,
                x: (self.x << 1) | (dir & 0x1),
                y: (self.y << 1) | ((dir >> 1) & 0x1),
                z: (self.z << 1) | ((dir >> 2) & 0x1),
            })
            .collect()
    }

    fn bounds(&self, root_bounds: &Bounds) -> Bounds {
        let side_size =
            (root_bounds.max.x - root_bounds.min.x) / 2_u32.pow(self.level as u32) as f64;
        Bounds {
            min: Vector {
                x: root_bounds.min.x + self.x as f64 * side_size,
                y: root_bounds.min.y + self.y as f64 * side_size,
                z: root_bounds.min.z + self.z as f64 * side_size,
            },
            max: Vector {
                x: root_bounds.min.x + (self.x + 1) as f64 * side_size,
                y: root_bounds.min.y + (self.y + 1) as f64 * side_size,
                z: root_bounds.min.z + (self.z + 1) as f64 * side_size,
            },
        }
    }
}

#[derive(Clone, Default, Debug)]
struct Entry {
    key: VoxelKey,
    offset: u64,
    byte_size: i32,
    point_count: i32,
}

impl Entry {
    fn read_from<R: Read>(read: &mut R) -> Result<Self> {
        Ok(Entry {
            key: VoxelKey::read_from(read)?,
            offset: read.read_u64::<LittleEndian>()?,
            byte_size: read.read_i32::<LittleEndian>()?,
            point_count: read.read_i32::<LittleEndian>()?,
        })
    }
}

#[derive(Clone, Debug)]
struct HierarchyPage {
    entries: Vec<Entry>,
}

impl HierarchyPage {
    fn read_from<R: Read>(mut read: R, page_size: u64) -> Result<Self> {
        let num_entries = page_size as usize / 32;
        let mut entries = Vec::with_capacity(num_entries);
        for _ in 0..num_entries {
            entries.push(Entry::read_from(&mut read)?);
        }
        Ok(HierarchyPage { entries })
    }
}

#[derive(Clone, Debug)]
struct OctreeNode {
    entry: Entry,
    bounds: Bounds,
    children: Vec<OctreeNode>,
}

impl OctreeNode {
    fn new() -> Self {
        OctreeNode {
            entry: Entry::default(),
            bounds: Bounds {
                min: Vector::default(),
                max: Vector::default(),
            },
            children: Vec::with_capacity(8),
        }
    }
}

// ───────────────────────── Decompressor ───────────────────────────

struct CopcDecompressor<'a, R: Read + Seek> {
    start: u64,
    vlr: &'a LazVlr,
    record_decompressor: LayeredPointRecordDecompressor<'a, R>,
}

impl<'a, R: Read + Seek> CopcDecompressor<'a, R> {
    fn new(mut source: R, vlr: &'a LazVlr) -> laz::Result<Self> {
        let start = source.stream_position()?;
        let mut record_decompressor = LayeredPointRecordDecompressor::new(source);
        record_decompressor.set_fields_from(vlr.items())?;
        Ok(Self { start, vlr, record_decompressor })
    }

    #[inline]
    fn source_seek(&mut self, offset: u64) -> laz::Result<()> {
        self.record_decompressor
            .get_mut()
            .seek(SeekFrom::Start(offset + self.start))?;
        self.record_decompressor.reset();
        self.record_decompressor.set_fields_from(self.vlr.items())
    }

    #[inline]
    fn decompress_one(&mut self, out: &mut [u8]) -> laz::Result<()> {
        self.record_decompressor
            .decompress_next(out)
            .map_err(laz::errors::LasZipError::IoError)
    }
}

// ──────────────────────── Public enums ─────────────────────────────

/// Level of detail selection for COPC queries.
pub enum LodSelection {
    /// Full resolution (all LODs).
    All,
    /// Minimum resolution (spacing between points).
    Resolution(f64),
    /// Single octree level.
    Level(i32),
    /// Level range [min, max).
    LevelMinMax(i32, i32),
}

/// Spatial bounds selection for COPC queries.
pub enum BoundsSelection {
    /// No bounds filter.
    All,
    /// Select points within bounds.
    Within(Bounds),
}

// ──────────────────────── CopcReader ───────────────────────────────

/// COPC file reader.
pub struct CopcReader<R> {
    start: u64,
    read: R,
    header: Header,
    copc_info: CopcInfo,
    laz_vlr: LazVlr,
    hierarchy_entries: HashMap<VoxelKey, Entry>,
}

impl CopcReader<BufReader<File>> {
    /// Open a COPC file from a path.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        File::open(path)
            .map_err(CopcError::from)
            .and_then(|file| CopcReader::new(BufReader::new(file)))
    }
}

impl<R: Read + Seek> CopcReader<R> {
    /// Setup by reading LAS header and LasZip VLRs.
    pub fn new(mut read: R) -> Result<Self> {
        let start = read.stream_position()?;
        let raw_header = raw::Header::read_from(&mut read)?;

        let mut position = raw_header.header_size as u64;
        let number_of_variable_length_records = raw_header.number_of_variable_length_records;
        let offset_to_point_data = raw_header.offset_to_point_data as u64;
        let evlr = raw_header.evlr;

        let mut builder = Builder::new(raw_header)?;

        for _ in 0..number_of_variable_length_records {
            let vlr = raw::Vlr::read_from(&mut read, false).map(Vlr::new)?;
            position += vlr.len(false) as u64;
            builder.vlrs.push(vlr);
        }

        match position.cmp(&offset_to_point_data) {
            Ordering::Less => {
                let _ = read
                    .by_ref()
                    .take(offset_to_point_data + start - position)
                    .read_to_end(&mut builder.vlr_padding)?;
            }
            Ordering::Equal => {}
            Ordering::Greater => Err(las::Error::OffsetToPointDataTooSmall(
                offset_to_point_data as u32,
            ))?,
        }

        if let Some(evlr) = evlr {
            let _ = read.seek(SeekFrom::Start(evlr.start_of_first_evlr + start))?;
            for _ in 0..evlr.number_of_evlrs {
                builder
                    .evlrs
                    .push(raw::Vlr::read_from(&mut read, true).map(Vlr::new)?);
            }
        }

        let header = builder.into_header()?;

        let mut copc_info = None;
        let mut laszip_vlr = None;
        let mut ept_hierarchy = None;

        for vlr in header.all_vlrs() {
            match (vlr.user_id.to_lowercase().as_str(), vlr.record_id) {
                ("copc", 1) => {
                    copc_info = Some(CopcInfo::read_from(vlr.data.as_slice())?);
                }
                ("copc", 1000) => {
                    ept_hierarchy = Some(vlr);
                }
                ("laszip encoded", 22204) => {
                    laszip_vlr = Some(LazVlr::read_from(vlr.data.as_slice())?);
                }
                _ => (),
            }
        }

        let copc_info = copc_info.ok_or(CopcError::CopcInfoVlrNotFound)?;

        let hierarchy_entries = match ept_hierarchy {
            None => return Err(CopcError::EptHierarchyVlrNotFound),
            Some(vlr) => {
                let mut hierarchy_entries = HashMap::new();
                let mut read_vlr = Cursor::new(vlr.data.as_slice());
                let mut page =
                    HierarchyPage::read_from(&mut read_vlr, copc_info.root_hier_size)?.entries;

                while let Some(entry) = page.pop() {
                    if entry.point_count == -1 {
                        read.seek(SeekFrom::Start(entry.offset - copc_info.root_hier_offset))?;
                        page.extend(
                            HierarchyPage::read_from(&mut read, entry.byte_size as u64)?.entries,
                        );
                    } else {
                        hierarchy_entries.insert(entry.key.clone(), entry);
                    }
                }
                hierarchy_entries
            }
        };

        let _ = read.seek(SeekFrom::Start(offset_to_point_data + start))?;
        Ok(CopcReader {
            start,
            read,
            header,
            copc_info,
            laz_vlr: laszip_vlr.ok_or(CopcError::LasZipVlrNotFound)?,
            hierarchy_entries,
        })
    }

    /// LAS header.
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// COPC info (center, halfsize, spacing, hierarchy offsets).
    pub fn copc_info(&self) -> &CopcInfo {
        &self.copc_info
    }

    /// Maximum octree level present in the hierarchy.
    pub fn max_octree_level(&self) -> i32 {
        self.hierarchy_entries.keys().map(|k| k.level).max().unwrap_or(0)
    }

    /// Load octree nodes matching level and bounds criteria.
    fn load_octree_for_query(
        &mut self,
        level_range: LodSelection,
        query_bounds: &BoundsSelection,
    ) -> Result<Vec<OctreeNode>> {
        let (level_min, level_max) = match level_range {
            LodSelection::All => (0, i32::MAX),
            LodSelection::Resolution(resolution) => {
                if !resolution.is_normal() || !resolution.is_sign_positive() {
                    return Err(CopcError::InvalidResolution(resolution));
                }
                (
                    0,
                    1.max((self.copc_info.spacing / resolution).log2().ceil() as i32 + 1),
                )
            }
            LodSelection::Level(level) => (level, level + 1),
            LodSelection::LevelMinMax(min, max) => (min, max),
        };

        let root_bounds = Bounds {
            min: Vector {
                x: self.copc_info.center.x - self.copc_info.halfsize,
                y: self.copc_info.center.y - self.copc_info.halfsize,
                z: self.copc_info.center.z - self.copc_info.halfsize,
            },
            max: Vector {
                x: self.copc_info.center.x + self.copc_info.halfsize,
                y: self.copc_info.center.y + self.copc_info.halfsize,
                z: self.copc_info.center.z + self.copc_info.halfsize,
            },
        };

        let mut satisfying_nodes = Vec::new();
        let mut node_stack = vec![OctreeNode::new()];
        node_stack[0].entry.key = VoxelKey { level: 0, x: 0, y: 0, z: 0 };

        while let Some(mut current_node) = node_stack.pop() {
            if current_node.entry.key.level >= level_max {
                continue;
            }

            let entry = match self.hierarchy_entries.get(&current_node.entry.key) {
                None => continue,
                Some(e) => e,
            };

            current_node.bounds = current_node.entry.key.bounds(&root_bounds);
            if let BoundsSelection::Within(bounds) = query_bounds {
                if !bounds_intersect(&current_node.bounds, bounds) {
                    continue;
                }
            }

            for child_key in current_node.entry.key.children() {
                let mut child_node = OctreeNode::new();
                child_node.entry.key = child_key;
                current_node.children.push(child_node.clone());
                node_stack.push(child_node);
            }

            if entry.point_count > 0
                && (level_min..level_max).contains(&current_node.entry.key.level)
            {
                current_node.entry = entry.clone();
                satisfying_nodes.push(current_node);
            }
        }

        satisfying_nodes.sort_by(|a, b| b.entry.offset.partial_cmp(&a.entry.offset).unwrap());
        Ok(satisfying_nodes)
    }

    /// Point iterator for selected level and bounds.
    pub fn points(
        &mut self,
        levels: LodSelection,
        bounds: BoundsSelection,
    ) -> Result<PointIter<R>> {
        let nodes = self.load_octree_for_query(levels, &bounds)?;
        let total_points_left = nodes.iter().map(|n| n.entry.point_count as usize).sum();
        let transforms = *self.header().transforms();

        let raw_bounds = match bounds {
            BoundsSelection::All => None,
            BoundsSelection::Within(bounds) => Some(RawBounds {
                min: Vector {
                    x: transforms.x.inverse(bounds.min.x).map_err(las::Error::from)?,
                    y: transforms.y.inverse(bounds.min.y).map_err(las::Error::from)?,
                    z: transforms.z.inverse(bounds.min.z).map_err(las::Error::from)?,
                },
                max: Vector {
                    x: transforms.x.inverse(bounds.max.x).map_err(las::Error::from)?,
                    y: transforms.y.inverse(bounds.max.y).map_err(las::Error::from)?,
                    z: transforms.z.inverse(bounds.max.z).map_err(las::Error::from)?,
                },
            }),
        };

        self.read.seek(SeekFrom::Start(self.start))?;
        let decompressor = CopcDecompressor::new(&mut self.read, &self.laz_vlr)?;
        let point = vec![
            0u8;
            (self.header.point_format().len() + self.header.point_format().extra_bytes) as usize
        ];

        Ok(PointIter {
            nodes,
            bounds: raw_bounds,
            point_format: *self.header.point_format(),
            transforms,
            decompressor,
            point_buffer: point,
            node_points_left: 0,
            total_points_left,
            current_node_level: 0,
        })
    }
}

// ──────────────────────── Helpers ──────────────────────────────────

struct RawBounds {
    min: Vector<i32>,
    max: Vector<i32>,
}

impl RawBounds {
    #[inline]
    fn contains_point(&self, p: &las::raw::Point) -> bool {
        !(p.x < self.min.x
            || p.y < self.min.y
            || p.z < self.min.z
            || p.x > self.max.x
            || p.y > self.max.y
            || p.z > self.max.z)
    }
}

#[inline]
fn bounds_intersect(a: &Bounds, b: &Bounds) -> bool {
    !(a.max.x < b.min.x
        || a.max.y < b.min.y
        || a.max.z < b.min.z
        || a.min.x > b.max.x
        || a.min.y > b.max.y
        || a.min.z > b.max.z)
}

// ──────────────────────── PointIter ───────────────────────────────

/// LasZip point iterator over COPC chunks.
/// Yields `(Point, octree_level)` pairs.
pub struct PointIter<'a, R: Read + Seek> {
    nodes: Vec<OctreeNode>,
    bounds: Option<RawBounds>,
    point_format: las::point::Format,
    transforms: Vector<Transform>,
    decompressor: CopcDecompressor<'a, &'a mut R>,
    point_buffer: Vec<u8>,
    node_points_left: usize,
    total_points_left: usize,
    current_node_level: i32,
}

impl<R: Read + Seek> Iterator for PointIter<'_, R> {
    type Item = (las::point::Point, i32);

    fn next(&mut self) -> Option<Self::Item> {
        if self.total_points_left == 0 {
            return None;
        }
        loop {
            while self.node_points_left == 0 {
                if let Some(node) = self.nodes.pop() {
                    self.decompressor.source_seek(node.entry.offset).unwrap();
                    self.node_points_left = node.entry.point_count as usize;
                    self.current_node_level = node.entry.key.level;
                } else {
                    return None;
                }
            }
            self.decompressor
                .decompress_one(self.point_buffer.as_mut_slice())
                .unwrap();
            let raw_point =
                las::raw::Point::read_from(self.point_buffer.as_slice(), &self.point_format)
                    .unwrap();
            self.node_points_left -= 1;
            self.total_points_left -= 1;

            let in_bounds = if let Some(bounds) = &self.bounds {
                bounds.contains_point(&raw_point)
            } else {
                true
            };

            if in_bounds {
                return Some((las::point::Point::new(raw_point, &self.transforms), self.current_node_level));
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.total_points_left, Some(self.total_points_left))
    }
}
