//! LAS/LAZ/COPC point cloud reading and Gaussian splat generation.
//!
//! Supports regular LAS/LAZ (via the `las` crate) and Cloud Optimized Point
//! Cloud (COPC) files (via `copc-rs`). COPC is detected automatically.
//!
//! Phase 4a: reads colored LiDAR (LAS/LAZ/COPC with per-point RGB) and produces
//! a `SplatCloud` spatially indexed into the same grid as the CityJSON features.
//! Each colored point becomes one Gaussian splat with identity rotation,
//! isotropic scale, and full opacity.
//!
//! Phase 4b (future): same pipeline but RGB is sampled from an external raster
//! instead of from the LAS point data.

use std::collections::HashMap;
use std::io::BufReader;
use std::path::Path;

use anyhow::{bail, Context, Result};
use log::{debug, info, warn};

use crate::copc_reader::CopcReader;
use crate::parser::{CityJSONFeatureVertices, Transform};
use crate::proj::Proj;
use crate::spatial_structs::{CellId, SquareGrid};

/// A Gaussian splat ready for GLB writing.
///
/// Phase 4a: rgb from LiDAR, identity rotation, isotropic scale, full opacity.
/// Phase 4b (future): rgb sampled from external image at point XY.
/// Both phases produce the same struct — only the color source differs.
pub struct Splat {
    /// Position in the input CRS (transformed to ECEF-local during GLB write).
    pub position: [f64; 3],
    /// Linear RGB in [0, 1].
    pub rgb: [f32; 3],
    /// Per-axis scale. Isotropic for Phase 4a.
    pub scale: [f32; 3],
    /// Quaternion [x, y, z, w]. Identity for Phase 4a.
    pub rotation: [f32; 4],
    /// Opacity in [0, 1]. 1.0 for Phase 4a.
    pub opacity: f32,
    /// LOD tier (0 = coarsest). Used for progressive refinement.
    pub lod_tier: u8,
}

/// Loaded and spatially indexed splat collection.
pub struct SplatCloud {
    pub splats: Vec<Splat>,
    /// Splat indices grouped by grid cell: (column, row) -> Vec<splat_idx>.
    pub cell_index: HashMap<(usize, usize), Vec<usize>>,
    /// Splat indices grouped by (tier, column, row) for LOD queries.
    pub lod_cell_index: HashMap<(u8, usize, usize), Vec<usize>>,
    /// Number of LOD tiers (1 = no LOD).
    pub n_lod_tiers: u8,
    /// Point spacing at each tier (meters), for geometricError computation.
    pub tier_spacings: Vec<f64>,
}

impl SplatCloud {
    /// Get all splat indices belonging to the cells of a quadtree node.
    pub fn splats_for_cells(&self, cells: &[&CellId]) -> Vec<usize> {
        let mut indices = Vec::new();
        for cell in cells {
            if let Some(cell_splats) = self.cell_index.get(&(cell.column, cell.row)) {
                indices.extend_from_slice(cell_splats);
            }
        }
        indices
    }

    /// Get splat indices for a specific LOD tier in the given cells.
    pub fn splats_for_cells_at_lod(&self, cells: &[&CellId], lod_tier: u8) -> Vec<usize> {
        let mut indices = Vec::new();
        for cell in cells {
            if let Some(cell_splats) = self.lod_cell_index.get(&(lod_tier, cell.column, cell.row)) {
                indices.extend_from_slice(cell_splats);
            }
        }
        indices
    }
}

/// Try to extract an EPSG code from LAS VLR records.
///
/// Checks for OGC WKT (record ID 2112) and GeoTIFF keys (record ID 34735).
fn extract_las_epsg(header: &las::Header) -> Option<u32> {
    for vlr in header.vlrs() {
        // OGC Coordinate System WKT
        if vlr.record_id == 2112 {
            let wkt = String::from_utf8_lossy(&vlr.data);
            // Look for AUTHORITY["EPSG","XXXX"] or ID["EPSG",XXXX]
            for pattern in &[r#"AUTHORITY["EPSG","#, r#"ID["EPSG","#, r#"ID["EPSG","#] {
                if let Some(pos) = wkt.find(pattern) {
                    let after = &wkt[pos + pattern.len()..];
                    let code_str: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
                    if let Ok(code) = code_str.parse::<u32>() {
                        return Some(code);
                    }
                }
            }
        }
        // GeoTIFF GeoKeyDirectoryTag
        if vlr.record_id == 34735 && vlr.data.len() >= 8 {
            // GeoKey directory: 4 u16 header values, then key entries of 4 u16 each.
            // Key 3072 = ProjectedCSTypeGeoKey, Key 2048 = GeographicTypeGeoKey.
            let data = &vlr.data;
            let n_keys = u16::from_le_bytes([data[6], data[7]]) as usize;
            for i in 0..n_keys {
                let base = 8 + i * 8;
                if base + 8 > data.len() {
                    break;
                }
                let key_id = u16::from_le_bytes([data[base], data[base + 1]]);
                let value_offset = u16::from_le_bytes([data[base + 6], data[base + 7]]);
                // ProjectedCSTypeGeoKey or GeographicTypeGeoKey with tiff_tag_location=0
                // means the value is directly in value_offset.
                let tiff_tag_location = u16::from_le_bytes([data[base + 2], data[base + 3]]);
                if (key_id == 3072 || key_id == 2048) && tiff_tag_location == 0 && value_offset > 0
                {
                    return Some(value_offset as u32);
                }
            }
        }
    }
    None
}

/// Read the bounding box from a LAS/LAZ/COPC file header without reading points.
///
/// Returns `(min, max)` as `([x, y, z], [x, y, z])` in the file's native CRS.
pub fn read_las_bounds(path: &Path) -> Result<([f64; 3], [f64; 3])> {
    // Try COPC first, fall back to regular LAS.
    let header = match CopcReader::from_path(path) {
        Ok(reader) => {
            debug!("read_las_bounds: detected COPC format");
            reader.header().clone()
        }
        Err(_) => {
            debug!("read_las_bounds: using regular LAS reader");
            let reader = las::Reader::from_path(path)
                .with_context(|| format!("Open LAS/LAZ file {:?}", path))?;
            reader.header().clone()
        }
    };

    let bounds = header.bounds();
    Ok((
        [bounds.min.x, bounds.min.y, bounds.min.z],
        [bounds.max.x, bounds.max.y, bounds.max.z],
    ))
}

/// Process a single LAS point into a splat, returning `Ok(true)` if added,
/// `Ok(false)` if skipped (no color or noise-classified).
fn process_point(
    point: &las::point::Point,
    transformer: &Option<Proj>,
    splats: &mut Vec<Splat>,
    cell_index: &mut HashMap<(usize, usize), Vec<usize>>,
    lod_cell_index: &mut HashMap<(u8, usize, usize), Vec<usize>>,
    grid: &SquareGrid,
    lod_tier: u8,
    de_noising: bool,
    skipped_noise: &mut u64,
) -> Result<bool> {
    // ASPRS classification: 7 = low noise, 18 = high noise
    if de_noising {
        let class: u8 = point.classification.into();
        if class == 7 || class == 18 {
            *skipped_noise += 1;
            return Ok(false);
        }
    }

    let color = match point.color {
        Some(c) => c,
        None => return Ok(false),
    };

    // LAS RGB is u16, downsample to linear f32 [0,1].
    let r = (color.red >> 8) as f32 / 255.0;
    let g = (color.green >> 8) as f32 / 255.0;
    let b = (color.blue >> 8) as f32 / 255.0;

    let (x, y, z) = if let Some(ref proj) = transformer {
        proj.convert((point.x, point.y, point.z))
            .map_err(|e| anyhow::anyhow!("Reproject LAS point: {}", e))?
    } else {
        (point.x, point.y, point.z)
    };

    let splat_idx = splats.len();
    splats.push(Splat {
        position: [x, y, z],
        rgb: [r, g, b],
        scale: [0.0, 0.0, 0.0], // placeholder, computed after all points
        rotation: [0.0, 0.0, 0.0, 1.0], // identity quaternion
        opacity: 1.0,
        lod_tier,
    });

    let cell = grid.locate_point(&[x, y]);
    cell_index
        .entry((cell.column, cell.row))
        .or_default()
        .push(splat_idx);
    lod_cell_index
        .entry((lod_tier, cell.column, cell.row))
        .or_default()
        .push(splat_idx);

    Ok(true)
}

/// Load a LAS/LAZ/COPC file with per-point RGB, generate splats, build cell index.
///
/// Automatically detects COPC files and uses the appropriate reader.
/// Points without RGB are skipped with a warning. If the LAS CRS differs from
/// `target_crs`, points are reprojected. If CRS cannot be determined, we assume
/// it matches `target_crs`.
pub fn load_las_as_splats(
    path: &Path,
    target_crs: &str,
    grid: &SquareGrid,
    n_lod_tiers: u8,
    de_noising: bool,
) -> Result<SplatCloud> {
    // Try COPC first, fall back to regular LAS/LAZ.
    match CopcReader::from_path(path) {
        Ok(copc_reader) => {
            info!("Detected COPC format, using COPC reader");
            load_copc_as_splats(copc_reader, path, target_crs, grid, n_lod_tiers, de_noising)
        }
        Err(copc_err) => {
            debug!("Not a COPC file ({}), trying regular LAS reader", copc_err);
            load_regular_las_as_splats(path, target_crs, grid, n_lod_tiers, de_noising)
        }
    }
}

/// Load a COPC file using the vendored COPC reader.
fn load_copc_as_splats(
    mut reader: CopcReader<BufReader<std::fs::File>>,
    path: &Path,
    target_crs: &str,
    grid: &SquareGrid,
    n_lod_tiers: u8,
    de_noising: bool,
) -> Result<SplatCloud> {
    let header = reader.header().clone();
    log_las_header(&header);

    let transformer = build_transformer(&header, target_crs)?;

    let max_level = reader.max_octree_level();
    let spacing = reader.copc_info().spacing;
    info!("COPC octree: max level {}, root spacing {:.3}m", max_level, spacing);

    // Compute tier spacings: spacing at tier T = spacing / 2^(tier_max_copc_level)
    let tier_spacings: Vec<f64> = (0..n_lod_tiers)
        .map(|t| {
            // Each tier covers a range of COPC levels.
            // Tier boundary = (t+1) * max_level / n_tiers
            let tier_max = ((t as i32 + 1) * max_level / n_lod_tiers as i32).min(max_level);
            spacing / 2_f64.powi(tier_max)
        })
        .collect();
    for (t, s) in tier_spacings.iter().enumerate() {
        info!("  LOD tier {}: spacing {:.4}m", t, s);
    }

    let mut splats = Vec::with_capacity(header.number_of_points() as usize);
    let mut cell_index: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    let mut lod_cell_index: HashMap<(u8, usize, usize), Vec<usize>> = HashMap::new();
    let mut skipped_no_color = 0u64;
    let mut skipped_noise = 0u64;

    let points = reader
        .points(
            crate::copc_reader::LodSelection::All,
            crate::copc_reader::BoundsSelection::All,
        )
        .context("Create COPC point iterator")?;

    for (point, octree_level) in points {
        // Map COPC octree level to LOD tier
        let tier = if n_lod_tiers <= 1 || max_level == 0 {
            0u8
        } else {
            ((octree_level as f32 / max_level as f32) * n_lod_tiers as f32)
                .floor()
                .min((n_lod_tiers - 1) as f32) as u8
        };
        if process_point(&point, &transformer, &mut splats, &mut cell_index, &mut lod_cell_index, grid, tier, de_noising, &mut skipped_noise)? {
            // counted
        } else {
            skipped_no_color += 1;
        }
    }

    finalize_splat_cloud(splats, cell_index, lod_cell_index, skipped_no_color, skipped_noise, &header, path, n_lod_tiers, tier_spacings)
}

/// Load a regular LAS/LAZ file using the las crate.
fn load_regular_las_as_splats(
    path: &Path,
    target_crs: &str,
    grid: &SquareGrid,
    n_lod_tiers: u8,
    de_noising: bool,
) -> Result<SplatCloud> {
    let mut reader =
        las::Reader::from_path(path).with_context(|| format!("Open LAS/LAZ file {:?}", path))?;
    let header = reader.header().clone();
    log_las_header(&header);

    let transformer = build_transformer(&header, target_crs)?;

    let mut splats = Vec::with_capacity(header.number_of_points() as usize);
    let mut cell_index: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    let mut lod_cell_index: HashMap<(u8, usize, usize), Vec<usize>> = HashMap::new();
    let mut skipped_no_color = 0u64;
    let mut skipped_noise = 0u64;
    let mut point_counter = 0u64;

    // For regular LAS: assign tiers by strided subsampling.
    // Tier 0 gets every stride-th point, tier 1 fills in, etc.
    let stride = 1u64 << (n_lod_tiers - 1); // 2^(n_tiers-1)

    for point_result in reader.points() {
        let point = point_result.context("Read LAS point")?;
        // Determine tier: count trailing zeros in (point_counter % stride)
        let tier = if n_lod_tiers <= 1 {
            0u8
        } else if point_counter % stride == 0 {
            0u8
        } else {
            // tier = n_tiers - 1 - trailing_zeros(point_counter)
            let tz = point_counter.trailing_zeros() as u8;
            (n_lod_tiers - 1).saturating_sub(tz)
        };
        if process_point(&point, &transformer, &mut splats, &mut cell_index, &mut lod_cell_index, grid, tier, de_noising, &mut skipped_noise)? {
            point_counter += 1;
        } else {
            skipped_no_color += 1;
        }
    }

    // Compute tier spacings from average 2D spacing
    let bounds = header.bounds();
    let area = (bounds.max.x - bounds.min.x) * (bounds.max.y - bounds.min.y);
    let avg_spacing = if point_counter > 0 && area > 0.0 {
        (area / point_counter as f64).sqrt()
    } else {
        1.0
    };
    let tier_spacings: Vec<f64> = (0..n_lod_tiers)
        .map(|t| {
            let tier_stride = 1u64 << (n_lod_tiers - 1 - t);
            avg_spacing * (tier_stride as f64).sqrt()
        })
        .collect();

    finalize_splat_cloud(splats, cell_index, lod_cell_index, skipped_no_color, skipped_noise, &header, path, n_lod_tiers, tier_spacings)
}

fn log_las_header(header: &las::Header) {
    info!(
        "LAS file: version {}.{}, point format {}, {} points",
        header.version().major,
        header.version().minor,
        header.point_format().to_u8().unwrap_or(0),
        header.number_of_points()
    );
}

fn build_transformer(header: &las::Header, target_crs: &str) -> Result<Option<Proj>> {
    match extract_las_epsg(header) {
        Some(epsg) => {
            let las_crs = format!("EPSG:{}", epsg);
            if las_crs == target_crs {
                info!("LAS CRS {} matches target CRS, no reprojection needed", las_crs);
                Ok(None)
            } else {
                info!(
                    "LAS CRS {} differs from target CRS {}, will reproject",
                    las_crs, target_crs
                );
                Ok(Some(
                    Proj::new_known_crs(&las_crs, target_crs, None)
                        .map_err(|e| anyhow::anyhow!("Create LAS→target CRS transformer: {}", e))?,
                ))
            }
        }
        None => {
            warn!("Could not extract CRS from LAS VLR records, assuming same CRS as input data ({})", target_crs);
            Ok(None)
        }
    }
}

fn finalize_splat_cloud(
    mut splats: Vec<Splat>,
    cell_index: HashMap<(usize, usize), Vec<usize>>,
    lod_cell_index: HashMap<(u8, usize, usize), Vec<usize>>,
    skipped_no_color: u64,
    skipped_noise: u64,
    header: &las::Header,
    path: &Path,
    n_lod_tiers: u8,
    tier_spacings: Vec<f64>,
) -> Result<SplatCloud> {
    if skipped_noise > 0 {
        info!(
            "Skipped {} noise-classified points (ASPRS classes 7/18)",
            skipped_noise
        );
    }
    if skipped_no_color > 0 {
        warn!(
            "Skipped {} LAS points without RGB color data",
            skipped_no_color
        );
    }

    if splats.is_empty() {
        bail!("No colored points found in LAS file {:?}. Ensure the file contains RGB data (point format 2, 3, 5, 7, 8, or 10).", path);
    }

    info!("Loaded {} colored LiDAR points as splats", splats.len());

    // Compute per-tier splat scale: coarser tiers get larger splats for visual coverage.
    let base_scale = estimate_splat_scale(header, splats.len());
    info!("Base isotropic splat scale: {:.4}", base_scale);
    for splat in &mut splats {
        let tier_multiplier = if n_lod_tiers > 1 {
            2_f32.powi((n_lod_tiers - 1 - splat.lod_tier) as i32)
        } else {
            1.0
        };
        let s = base_scale * tier_multiplier;
        splat.scale = [s, s, s];
    }

    // Log tier distribution
    if n_lod_tiers > 1 {
        for t in 0..n_lod_tiers {
            let count = splats.iter().filter(|s| s.lod_tier == t).count();
            let tier_scale = base_scale * 2_f32.powi((n_lod_tiers - 1 - t) as i32);
            info!("  Tier {}: {} splats, scale {:.4}", t, count, tier_scale);
        }
    }

    Ok(SplatCloud {
        splats,
        cell_index,
        lod_cell_index,
        n_lod_tiers,
        tier_spacings,
    })
}

/// Estimate a reasonable isotropic splat scale from the point cloud density.
///
/// Uses the bounding box volume and point count to approximate average spacing.
/// Scale = half the average nearest-neighbor distance ≈ 0.5 × (volume / n)^(1/3).
fn estimate_splat_scale(header: &las::Header, n_points: usize) -> f32 {
    if n_points == 0 {
        return 0.1;
    }

    let bounds = header.bounds();
    let dx = bounds.max.x - bounds.min.x;
    let dy = bounds.max.y - bounds.min.y;

    // For mostly planar point clouds (buildings, terrain), use 2D area.
    let area = dx * dy;
    if area <= 0.0 {
        return 0.1;
    }

    // Average 2D spacing ≈ sqrt(area / n), scale = half of that.
    let avg_spacing = (area / n_points as f64).sqrt();
    let scale = (avg_spacing * 0.5) as f32;

    // Clamp to reasonable range.
    scale.clamp(0.01, 5.0)
}

/// Filter features to only those whose centroid falls within the LAS bounding box.
///
/// Converts the LAS bbox (real-world coordinates) to quantized coordinates using
/// the provided transform, then checks each feature's vertex centroid against it.
/// This avoids pre-subsetting GeoParquet files on disk.
pub fn filter_features_by_las_bbox(
    features: Vec<CityJSONFeatureVertices>,
    transform: &Transform,
    las_bbox_min: &[f64; 3],
    las_bbox_max: &[f64; 3],
) -> Vec<CityJSONFeatureVertices> {
    let before = features.len();

    let filtered: Vec<_> = features
        .into_iter()
        .filter(|f| {
            if f.vertices.is_empty() {
                return false;
            }
            // Compute centroid in real-world coordinates.
            let n = f.vertices.len() as f64;
            let cx = f.vertices.iter().map(|v| v[0] as f64).sum::<f64>() / n
                * transform.scale[0]
                + transform.translate[0];
            let cy = f.vertices.iter().map(|v| v[1] as f64).sum::<f64>() / n
                * transform.scale[1]
                + transform.translate[1];
            cx >= las_bbox_min[0]
                && cx <= las_bbox_max[0]
                && cy >= las_bbox_min[1]
                && cy <= las_bbox_max[1]
        })
        .collect();

    info!(
        "Spatial filter: {} of {} features within LAS bounding box",
        filtered.len(),
        before
    );
    filtered
}
