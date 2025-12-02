//! GeoParquet writer for CityJSON data.
//! Converts CityJSON features to GeoParquet format optimized for analytical queries.
// Copyright 2025
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;

use log::{debug, warn};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::Value;
use anyhow::{Result, Context};

use crate::parser::{CityJSONMetadata, Transform};

/// Configuration for GeoParquet conversion
#[derive(Debug, Clone)]
pub struct GeoParquetConfig {
    /// Attributes to include (if None, uses sensible defaults)
    pub include_attributes: Option<Vec<String>>,
    /// Attributes to exclude (in addition to defaults)
    pub exclude_attributes: Option<Vec<String>>,
    /// Output file path
    pub output_path: PathBuf,
    /// Compression algorithm
    /// - ZSTD (default): Modern algorithm with excellent balance of compression ratio and speed
    /// - Uncompressed: Fastest reads (no CPU overhead), largest files
    /// - Snappy: Very fast decompression, commonly used in Parquet
    /// - LZ4: Very fast decompression, similar to Snappy
    pub compression: Option<Compression>,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
#[clap(rename_all = "lower")]
pub enum Compression {
    /// ZSTD (Zstandard) - Modern compression with excellent balance (default)
    /// High compression ratios with fast compression/decompression speeds.
    /// Popular choice for geospatial data due to efficiency and performance.
    Zstd,
    /// No compression - fastest reads, largest files
    Uncompressed,
    /// Snappy compression - very fast decompression, commonly used in Parquet
    Snappy,
    /// LZ4 compression - very fast decompression
    LZ4,
}

impl Default for GeoParquetConfig {
    fn default() -> Self {
        Self {
            include_attributes: None,
            exclude_attributes: None,
            output_path: PathBuf::from("buildings.parquet"),
            compression: Some(Compression::Zstd), // ZSTD: excellent balance for geospatial data
        }
    }
}

/// Sensible default attributes for analytical queries
const DEFAULT_INCLUDED_ATTRIBUTES: &[&str] = &[
    "feature_id",
    "bldgarea",
    "heightmax",
    "heightmin",
    "elevmax",
    "elevmin",
    "rf_roof_n_planes",
    "rf_roof_type",
    "rf_is_glass_roof",
    "rf_volume_lod22",
];

/// Attributes to exclude (metadata/process info)
const DEFAULT_EXCLUDED_ATTRIBUTES: &[&str] = &[
    "acqtech", "acqtech_en", "acqtech_fr",
    "provider", "provideren", "providerfr",
    "qltylvl", "qltylvl_en", "qltylvl_fr",
    "haccmax", "haccmin", "vaccmax", "vaccmin",
    "rf_pc_select_", "rf_pc_source_", "rf_pc_year_", "rf_reconstruction_time_",
    "rf_val3dity_", "rf_rmse_",
    "rf_pt_density_", "rf_nodata_", "rf_h_pc_98p",
    "datemax", "datemin",
    "rf_extrusion_mode", "rf_force_lod11_", "rf_pointcloud_unusable", "rf_success_",
    "SHAPE_Area", "SHAPE_Leng",
];

/// Extended CityJSON feature structure that includes all fields we need
#[derive(Deserialize, Debug, Clone)]
#[allow(dead_code)] // Some fields are only used for deserialization
pub struct ExtendedCityJSONFeature {
    #[serde(rename = "CityObjects")]
    pub cityobjects: HashMap<String, ExtendedCityObject>,
    pub id: String,
    #[serde(rename = "type")]
    pub feature_type: String,
    pub vertices: Vec<[i64; 3]>,
}

#[derive(Deserialize, Debug, Clone)]
#[allow(dead_code)] // Some fields are only used for deserialization
pub struct ExtendedCityObject {
    #[serde(rename = "type")]
    pub cotype: String,
    pub attributes: Option<HashMap<String, Value>>,
    #[serde(rename = "geographicalExtent")]
    pub geographical_extent: Option<[f64; 6]>, // [minx, miny, minz, maxx, maxy, maxz]
    pub geometry: Option<Vec<ExtendedGeometry>>,
    pub children: Option<Vec<String>>,
    pub parents: Option<Vec<String>>,
}

#[derive(Deserialize, Debug, Clone)]
#[allow(dead_code)] // Some fields are only used for deserialization
pub struct ExtendedGeometry {
    #[serde(rename = "type")]
    pub geometry_type: String,
    pub lod: String,
    pub boundaries: Value, // Can be various structures
    pub semantics: Option<Semantics>,
}

#[derive(Deserialize, Debug, Clone)]
#[allow(dead_code)] // Some fields are only used for deserialization
pub struct Semantics {
    pub surfaces: Option<Vec<SemanticSurface>>,
    pub values: Option<Value>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct SemanticSurface {
    #[serde(rename = "type")]
    pub surface_type: String,
    #[serde(flatten)]
    pub attributes: HashMap<String, Value>,
}

/// Aggregated roof data from semantic surfaces
#[derive(Debug, Clone, Default)]
pub struct RoofData {
    pub n_planes: Option<i32>,
    pub roof_type: Option<String>,
    pub azimuths: Vec<f64>,
    pub slopes: Vec<f64>,
    pub elevation_min: Option<f64>,
    pub elevation_max: Option<f64>,
    pub elevation_50p: Option<f64>,
    pub elevation_70p: Option<f64>,
    pub is_glass: bool,
}

/// Surface counts
#[derive(Debug, Clone, Default)]
pub struct SurfaceCounts {
    pub n_ground: i32,
    pub n_wall: i32,
    pub n_roof: i32,
}

/// Convert CityJSON features to GeoParquet
pub fn convert_to_geoparquet(
    features_dir: &Path,
    metadata_path: &Path,
    config: GeoParquetConfig,
    object_types: Option<&[crate::parser::CityObjectType]>,
) -> Result<()> {
    debug!("Starting GeoParquet conversion");
    
    // Load metadata
    let metadata = CityJSONMetadata::from_file(metadata_path)
        .map_err(|e| anyhow::anyhow!("Failed to load CityJSON metadata: {}", e))?;
    
    // Find all JSONL files
    let feature_files = find_jsonl_files(features_dir)?;
    debug!("Found {} feature files", feature_files.len());
    
    // Process features in parallel
    let total_files = feature_files.len();
    
    debug!("Processing {} feature files in parallel...", total_files);
    
    // Use rayon to process files in parallel
    // Clone transform and config for parallel processing
    let transform = metadata.transform.clone();
    let config_clone = config.clone();
    let object_types_clone = object_types.map(|types| types.to_vec());
    
    let results: Vec<_> = feature_files
        .par_iter()
        .map(|feature_file| {
            process_feature_file(
                feature_file, 
                &transform, 
                &config_clone, 
                object_types_clone.as_deref()
            )
        })
        .collect();
    
    // Collect successful results
    // For debugging: fail on first error to see what's wrong
    let mut rows: Vec<BuildingRow> = Vec::new();
    for (idx, result) in results.into_iter().enumerate() {
        match result {
            Ok(Some(row)) => rows.push(row),
            Ok(None) => {
                // Skipped - no matching city objects
            }
            Err(e) => {
                // Fail immediately to see what's wrong
                return Err(anyhow::anyhow!("Failed to process feature file {:?}: {}", feature_files[idx], e));
            }
        }
    }
    
    debug!("Finished processing all {} feature files, extracted {} building features", total_files, rows.len());
    
    debug!("Processed {} building features", rows.len());
    
    // Write to GeoParquet
    write_geoparquet(&rows, &metadata, &config)
        .context("Failed to write GeoParquet file")?;
    
    Ok(())
}

fn find_jsonl_files(dir: &Path) -> Result<Vec<PathBuf>> {
    use walkdir::WalkDir;
    
    let mut files = Vec::new();
    for entry in WalkDir::new(dir) {
        let entry = entry?;
        if entry.file_type().is_file() {
            if let Some(ext) = entry.path().extension() {
                if ext == "jsonl" {
                    files.push(entry.path().to_path_buf());
                }
            }
        }
    }
    Ok(files)
}

fn process_feature_file(
    path: &Path,
    transform: &Transform,
    config: &GeoParquetConfig,
    object_types: Option<&[crate::parser::CityObjectType]>,
) -> Result<Option<BuildingRow>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let feature: ExtendedCityJSONFeature = serde_json::from_reader(reader)
        .context("Failed to parse CityJSON feature")?;
    
    // Find the main object matching the requested object types
    // For GeoParquet conversion, we only process Building objects (not BuildingPart)
    // because we only need the 2D footprint from geographicalExtent for geospatial analysis
    let has_building = object_types.map_or(true, |types| {
        types.iter().any(|t| matches!(t, crate::parser::CityObjectType::Building))
    });
    let building_related = has_building;
    
    // Only look for Building objects (skip BuildingPart entirely)
    let main_object = if building_related {
        feature.cityobjects.values()
            .find(|co| co.cotype == "Building")
    } else if let Some(types) = object_types {
        // For other object types, find the first matching one
        feature.cityobjects.values()
            .find(|co| {
                types.iter().any(|t| {
                    let type_str = format!("{}", t);
                    let expected_str = if type_str == "GenericCityObject" {
                        "+GenericCityObject"
                    } else {
                        &type_str
                    };
                    co.cotype == expected_str
                })
            })
    } else {
        None
    };
    
    if let Some(co) = main_object {
        let row = extract_building_row(co, &feature, transform, config)?;
        Ok(Some(row))
    } else {
        Ok(None)
    }
}

/// Building row data for GeoParquet
#[derive(Debug, Clone)]
struct BuildingRow {
    feature_id: String,
    cityobject_id: String,
    cityobject_type: String,
    geometry: geo_types::Polygon<f64>, // 2D footprint
    bldgarea: Option<f64>,
    heightmax: Option<f64>,
    heightmin: Option<f64>,
    elevmax: Option<f64>,
    elevmin: Option<f64>,
    volume_lod22: Option<f64>,
    roof_data: RoofData,
    surface_counts: SurfaceCounts,
    bbox_minx: f64,
    bbox_miny: f64,
    bbox_maxx: f64,
    bbox_maxy: f64,
    bbox_minz: f64,
    bbox_maxz: f64,
    #[allow(dead_code)] // Collected but not yet written to parquet (reserved for future use)
    additional_attributes: HashMap<String, Value>,
}

fn extract_building_row(
    co: &ExtendedCityObject,
    feature: &ExtendedCityJSONFeature,
    transform: &Transform,
    config: &GeoParquetConfig,
) -> Result<BuildingRow> {
    // Try to extract actual footprint from building geometry (GroundSurface)
    // This gives us the oriented footprint aligned with the building, not just a bounding box
    // For debugging: fail on first error to see what's wrong
    let (geometry, bbox) = match extract_footprint_from_geometry(co, feature, transform) {
        Ok(footprint) => {
            // Use the actual footprint from geometry
            debug!("Using oriented footprint from GroundSurface geometry for building {}", co.cotype);
            (footprint.0, footprint.1)
        }
        Err(e) => {
            // Fail immediately to see what's wrong
            return Err(anyhow::anyhow!("Failed to extract footprint from geometry for building {}: {}", 
                                     co.cotype, e));
        }
    };
    
    // Extract attributes
    let empty_attrs = HashMap::new();
    let attributes = co.attributes.as_ref().unwrap_or(&empty_attrs);
    
    // Filter attributes
    let additional_attributes = filter_attributes(attributes, config);
    
    // Extract building metrics
    let bldgarea = attributes.get("bldgarea")
        .and_then(|v| v.as_f64());
    let heightmax = attributes.get("heightmax")
        .and_then(|v| v.as_f64());
    let heightmin = attributes.get("heightmin")
        .and_then(|v| v.as_f64());
    let elevmax = attributes.get("elevmax")
        .and_then(|v| v.as_f64());
    let elevmin = attributes.get("elevmin")
        .and_then(|v| v.as_f64());
    let volume_lod22 = attributes.get("rf_volume_lod22")
        .and_then(|v| v.as_f64());
    
    // Extract feature_id
    let feature_id = attributes.get("feature_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| feature.id.to_string());
    
    // Aggregate semantic surface data
    let (roof_data, surface_counts) = aggregate_semantic_surfaces(co, feature);
    
    // Get cityobject ID (first key in cityobjects)
    let cityobject_id = feature.cityobjects.keys()
        .next()
        .map(|s| s.as_str())
        .unwrap_or("unknown")
        .to_string();
    
    Ok(BuildingRow {
        feature_id,
        cityobject_id,
        cityobject_type: co.cotype.clone(),
        geometry,
        bldgarea,
        heightmax,
        heightmin,
        elevmax,
        elevmin,
        volume_lod22,
        roof_data,
        surface_counts,
        bbox_minx: bbox[0],
        bbox_miny: bbox[1],
        bbox_maxx: bbox[3],
        bbox_maxy: bbox[4],
        bbox_minz: bbox[2],
        bbox_maxz: bbox[5],
        additional_attributes,
    })
}

/// Filter attributes based on sensible defaults and configuration
pub(crate) fn filter_attributes(
    attributes: &HashMap<String, Value>,
    config: &GeoParquetConfig,
) -> HashMap<String, Value> {
    let mut filtered = HashMap::new();
    
    // Determine which attributes to include
    let include_set: std::collections::HashSet<&str> = if let Some(ref include_list) = config.include_attributes {
        include_list.iter().map(|s| s.as_str()).collect()
    } else {
        DEFAULT_INCLUDED_ATTRIBUTES.iter().copied().collect()
    };
    
    // Determine which attributes to exclude
    let mut exclude_set: std::collections::HashSet<&str> = DEFAULT_EXCLUDED_ATTRIBUTES.iter().copied().collect();
    if let Some(ref exclude_list) = config.exclude_attributes {
        for attr in exclude_list {
            exclude_set.insert(attr.as_str());
        }
    }
    
    // Filter attributes
    for (key, value) in attributes {
        // Check if key should be excluded
        let should_exclude = exclude_set.iter().any(|&excluded| key.starts_with(excluded));
        
        if !should_exclude {
            // Check if key should be included (if include list is specified)
            if config.include_attributes.is_none() || include_set.contains(key.as_str()) {
                filtered.insert(key.clone(), value.clone());
            }
        }
    }
    
    filtered
}

/// Extract 2D footprint from building geometry (GroundSurface)
/// Returns (polygon, bbox) if successful, error if geometry not available
/// For debugging: fails on first error to see what's wrong
fn extract_footprint_from_geometry(
    co: &ExtendedCityObject,
    feature: &ExtendedCityJSONFeature,
    transform: &Transform,
) -> Result<(geo_types::Polygon<f64>, [f64; 6])> {
    // Check if geometry exists at all
    if co.geometry.is_none() {
        return Err(anyhow::anyhow!("Building has no geometry field"));
    }
    
    let geoms = co.geometry.as_ref().unwrap();
    debug!("Building has {} geometry entries", geoms.len());
    
    // Find geometry with highest LOD that has GroundSurface
    // Try LODs in order of preference: 2.2, 1.3, 1.2, 1, 0
    // If none match, use the first geometry that has semantics (any LOD)
    let geometry = geoms.iter()
        .find(|g| g.lod == "2.2")
        .or_else(|| geoms.iter().find(|g| g.lod == "1.3"))
        .or_else(|| geoms.iter().find(|g| g.lod == "1.2"))
        .or_else(|| geoms.iter().find(|g| g.lod == "1"))
        .or_else(|| geoms.iter().find(|g| g.lod == "0"))
        .or_else(|| geoms.iter().find(|g| g.semantics.is_some())); // Fallback: any geometry with semantics
    
    let geometry = match geometry {
        Some(g) => {
            debug!("Found geometry with LOD: {}", g.lod);
            g
        }
        None => {
            let available_lods: Vec<_> = geoms.iter().map(|g| g.lod.clone()).collect();
            return Err(anyhow::anyhow!("No geometry found with LOD 2.2, 1.3, 1.2, 1, or 0, and no geometry with semantics. Available LODs: {:?}", available_lods));
        }
    };
    
    // Check if geometry has semantics with GroundSurface
    // For LOD 0, there might be no semantics, so we'll extract from all surfaces
    let ground_indices: Vec<usize> = if let Some(semantics) = geometry.semantics.as_ref() {
        debug!("Geometry has semantics: surfaces={}, values={}", 
               semantics.surfaces.is_some(), semantics.values.is_some());
        
        // Try semantics.surfaces first (explicit surface definitions)
        if let Some(surfaces) = semantics.surfaces.as_ref() {
            debug!("Found {} semantic surfaces", surfaces.len());
            
            let indices: Vec<usize> = surfaces.iter()
                .enumerate()
                .filter(|(i, s)| {
                    let is_ground = s.surface_type == "GroundSurface";
                    if is_ground {
                        debug!("Found GroundSurface at index {}", i);
                    }
                    is_ground
                })
                .map(|(i, _)| i)
                .collect();
            
            if indices.is_empty() {
                let surface_types: Vec<_> = surfaces.iter().map(|s| s.surface_type.clone()).collect();
                debug!("No GroundSurface found in semantics. Available surface types: {:?}. Will try to extract from all surfaces.", surface_types);
                // Fall through to extract from all surfaces
                (0..surfaces.len()).collect()
            } else {
                debug!("Found {} GroundSurface indices", indices.len());
                indices
            }
        } else {
            debug!("Geometry uses semantics.values format (not yet supported). Will try to extract from all surfaces.");
            // We don't know which surfaces are ground, so we'll need to extract from boundaries
            // and find the lowest Z surface
            vec![] // Empty means we'll process all surfaces
        }
    } else {
        debug!("Geometry has no semantics (LOD 0). Will extract footprint from all surfaces (find lowest Z).");
        vec![] // Empty means we'll process all surfaces
    };
    
    // Parse boundaries - CityJSON uses MultiSurface format
    // boundaries is an array of surfaces, each surface is an array of rings
    let boundaries: Vec<Vec<Vec<usize>>> = serde_json::from_value::<Vec<Vec<Vec<usize>>>>(geometry.boundaries.clone())
        .map_err(|e| anyhow::anyhow!("Failed to parse boundaries: {}. Boundaries value: {:?}", e, geometry.boundaries))?;
    
    debug!("Successfully parsed {} boundaries", boundaries.len());
    
    // If no ground indices (no semantics or no GroundSurface), find the surface with lowest Z
    let surfaces_to_use: Vec<usize> = if ground_indices.is_empty() {
        debug!("No GroundSurface found, finding surface with lowest Z coordinates");
        // Calculate average Z for each surface and pick the one with lowest Z
        let mut surface_avg_z: Vec<(usize, f64)> = Vec::new();
        for (idx, surface) in boundaries.iter().enumerate() {
            if let Some(ring) = surface.first() {
                let mut z_sum = 0.0;
                let mut z_count = 0;
                for &vtx_idx in ring {
                    if vtx_idx < feature.vertices.len() {
                        let vtx = &feature.vertices[vtx_idx];
                        let z = (vtx[2] as f64 * transform.scale[2]) + transform.translate[2];
                        z_sum += z;
                        z_count += 1;
                    }
                }
                if z_count > 0 {
                    surface_avg_z.push((idx, z_sum / z_count as f64));
                }
            }
        }
        if surface_avg_z.is_empty() {
            return Err(anyhow::anyhow!("No valid surfaces found in boundaries"));
        }
        // Sort by Z and take the lowest (ground surface)
        surface_avg_z.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let lowest_idx = surface_avg_z[0].0;
        debug!("Selected surface {} as ground (lowest Z: {})", lowest_idx, surface_avg_z[0].1);
        vec![lowest_idx]
    } else {
        ground_indices
    };
    
    // Use the first surface from our list
    if let Some(&first_idx) = surfaces_to_use.first() {
        if first_idx < boundaries.len() {
            if let Some(ring) = boundaries[first_idx].first() {
                let mut exterior_points: Vec<(f64, f64)> = ring.iter()
                    .filter_map(|&vtx_idx| {
                        if vtx_idx < feature.vertices.len() {
                            let vtx = &feature.vertices[vtx_idx];
                            let x = (vtx[0] as f64 * transform.scale[0]) + transform.translate[0];
                            let y = (vtx[1] as f64 * transform.scale[1]) + transform.translate[1];
                            Some((x, y))
                        } else {
                            None
                        }
                    })
                    .collect();
                
                // Close the ring if not already closed
                if exterior_points.len() >= 3 {
                    if exterior_points.first() != exterior_points.last() {
                        exterior_points.push(*exterior_points.first().unwrap());
                    }
                    
                    // Create polygon
                    let exterior = geo_types::LineString::from(exterior_points);
                    let polygon = geo_types::Polygon::new(exterior, vec![]);
                    
                    // Calculate bbox
                    let bbox = if let Some(extent) = co.geographical_extent {
                        let minx = (extent[0] * transform.scale[0]) + transform.translate[0];
                        let miny = (extent[1] * transform.scale[1]) + transform.translate[1];
                        let minz = (extent[2] * transform.scale[2]) + transform.translate[2];
                        let maxx = (extent[3] * transform.scale[0]) + transform.translate[0];
                        let maxy = (extent[4] * transform.scale[1]) + transform.translate[1];
                        let maxz = (extent[5] * transform.scale[2]) + transform.translate[2];
                        [minx, miny, minz, maxx, maxy, maxz]
                    } else {
                        // Calculate from polygon and vertices
                        let mut minx = f64::INFINITY;
                        let mut miny = f64::INFINITY;
                        let mut minz = f64::INFINITY;
                        let mut maxx = f64::NEG_INFINITY;
                        let mut maxy = f64::NEG_INFINITY;
                        let mut maxz = f64::NEG_INFINITY;
                        for point in polygon.exterior().points() {
                            minx = minx.min(point.x());
                            miny = miny.min(point.y());
                            maxx = maxx.max(point.x());
                            maxy = maxy.max(point.y());
                        }
                        // Calculate Z bounds from ring vertices
                        for &vtx_idx in ring {
                            if vtx_idx < feature.vertices.len() {
                                let vtx = &feature.vertices[vtx_idx];
                                let z = (vtx[2] as f64 * transform.scale[2]) + transform.translate[2];
                                minz = minz.min(z);
                                maxz = maxz.max(z);
                            }
                        }
                        [minx, miny, minz, maxx, maxy, maxz]
                    };
                    
                    return Ok((polygon, bbox));
                } else {
                    return Err(anyhow::anyhow!("Exterior ring has insufficient points: {}", exterior_points.len()));
                }
            } else {
                return Err(anyhow::anyhow!("Ground surface at index {} has no rings", first_idx));
            }
        } else {
            return Err(anyhow::anyhow!("Ground surface index {} is out of bounds (boundaries length: {})", first_idx, boundaries.len()));
        }
    } else {
        return Err(anyhow::anyhow!("No ground indices found (this should not happen)"));
    }
}

fn aggregate_semantic_surfaces(
    co: &ExtendedCityObject,
    _feature: &ExtendedCityJSONFeature,
) -> (RoofData, SurfaceCounts) {
    let mut roof_data = RoofData::default();
    let mut surface_counts = SurfaceCounts::default();
    
    // Find geometry with highest LOD (prefer 2.2, then 1.3, then 1.2)
    let geometry = co.geometry.as_ref().and_then(|geoms| {
        geoms.iter()
            .find(|g| g.lod == "2.2")
            .or_else(|| geoms.iter().find(|g| g.lod == "1.3"))
            .or_else(|| geoms.iter().find(|g| g.lod == "1.2"))
    });
    
    if let Some(geom) = geometry {
        if let Some(ref semantics) = geom.semantics {
            if let Some(ref surfaces) = semantics.surfaces {
                let mut roof_azimuths = Vec::new();
                let mut roof_slopes = Vec::new();
                let mut roof_elevations_min = Vec::new();
                let mut roof_elevations_max = Vec::new();
                let mut roof_elevations_50p = Vec::new();
                let mut roof_elevations_70p = Vec::new();
                
                for surface in surfaces {
                    match surface.surface_type.as_str() {
                        "GroundSurface" => surface_counts.n_ground += 1,
                        "WallSurface" => surface_counts.n_wall += 1,
                        "RoofSurface" => {
                            surface_counts.n_roof += 1;
                            
                            // Extract roof attributes
                            if let Some(azimuth) = surface.attributes.get("rf_azimuth")
                                .and_then(|v| v.as_f64()) {
                                roof_azimuths.push(azimuth);
                            }
                            if let Some(slope) = surface.attributes.get("rf_slope")
                                .and_then(|v| v.as_f64()) {
                                roof_slopes.push(slope);
                            }
                            if let Some(elev) = surface.attributes.get("rf_roof_elevation_min")
                                .and_then(|v| v.as_f64()) {
                                roof_elevations_min.push(elev);
                            }
                            if let Some(elev) = surface.attributes.get("rf_roof_elevation_max")
                                .and_then(|v| v.as_f64()) {
                                roof_elevations_max.push(elev);
                            }
                            if let Some(elev) = surface.attributes.get("rf_roof_elevation_50p")
                                .and_then(|v| v.as_f64()) {
                                roof_elevations_50p.push(elev);
                            }
                            if let Some(elev) = surface.attributes.get("rf_roof_elevation_70p")
                                .and_then(|v| v.as_f64()) {
                                roof_elevations_70p.push(elev);
                            }
                            if let Some(is_glass) = surface.attributes.get("rf_is_glass_roof")
                                .and_then(|v| v.as_bool()) {
                                roof_data.is_glass = roof_data.is_glass || is_glass;
                            }
                        }
                        _ => {}
                    }
                }
                
                // Aggregate roof data
                roof_data.azimuths = roof_azimuths;
                roof_data.slopes = roof_slopes;
                roof_data.elevation_min = roof_elevations_min.iter().copied().reduce(f64::min);
                roof_data.elevation_max = roof_elevations_max.iter().copied().reduce(f64::max);
                roof_data.elevation_50p = roof_elevations_50p.iter().copied().reduce(|a, b| a + b)
                    .map(|sum| sum / roof_elevations_50p.len() as f64);
                roof_data.elevation_70p = roof_elevations_70p.iter().copied().reduce(|a, b| a + b)
                    .map(|sum| sum / roof_elevations_70p.len() as f64);
            }
        }
    }
    
    // Get roof metadata from attributes
    if let Some(ref attrs) = co.attributes {
        if let Some(n_planes) = attrs.get("rf_roof_n_planes")
            .and_then(|v| v.as_i64())
            .map(|n| n as i32) {
            roof_data.n_planes = Some(n_planes);
        }
        if let Some(roof_type) = attrs.get("rf_roof_type")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()) {
            roof_data.roof_type = Some(roof_type);
        }
        if let Some(is_glass) = attrs.get("rf_is_glass_roof")
            .and_then(|v| v.as_bool()) {
            roof_data.is_glass = roof_data.is_glass || is_glass;
        }
    }
    
    (roof_data, surface_counts)
}

fn write_geoparquet(
    rows: &[BuildingRow],
    metadata: &CityJSONMetadata,
    config: &GeoParquetConfig,
) -> Result<()> {
    use arrow::array::*;
    use arrow::datatypes::*;
    use arrow::record_batch::RecordBatch;
    
    debug!("Writing {} rows to GeoParquet file: {:?}", rows.len(), config.output_path);
    
    if rows.is_empty() {
        return Err(anyhow::anyhow!("No rows to write"));
    }
    
    // Pre-allocate vectors with known capacity to avoid reallocations
    let capacity = rows.len();
    debug!("Building Arrow arrays for {} rows...", capacity);
    
    // Build GeoArrow PolygonArray for geometry column (required by encoder API)
    use geoarrow::array::PolygonArray;
    use geoarrow_array::builder::PolygonBuilder;
    use geoarrow_array::IntoArrow;
    use geoarrow_schema::{PolygonType, Dimension, Metadata};
    use geoarrow_schema::crs::Crs;
    
    // Extract CRS from CityJSON metadata and create GeoArrow metadata
    // The EPSG code is read from metadata.metadata.reference_system (e.g., "EPSG:2961" for Halifax)
    // Use PROJJSON format so DefaultCrsTransform can extract it properly
    let geoarrow_crs = metadata.metadata.reference_system.to_epsg()
        .map_err(|e| anyhow::anyhow!("Failed to extract EPSG code from CRS: {}", e))
        .and_then(|epsg_code| {
            // Create PROJJSON for the EPSG code - use a more complete structure
            // that includes conversion_accuracy for better compatibility
            use serde_json::json;
            let projjson = json!({
                "type": "ProjectedCRS",
                "id": {
                    "authority": "EPSG",
                    "code": epsg_code
                },
                "conversion_accuracy": "unknown"
            });
            debug!("Created PROJJSON CRS for EPSG:{}: {}", epsg_code, serde_json::to_string(&projjson).unwrap_or_default());
            // Use from_projjson so the crate can extract it properly
            Ok(Crs::from_projjson(projjson))
        })
        .unwrap_or_else(|_| {
            // Fallback to default CRS if extraction fails
            warn!("Failed to extract CRS from metadata, using default (no CRS)");
            Crs::default()
        });
    
    let geoarrow_metadata = Arc::new(Metadata::new(geoarrow_crs.clone(), None));
    let polygon_type = PolygonType::new(Dimension::XY, geoarrow_metadata.clone());
    
    // Debug: Verify CRS is set in GeoArrow metadata (encoder will extract it automatically)
    debug!("GeoArrow CRS: {:?}", geoarrow_crs);
    let mut geometry_builder = PolygonBuilder::new(polygon_type.clone());
    let mut feature_ids = Vec::with_capacity(capacity);
    let mut cityobject_ids = Vec::with_capacity(capacity);
    let mut cityobject_types = Vec::with_capacity(capacity);
    let mut bldgareas = Vec::with_capacity(capacity);
    let mut heightmaxs = Vec::with_capacity(capacity);
    let mut heightmins = Vec::with_capacity(capacity);
    let mut elevmaxs = Vec::with_capacity(capacity);
    let mut elevmins = Vec::with_capacity(capacity);
    let mut volume_lod22s = Vec::with_capacity(capacity);
    let mut roof_n_planes = Vec::with_capacity(capacity);
    let mut roof_types = Vec::with_capacity(capacity);
    let mut roof_azimuths = Vec::with_capacity(capacity);
    let mut roof_slopes = Vec::with_capacity(capacity);
    let mut roof_elevation_mins = Vec::with_capacity(capacity);
    let mut roof_elevation_maxs = Vec::with_capacity(capacity);
    let mut roof_elevation_50ps = Vec::with_capacity(capacity);
    let mut roof_elevation_70ps = Vec::with_capacity(capacity);
    let mut roof_is_glass = Vec::with_capacity(capacity);
    let mut n_ground_surfaces = Vec::with_capacity(capacity);
    let mut n_wall_surfaces = Vec::with_capacity(capacity);
    let mut n_roof_surfaces = Vec::with_capacity(capacity);
    let mut bbox_minxs = Vec::with_capacity(capacity);
    let mut bbox_minys = Vec::with_capacity(capacity);
    let mut bbox_maxxs = Vec::with_capacity(capacity);
    let mut bbox_maxys = Vec::with_capacity(capacity);
    let mut bbox_minzs = Vec::with_capacity(capacity);
    let mut bbox_maxzs = Vec::with_capacity(capacity);
    
    let array_build_interval = if capacity > 10000 { 1000 } else if capacity > 1000 { 100 } else { 50 };
    debug!("Building arrays (progress every {} rows)...", array_build_interval);
    
    for (idx, row) in rows.iter().enumerate() {
        if idx > 0 && idx % array_build_interval == 0 {
            debug!("Array building progress: {}/{} rows ({}%)", 
                   idx, capacity, (idx * 100) / capacity);
        }
        
        // Verify geometry is valid before processing
        let exterior = row.geometry.exterior();
        let exterior_len = exterior.0.len();
        if exterior_len < 4 {
            warn!("Polygon {} has invalid exterior ring length: {} (expected at least 4 points). Skipping row.", 
                  row.feature_id, exterior_len);
            continue; // Skip entire row if geometry is invalid
        }
        
        // Verify polygon has valid coordinates (not NaN or infinite)
        let has_invalid_coords = exterior.0.iter().any(|p| !p.x.is_finite() || !p.y.is_finite());
        if has_invalid_coords {
            warn!("Polygon {} has invalid coordinates (NaN or infinite). Skipping row.", row.feature_id);
            continue; // Skip entire row if coordinates are invalid
        }
        
        // All arrays must have the same length, so we add data for all fields
        feature_ids.push(Some(row.feature_id.clone()));
        cityobject_ids.push(Some(row.cityobject_id.clone()));
        cityobject_types.push(Some(row.cityobject_type.clone()));
        
        // Push geometry to GeoArrow builder (encoder API requires GeoArrow input)
        geometry_builder.push_polygon(Some(&row.geometry))
            .map_err(|e| anyhow::anyhow!("Failed to push polygon to builder: {} (polygon extent: [{:.2}, {:.2}] to [{:.2}, {:.2}], exterior points: {})", 
                e, row.bbox_minx, row.bbox_miny, row.bbox_maxx, row.bbox_maxy, exterior_len))?;
        
        bldgareas.push(row.bldgarea);
        heightmaxs.push(row.heightmax);
        heightmins.push(row.heightmin);
        elevmaxs.push(row.elevmax);
        elevmins.push(row.elevmin);
        volume_lod22s.push(row.volume_lod22);
        
        roof_n_planes.push(row.roof_data.n_planes.map(|n| n as i64));
        roof_types.push(row.roof_data.roof_type.clone());
        
        // Store arrays as JSON strings for now (GeoParquet supports nested arrays)
        // TODO: Use proper Arrow ListArray when geoparquet supports it
        roof_azimuths.push(if row.roof_data.azimuths.is_empty() {
            None
        } else {
            Some(serde_json::to_string(&row.roof_data.azimuths)?)
        });
        roof_slopes.push(if row.roof_data.slopes.is_empty() {
            None
        } else {
            Some(serde_json::to_string(&row.roof_data.slopes)?)
        });
        
        roof_elevation_mins.push(row.roof_data.elevation_min);
        roof_elevation_maxs.push(row.roof_data.elevation_max);
        roof_elevation_50ps.push(row.roof_data.elevation_50p);
        roof_elevation_70ps.push(row.roof_data.elevation_70p);
        roof_is_glass.push(Some(row.roof_data.is_glass));
        
        n_ground_surfaces.push(Some(row.surface_counts.n_ground as i64));
        n_wall_surfaces.push(Some(row.surface_counts.n_wall as i64));
        n_roof_surfaces.push(Some(row.surface_counts.n_roof as i64));
        
        bbox_minxs.push(Some(row.bbox_minx));
        bbox_minys.push(Some(row.bbox_miny));
        bbox_maxxs.push(Some(row.bbox_maxx));
        bbox_maxys.push(Some(row.bbox_maxy));
        bbox_minzs.push(Some(row.bbox_minz));
        bbox_maxzs.push(Some(row.bbox_maxz));
    }
    
    debug!("Finished building arrays, creating Arrow arrays and schema...");
    
    // Finish building the GeoArrow PolygonArray (required by encoder API)
    let polygon_array: PolygonArray = geometry_builder.finish();
    // Keep the PolygonArray as-is - don't convert to generic Arrow array yet
    // The encoder needs to see it as a GeoArrow type to recognize it as a geometry column
    
    // Create Arrow arrays
    let feature_id_array = Arc::new(StringArray::from(feature_ids)) as Arc<dyn arrow::array::Array>;
    let cityobject_id_array = Arc::new(StringArray::from(cityobject_ids)) as Arc<dyn arrow::array::Array>;
    let cityobject_type_array = Arc::new(StringArray::from(cityobject_types)) as Arc<dyn arrow::array::Array>;
    
    let bldgarea_array = Arc::new(Float64Array::from(bldgareas)) as Arc<dyn arrow::array::Array>;
    let heightmax_array = Arc::new(Float64Array::from(heightmaxs)) as Arc<dyn arrow::array::Array>;
    let heightmin_array = Arc::new(Float64Array::from(heightmins)) as Arc<dyn arrow::array::Array>;
    let elevmax_array = Arc::new(Float64Array::from(elevmaxs)) as Arc<dyn arrow::array::Array>;
    let elevmin_array = Arc::new(Float64Array::from(elevmins)) as Arc<dyn arrow::array::Array>;
    let volume_lod22_array = Arc::new(Float64Array::from(volume_lod22s)) as Arc<dyn arrow::array::Array>;
    
    let roof_n_planes_array = Arc::new(Int64Array::from(roof_n_planes)) as Arc<dyn arrow::array::Array>;
    let roof_type_array = Arc::new(StringArray::from(roof_types)) as Arc<dyn arrow::array::Array>;
    let roof_azimuth_array = Arc::new(StringArray::from(roof_azimuths)) as Arc<dyn arrow::array::Array>;
    let roof_slope_array = Arc::new(StringArray::from(roof_slopes)) as Arc<dyn arrow::array::Array>;
    let roof_elevation_min_array = Arc::new(Float64Array::from(roof_elevation_mins)) as Arc<dyn arrow::array::Array>;
    let roof_elevation_max_array = Arc::new(Float64Array::from(roof_elevation_maxs)) as Arc<dyn arrow::array::Array>;
    let roof_elevation_50p_array = Arc::new(Float64Array::from(roof_elevation_50ps)) as Arc<dyn arrow::array::Array>;
    let roof_elevation_70p_array = Arc::new(Float64Array::from(roof_elevation_70ps)) as Arc<dyn arrow::array::Array>;
    let roof_is_glass_array = Arc::new(BooleanArray::from(roof_is_glass)) as Arc<dyn arrow::array::Array>;
    
    let n_ground_array = Arc::new(Int64Array::from(n_ground_surfaces)) as Arc<dyn arrow::array::Array>;
    let n_wall_array = Arc::new(Int64Array::from(n_wall_surfaces)) as Arc<dyn arrow::array::Array>;
    let n_roof_array = Arc::new(Int64Array::from(n_roof_surfaces)) as Arc<dyn arrow::array::Array>;
    
    let bbox_minx_array = Arc::new(Float64Array::from(bbox_minxs)) as Arc<dyn arrow::array::Array>;
    let bbox_miny_array = Arc::new(Float64Array::from(bbox_minys)) as Arc<dyn arrow::array::Array>;
    let bbox_maxx_array = Arc::new(Float64Array::from(bbox_maxxs)) as Arc<dyn arrow::array::Array>;
    let bbox_maxy_array = Arc::new(Float64Array::from(bbox_maxys)) as Arc<dyn arrow::array::Array>;
    let bbox_minz_array = Arc::new(Float64Array::from(bbox_minzs)) as Arc<dyn arrow::array::Array>;
    let bbox_maxz_array = Arc::new(Float64Array::from(bbox_maxzs)) as Arc<dyn arrow::array::Array>;
    
    // Create schema with GeoArrow geometry column
    // Use the PolygonType's to_field method to create a field with GeoArrow metadata
    let geometry_field = polygon_type.to_field("geometry", true);
    let fields = vec![
        Field::new("feature_id", DataType::Utf8, false),
        Field::new("cityobject_id", DataType::Utf8, false),
        Field::new("cityobject_type", DataType::Utf8, false),
        // Geometry field - use field from PolygonArray to preserve GeoArrow metadata
        geometry_field,
        Field::new("bldgarea", DataType::Float64, true),
        Field::new("heightmax", DataType::Float64, true),
        Field::new("heightmin", DataType::Float64, true),
        Field::new("elevmax", DataType::Float64, true),
        Field::new("elevmin", DataType::Float64, true),
        Field::new("volume_lod22", DataType::Float64, true),
        Field::new("roof_n_planes", DataType::Int64, true),
        Field::new("roof_type", DataType::Utf8, true),
        Field::new("roof_azimuths", DataType::Utf8, true), // JSON array as string for now
        Field::new("roof_slopes", DataType::Utf8, true), // JSON array as string for now
        Field::new("roof_elevation_min", DataType::Float64, true),
        Field::new("roof_elevation_max", DataType::Float64, true),
        Field::new("roof_elevation_50p", DataType::Float64, true),
        Field::new("roof_elevation_70p", DataType::Float64, true),
        Field::new("roof_is_glass", DataType::Boolean, true),
        Field::new("n_ground_surfaces", DataType::Int64, true),
        Field::new("n_wall_surfaces", DataType::Int64, true),
        Field::new("n_roof_surfaces", DataType::Int64, true),
        Field::new("bbox_minx", DataType::Float64, false),
        Field::new("bbox_miny", DataType::Float64, false),
        Field::new("bbox_maxx", DataType::Float64, false),
        Field::new("bbox_maxy", DataType::Float64, false),
        Field::new("bbox_minz", DataType::Float64, false),
        Field::new("bbox_maxz", DataType::Float64, false),
    ];
    
    let schema = Arc::new(Schema::new(fields));
    
    // Create record batch
    // Convert PolygonArray to Arrow array for the batch
    let geometry_array = Arc::new(polygon_array.into_arrow()) as Arc<dyn arrow::array::Array>;
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            feature_id_array,
            cityobject_id_array,
            cityobject_type_array,
            geometry_array,
            bldgarea_array,
            heightmax_array,
            heightmin_array,
            elevmax_array,
            elevmin_array,
            volume_lod22_array,
            roof_n_planes_array,
            roof_type_array,
            roof_azimuth_array,
            roof_slope_array,
            roof_elevation_min_array,
            roof_elevation_max_array,
            roof_elevation_50p_array,
            roof_elevation_70p_array,
            roof_is_glass_array,
            n_ground_array,
            n_wall_array,
            n_roof_array,
            bbox_minx_array,
            bbox_miny_array,
            bbox_maxx_array,
            bbox_maxy_array,
            bbox_minz_array,
            bbox_maxz_array,
        ],
    )?;
    
    // Write to GeoParquet using geoparquet crate's encoder API
    use geoparquet::writer::{GeoParquetRecordBatchEncoder, GeoParquetWriterOptionsBuilder, GeoParquetWriterEncoding};
    
    debug!("Creating GeoParquet encoder...");
    
    // Build writer options with WKB encoding
    let writer_options = GeoParquetWriterOptionsBuilder::default()
        .set_encoding(GeoParquetWriterEncoding::WKB)
        .set_primary_column("geometry".to_string())
        .build();
    
    // Create encoder - it will handle metadata automatically
    // The encoder identifies geometry columns from the schema field metadata
    let mut encoder = GeoParquetRecordBatchEncoder::try_new(&schema, &writer_options)
        .context("Failed to create GeoParquet encoder - check that geometry field has GeoArrow metadata")?;
    
    debug!("GeoParquet encoder created successfully. Geometry column should be recognized from schema metadata.");
    
    // Encode the batch (converts GeoArrow to WKB format and tracks metadata)
    debug!("Encoding record batch (converting GeoArrow to WKB)...");
    let encoded_batch = encoder.encode_record_batch(&batch)
        .context("Failed to encode record batch - geometry may not be recognized as GeoArrow type")?;
    
    debug!("Record batch encoded successfully. Geometry column is now in WKB format.");
    
    // Debug: Check the geometry column type in encoded batch
    if let Some(geometry_col) = encoded_batch.column_by_name("geometry") {
        debug!("Encoded geometry column type: {:?}", geometry_col.data_type());
        debug!("Encoded geometry column length: {}", geometry_col.len());
    } else {
        warn!("Geometry column not found in encoded batch!");
    }
    
    // Get the target schema for the parquet writer
    let target_schema = encoder.target_schema();
    
    // Debug: Log the target schema to verify geometry column type
    debug!("Target schema after encoding:");
    for field in target_schema.fields() {
        debug!("  Field: {} - Type: {:?}", field.name(), field.data_type());
        if field.name() == "geometry" {
            debug!("    Geometry field metadata: {:?}", field.metadata());
        }
    }
    
    // Get the GeoParquet metadata as key-value for parquet file
    // The encoder automatically extracts CRS from GeoArrow metadata using DefaultCrsTransform
    let geo_metadata_kv = encoder.into_keyvalue()
        .context("Failed to get GeoParquet metadata")?;
    
    // Debug: Log the GeoParquet metadata to verify geometry column is recognized
    debug!("GeoParquet metadata key: {}", geo_metadata_kv.key);
    if let Some(meta_str) = &geo_metadata_kv.value {
        debug!("GeoParquet metadata value (first 500 chars): {}", 
               meta_str.chars().take(500).collect::<String>());
        
        // Debug: Check if CRS is in the metadata
        if meta_str.contains("\"crs\"") {
            debug!("CRS found in GeoParquet metadata");
        } else {
            warn!("CRS not found in GeoParquet metadata - encoder should extract it from GeoArrow metadata");
        }
    } else {
        debug!("GeoParquet metadata value is None");
    }
    
    // Create writer properties with compression
    let compression = match config.compression {
        Some(Compression::Zstd) => parquet::basic::Compression::ZSTD(parquet::basic::ZstdLevel::default()),
        Some(Compression::Snappy) => parquet::basic::Compression::SNAPPY,
        Some(Compression::LZ4) => parquet::basic::Compression::LZ4,
        Some(Compression::Uncompressed) | None => parquet::basic::Compression::UNCOMPRESSED,
    };
    
    let props = parquet::file::properties::WriterProperties::builder()
        .set_compression(compression)
        .build();
    
    // Write using standard parquet writer
    debug!("Writing record batch to GeoParquet file (this may take a while for large datasets)...");
    use parquet::arrow::ArrowWriter;
    let file = std::fs::File::create(&config.output_path)?;
    let mut writer = ArrowWriter::try_new(file, target_schema, Some(props.into()))
        .context("Failed to create Parquet writer")?;
    
    // Write the encoded batch
    writer.write(&encoded_batch)
        .context("Failed to write record batch")?;
    
    // Add GeoParquet metadata before closing
    // This must be done before close() to ensure metadata is written to the file footer
    writer.append_key_value_metadata(geo_metadata_kv);
    
    // Close writer (this will finalize the file and write the footer with metadata)
    writer.close()
        .context("Failed to finalize GeoParquet file")?;
    
    debug!("Successfully wrote {} rows to GeoParquet file", rows.len());
    
    Ok(())
}


