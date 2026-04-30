use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::rc::Rc;

use ahash::AHashMap;
use anyhow::{bail, Context, Result};
use earcutr::earcut;
use gltf::json as json;

use crate::cli::TilesVersion;
use crate::las_source::SplatCloud;
use crate::material::MaterialConfig;
use crate::parser::{CityJSONFeatureVertices, CityObjectType, Geometry, Transform, World};
use crate::proj::Proj;
use crate::spatial_structs::{QuadTree, QuadTreeNodeId};

const GLTF_VERSION: &str = "2.0";

/// Create PBR material with specified metallic and roughness parameters.
/// Base color is white so per-vertex COLOR_0 provides the actual coloring.
fn create_material(base_color: &str, metallic: f32, roughness: f32) -> Result<json::Material, anyhow::Error> {
    let base_color_rgba = crate::material::hex_to_rgba(base_color)?;
    
    Ok(json::Material {
        name: None,
        extensions: Default::default(),
        extras: Default::default(),
        pbr_metallic_roughness: json::material::PbrMetallicRoughness {
            base_color_factor: json::material::PbrBaseColorFactor(base_color_rgba),
            metallic_factor: json::material::StrengthFactor(metallic),
            roughness_factor: json::material::StrengthFactor(roughness),
            base_color_texture: None,
            metallic_roughness_texture: None,
            extensions: Default::default(),
            extras: Default::default(),
        },
        normal_texture: None,
        occlusion_texture: None,
        emissive_texture: None,
        emissive_factor: json::material::EmissiveFactor([0.0, 0.0, 0.0]),
        alpha_mode: json::validation::Checked::Valid(json::material::AlphaMode::Opaque),
        alpha_cutoff: None,
        double_sided: true,
    })
}

pub fn write_tile_glb<P: AsRef<Path>>(
    world: &World,
    quadtree: &QuadTree,
    qtree_node_id: QuadTreeNodeId,
    output_path: P,
    material_config: &MaterialConfig,
    tiles_version: TilesVersion,
    lod_filter: Option<&str>,
    attribute_whitelist: Option<&std::collections::HashSet<&str>>,
    crs_from: &str,
    root_center_ecef: (f64, f64, f64),
    splat_cloud: Option<&SplatCloud>,
    splat_lod_tier: Option<u8>,
    splats_only: bool,
) -> Result<()> {
    let qtree_node = quadtree
        .node(&qtree_node_id)
        .context("Tile not present in quadtree")?;

    // Create per-tile PROJ transformer (Proj is !Send, so cannot be shared across threads).
    // Using thread_local! avoids recreating the transformer for every tile on the same thread.
    thread_local! {
        static TL_PROJ_ECEF: std::cell::RefCell<Option<(String, Proj)>> = const { std::cell::RefCell::new(None) };
        static TL_PROJ_ELL: std::cell::RefCell<Option<(String, Proj)>> = const { std::cell::RefCell::new(None) };
    }

    let transformer_to_ecef = TL_PROJ_ECEF.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.as_ref().map_or(true, |(k, _)| k != crs_from) {
            let proj = Proj::new_known_crs(crs_from, "EPSG:4978", None)
                .expect("Create CRS to ECEF transformer");
            *slot = Some((crs_from.to_string(), proj));
        }
        // SAFETY: we return a raw pointer to the Proj inside the RefCell.
        // This is safe because the Proj lives in thread-local storage and is only
        // accessed by the current thread. The pointer is valid for the duration of
        // write_tile_glb since we don't drop/replace it during this call.
        let ptr = &slot.as_ref().unwrap().1 as *const Proj;
        ptr
    });
    // SAFETY: see comment above — pointer to thread-local Proj, valid for this call.
    let transformer_to_ecef = unsafe { &*transformer_to_ecef };

    // Use tile center for vertical geoid correction (local to tile)
    let tile_bbox = qtree_node.bbox(&world.grid);
    let tile_center_original = [
        (tile_bbox[0] + tile_bbox[3]) * 0.5,
        (tile_bbox[1] + tile_bbox[4]) * 0.5,
        tile_bbox[2],
    ];

    let vertical_geoid_n = TL_PROJ_ELL.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.as_ref().map_or(true, |(k, _)| k != crs_from) {
            let proj = Proj::new_known_crs(crs_from, "EPSG:4979", None)
                .expect("Create CRS to ellipsoidal transformer");
            *slot = Some((crs_from.to_string(), proj));
        }
        let proj = &slot.as_ref().unwrap().1;
        proj.convert((
            tile_center_original[0],
            tile_center_original[1],
            tile_center_original[2],
        ))
        .map(|(_, _, h_ell)| h_ell - tile_center_original[2])
        .unwrap_or(0.0)
    });

    // For 3D Tiles with root transform, GLB coordinates must be in ECEF and relative to ROOT center in ECEF
    // This ensures coordinate system consistency: root transform (ECEF) + GLB content (ECEF) = correct positioning
    let mut builder = MeshBuilder::new(
        transformer_to_ecef,
        root_center_ecef,
        vertical_geoid_n,
        material_config.metallic_factor,
        material_config.roughness_factor,
    );

    // Deduplicate by feature id: a building can be referenced by multiple cells (e.g. bbox
    // intersection), so we add each feature at most once per tile to avoid duplicate geometry
    // and the same BuildingId appearing on multiple meshes.
    if !splats_only {
        let mut seen_fids: HashSet<usize> = HashSet::new();
        for cellid in qtree_node.cells() {
            let cell = world.grid.cell(cellid);
            for fid in cell.feature_ids.iter() {
                if !seen_fids.insert(*fid) {
                    continue; // already added this feature to this tile
                }
                // Borrow from in-memory data if available, otherwise read from file
                let cf_owned;
                let cf_ref = if let Some(cf) = world.feature_data.get(*fid).and_then(|o| o.as_ref()) {
                    cf
                } else {
                    let feature = &world.features[*fid];
                    cf_owned = CityJSONFeatureVertices::from_file(&feature.path_jsonl)
                        .map_err(|e| anyhow::anyhow!("Failed to read {:?}: {}", feature.path_jsonl, e))?;
                    &cf_owned
                };
                builder.add_feature(cf_ref, &world.transform, &material_config.color_map, lod_filter, attribute_whitelist, tiles_version)?;
            }
        }
    }

    // Collect splat indices for this tile.
    let tile_cells = qtree_node.cells();
    let tile_splat_indices = match (splat_cloud, splat_lod_tier) {
        (Some(cloud), Some(tier)) => cloud.splats_for_cells_at_lod(&tile_cells, tier),
        (Some(cloud), None) => cloud.splats_for_cells(&tile_cells),
        _ => Vec::new(),
    };

    builder.write_glb(output_path, tiles_version, splat_cloud, &tile_splat_indices)
}

/// Inferred column type for forwarded CityObject attributes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttrType {
    String,
    Float32,
    Int32,
    Boolean,
}

/// Tracks a serialized attribute column's buffer view locations.
struct AttrBufferView {
    col_name: String,
    col_type: AttrType,
    /// Byte offset of the data buffer view in the binary buffer.
    data_bv_offset: usize,
    /// Byte length of the data buffer view.
    data_bv_len: usize,
    /// For STRING columns: byte offset of the string offsets buffer view.
    offsets_bv_offset: Option<usize>,
    /// For STRING columns: byte length of the string offsets buffer view.
    offsets_bv_len: Option<usize>,
}

/// Byte offsets and metadata for Gaussian splat attribute buffers in the GLB binary.
struct SplatBufferData {
    count: usize,
    position_offset: usize,
    color_offset: usize,
    rotation_offset: usize,
    scale_offset: usize,
    opacity_offset: usize,
    sh0_offset: usize,
    pos_min: [f32; 3],
    pos_max: [f32; 3],
}

struct MeshBuilder<'a> {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    colors: Vec<[f32; 4]>,
    batch_ids: Vec<u32>,
    indices: Vec<u32>,
    next_batch_index: u32,
    /// CityJSON object id per batch index (batch_id_to_cityobject_id[i] = id for batch index i)
    batch_id_to_cityobject_id: Vec<Rc<String>>,
    /// CityObjectType per batch index (e.g. "Building", "WaterBody")
    batch_id_to_cityobject_type: Vec<Rc<String>>,
    /// Semantic surface type per batch index (e.g. "RoofSurface", "WallSurface")
    batch_id_to_surface_type: Vec<Rc<String>>,
    /// Forwarded CityObject attributes per batch index (one map per surface row).
    /// Each entry contains the merged attributes from the parent Building/BuildingPart.
    batch_id_to_attributes: Vec<Option<Rc<serde_json::Map<String, serde_json::Value>>>>,
    /// Computed surface area in m² per batch index.
    batch_id_to_area: Vec<f32>,
    /// Computed surface azimuth in degrees (0-360, north=0 clockwise) per batch index.
    batch_id_to_azimuth: Vec<f32>,
    /// Computed surface elevation in degrees from horizontal (0=vertical wall, 90=flat roof) per batch index.
    batch_id_to_elevation: Vec<f32>,
    transformer_to_ecef: &'a Proj,
    root_center_ecef: (f64, f64, f64),
    vertical_bias: f64,
    metallic_factor: f32,
    roughness_factor: f32,
}

impl<'a> MeshBuilder<'a> {
    fn new(
        transformer_to_ecef: &'a Proj,
        root_center_ecef: (f64, f64, f64),
        vertical_bias: f64,
        metallic_factor: f32,
        roughness_factor: f32,
    ) -> Self {
        Self {
            positions: Vec::new(),
            normals: Vec::new(),
            colors: Vec::new(),
            batch_ids: Vec::new(),
            indices: Vec::new(),
            next_batch_index: 0,
            batch_id_to_cityobject_id: Vec::new(),
            batch_id_to_cityobject_type: Vec::new(),
            batch_id_to_surface_type: Vec::new(),
            batch_id_to_attributes: Vec::new(),
            batch_id_to_area: Vec::new(),
            batch_id_to_azimuth: Vec::new(),
            batch_id_to_elevation: Vec::new(),
            transformer_to_ecef,
            root_center_ecef,
            vertical_bias,
            metallic_factor,
            roughness_factor,
        }
    }

    /// Add one feature (one CityJSON feature = one building or tree).
    /// For buildings: allocates a separate batch index per geometric surface for
    /// per-surface metadata in the 3D Tiles property table.
    /// For trees (SolitaryVegetationObject): emits geometry only with a sentinel
    /// batch ID (u32::MAX) — no property table rows, no computed props.
    fn add_feature(
        &mut self,
        feature: &CityJSONFeatureVertices,
        transform: &Transform,
        color_map: &HashMap<CityObjectType, [f32; 4]>,
        lod_filter: Option<&str>,
        attribute_whitelist: Option<&std::collections::HashSet<&str>>,
        tiles_version: TilesVersion,
    ) -> Result<()> {
        let building_id = feature
            .id
            .clone()
            .or_else(|| {
                feature
                    .cityobjects
                    .iter()
                    .find(|(_, co)| co.cotype == CityObjectType::Building)
                    .map(|(k, _)| k.clone())
            })
            .or_else(|| {
                feature
                    .cityobjects
                    .iter()
                    .find(|(_, co)| co.cotype == CityObjectType::BuildingPart)
                    .and_then(|(k, _)| k.strip_suffix("-0").map(|s| s.to_string()))
            })
            .unwrap_or_else(|| "unknown".to_string());

        // Resolve primary CityObjectType: Building > BuildingPart > first type
        let primary_type = feature
            .cityobjects
            .values()
            .find(|co| co.cotype == CityObjectType::Building)
            .or_else(|| {
                feature
                    .cityobjects
                    .values()
                    .find(|co| co.cotype == CityObjectType::BuildingPart)
            })
            .unwrap_or_else(|| feature.cityobjects.values().next().unwrap());
        let primary_type_str = primary_type.cotype.as_str().to_string();

        // Trees get geometry only — no per-surface metadata, no property table rows.
        // This avoids costly per-surface batch ID allocation, string pushes, and
        // azimuth/elevation trig for tens of thousands of tree surfaces.
        let skip_metadata = primary_type.cotype == CityObjectType::SolitaryVegetationObject
            && tiles_version == TilesVersion::V1_1;

        // Collect forwarded attributes: prefer Building attributes over BuildingPart.
        // Skip entirely for trees (they have no CityJSON attributes anyway).
        let merged_attributes: Option<Rc<serde_json::Map<String, serde_json::Value>>> = if skip_metadata {
            None
        } else {
            feature
            .cityobjects
            .values()
            .find(|co| co.cotype == CityObjectType::Building)
            .or_else(|| {
                feature
                    .cityobjects
                    .values()
                    .find(|co| co.cotype == CityObjectType::BuildingPart)
            })
            .and_then(|co| co.attributes.clone())
            .and_then(|mut attrs| {
                match attribute_whitelist {
                    Some(wl) => {
                        attrs.retain(|k, _| wl.contains(k.as_str()));
                        if attrs.is_empty() { None } else { Some(Rc::new(attrs)) }
                    }
                    None => None, // no flags set → no attributes forwarded
                }
            })
        };

        let default_color = [1.0_f32, 0.753, 0.796, 1.0]; // #FFC0CB pink
        let lod_filter_owned: Option<String> = lod_filter.map(|s| s.to_string());
        // Sentinel batch ID for trees: not in property table, viewers show no metadata.
        let tree_sentinel: u32 = u32::MAX;
        // Wrap per-feature strings in Rc to avoid cloning per surface (pointer increment vs heap copy).
        let building_id_rc = Rc::new(building_id);
        let primary_type_str_rc = Rc::new(primary_type_str);

        // Per-feature vertex cache: share vertices across surfaces of the same building.
        // This matches the behavior of the old geoflow-based tyler and avoids
        // duplicating vertices at surface boundaries (wall/roof/ground junctions).
        let mut feature_vertex_cache: AHashMap<usize, u32> = AHashMap::new();

        for (_, co) in feature.cityobjects.iter() {
            let color = color_map.get(&co.cotype).copied().unwrap_or(default_color);
            if let Some(geoms) = &co.geometry {
                // Determine which LoD to use for this CityObject:
                // - If --lod is set, use only that LoD (hoisted above the loop)
                // - Otherwise, use the highest LoD available per CityObject
                let effective_lod: Option<&str> = if lod_filter_owned.is_some() {
                    lod_filter_owned.as_deref()
                } else {
                    geoms.iter()
                        .filter_map(|g| g.lod())
                        .max_by(|a, b| a.parse::<f64>().unwrap_or(0.0)
                            .partial_cmp(&b.parse::<f64>().unwrap_or(0.0))
                            .unwrap_or(std::cmp::Ordering::Equal))
                };
                for geometry in geoms {
                    // Skip geometries that do not match the target LoD
                    if let Some(target) = effective_lod {
                        if geometry.lod() != Some(target) {
                            continue;
                        }
                    }
                    let semantics = geometry.semantics();
                    match geometry {
                        Geometry::MultiSurface { boundaries, .. } => {
                            for (surf_idx, surface) in boundaries.iter().enumerate() {
                                if skip_metadata {
                                    let _ = self.add_surface_cached(surface, &feature.vertices, transform, tree_sentinel, color, &mut feature_vertex_cache)?;
                                } else {
                                    let surface_type = semantics
                                        .and_then(|s| s.surface_type(0, surf_idx))
                                        .unwrap_or("Unknown");
                                    let batch_index = self.next_batch_index;
                                    self.next_batch_index += 1;
                                    self.batch_id_to_cityobject_id.push(Rc::clone(&building_id_rc));
                                    self.batch_id_to_cityobject_type.push(Rc::clone(&primary_type_str_rc));
                                    self.batch_id_to_surface_type.push(Rc::new(surface_type.to_string()));
                                    self.batch_id_to_attributes.push(merged_attributes.clone());
                                    let (area, wn, centroid) = self.add_surface_cached(surface, &feature.vertices, transform, batch_index, color, &mut feature_vertex_cache)?;
                                    self.push_computed_surface_props(area, wn, centroid);
                                }
                            }
                        }
                        Geometry::Solid { boundaries, .. } => {
                            for (shell_idx, shell) in boundaries.iter().enumerate() {
                                for (surf_idx, surface) in shell.iter().enumerate() {
                                    if skip_metadata {
                                        let _ = self.add_surface_cached(surface, &feature.vertices, transform, tree_sentinel, color, &mut feature_vertex_cache)?;
                                    } else {
                                        let surface_type = semantics
                                            .and_then(|s| s.surface_type(shell_idx, surf_idx))
                                            .unwrap_or("Unknown");
                                        let batch_index = self.next_batch_index;
                                        self.next_batch_index += 1;
                                        self.batch_id_to_cityobject_id.push(Rc::clone(&building_id_rc));
                                        self.batch_id_to_cityobject_type.push(Rc::clone(&primary_type_str_rc));
                                        self.batch_id_to_surface_type.push(Rc::new(surface_type.to_string()));
                                        self.batch_id_to_attributes.push(merged_attributes.clone());
                                        let (area, wn, centroid) = self.add_surface_cached(surface, &feature.vertices, transform, batch_index, color, &mut feature_vertex_cache)?;
                                        self.push_computed_surface_props(area, wn, centroid);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Push computed surface area and azimuth/elevation angles for one surface.
    /// Converts the ECEF weighted normal to local ENU using the surface centroid,
    /// then derives azimuth (0-360° from north, clockwise) and elevation
    /// (degrees from horizontal: 0 = vertical wall, 90 = flat roof).
    fn push_computed_surface_props(&mut self, area: f32, weighted_normal: [f32; 3], centroid_ecef: [f64; 3]) {
        self.batch_id_to_area.push(area);
        let len = (weighted_normal[0] * weighted_normal[0]
            + weighted_normal[1] * weighted_normal[1]
            + weighted_normal[2] * weighted_normal[2])
            .sqrt();
        if len > 0.0 {
            let nx = (weighted_normal[0] / len) as f64;
            let ny = (weighted_normal[1] / len) as f64;
            let nz = (weighted_normal[2] / len) as f64;

            // Geocentric lat/lon from ECEF centroid -> local ENU frame
            let cx = centroid_ecef[0];
            let cy = centroid_ecef[1];
            let cz = centroid_ecef[2];
            let lon = cy.atan2(cx);
            let lat = cz.atan2((cx * cx + cy * cy).sqrt());

            let (sin_lon, cos_lon) = lon.sin_cos();
            let (sin_lat, cos_lat) = lat.sin_cos();

            // Project ECEF unit normal into ENU
            let e = -sin_lon * nx + cos_lon * ny;
            let n = -sin_lat * cos_lon * nx - sin_lat * sin_lon * ny + cos_lat * nz;
            let u =  cos_lat * cos_lon * nx + cos_lat * sin_lon * ny + sin_lat * nz;

            let mut azimuth = e.atan2(n).to_degrees();
            if azimuth < 0.0 { azimuth += 360.0; }
            let elevation = u.clamp(-1.0, 1.0).asin().to_degrees();

            self.batch_id_to_azimuth.push(azimuth as f32);
            self.batch_id_to_elevation.push(elevation as f32);
        } else {
            self.batch_id_to_azimuth.push(0.0);
            self.batch_id_to_elevation.push(0.0);
        }
    }

    /// Add a single CityJSON boundary surface. Returns (surface_area, [nx, ny, nz])
    /// where the normal is the area-weighted average (not yet normalized).
    fn add_surface(
        &mut self,
        surface: &[Vec<usize>],
        vertices_qc: &[[i64; 3]],
        transform: &Transform,
        batch_index: u32,
        color: [f32; 4],
    ) -> Result<(f32, [f32; 3], [f64; 3])> {
        let mut local_cache: AHashMap<usize, u32> = AHashMap::new();
        self.add_surface_cached(surface, vertices_qc, transform, batch_index, color, &mut local_cache)
    }

    /// Add a single CityJSON boundary surface with an external vertex cache.
    /// Vertices already in the cache are reused across surfaces of the same feature,
    /// reducing GLB vertex count at surface boundaries.
    fn add_surface_cached(
        &mut self,
        surface: &[Vec<usize>],
        vertices_qc: &[[i64; 3]],
        transform: &Transform,
        batch_index: u32,
        color: [f32; 4],
        cache: &mut AHashMap<usize, u32>,
    ) -> Result<(f32, [f32; 3], [f64; 3])> {
        let zero_centroid = [0.0f64; 3];
        if surface.is_empty() {
            return Ok((0.0, [0.0, 0.0, 0.0], zero_centroid));
        }
        let exterior = &surface[0];
        if exterior.len() < 3 {
            return Ok((0.0, [0.0, 0.0, 0.0], zero_centroid));
        }

        let total_verts: usize = surface.iter().map(|r| r.len()).sum();
        let mut local_positions: Vec<[f32; 3]> = Vec::with_capacity(total_verts);
        let mut glb_indices: Vec<u32> = Vec::with_capacity(total_verts);
        let mut hole_indices: Vec<usize> = Vec::with_capacity(surface.len().saturating_sub(1));
        let mut vertex_count = 0usize;

        for (ring_idx, ring) in surface.iter().enumerate() {
            if ring.len() < 3 {
                continue;
            }
            if ring_idx > 0 {
                hole_indices.push(vertex_count);
            }

            for &vertex_id in ring {
                let (position, glb_index) = if let Some(&existing) = cache.get(&vertex_id) {
                    (self.positions[existing as usize], existing)
                } else {
                    let pos = self.compute_local_position(vertex_id, vertices_qc, transform)?;
                    self.positions.push(pos);
                    self.normals.push([0.0, 0.0, 0.0]);
                    self.colors.push(color);
                    self.batch_ids.push(batch_index);
                    let idx = (self.positions.len() - 1) as u32;
                    cache.insert(vertex_id, idx);
                    (pos, idx)
                };
                local_positions.push(position);
                glb_indices.push(glb_index);
                vertex_count += 1;
            }
        }

        if glb_indices.len() < 3 {
            return Ok((0.0, [0.0, 0.0, 0.0], zero_centroid));
        }

        // Compute surface centroid in full ECEF for azimuth/elevation calculation
        let n_pts = local_positions.len() as f64;
        let centroid_ecef = [
            local_positions.iter().map(|p| p[0] as f64).sum::<f64>() / n_pts + self.root_center_ecef.0,
            local_positions.iter().map(|p| p[1] as f64).sum::<f64>() / n_pts + self.root_center_ecef.1,
            local_positions.iter().map(|p| p[2] as f64).sum::<f64>() / n_pts + self.root_center_ecef.2,
        ];

        // Fast path: single ring with 3 or 4 vertices (triangles and quads, e.g. tree
        // crown sides and trunk sides). Triangulate in 3D so vertical faces are not
        // collapsed by the drop-axis projection.
        if surface.len() == 1 && hole_indices.is_empty() {
            let n = exterior.len();
            if n == 3 {
                let (area, wn) = self.emit_triangles(vec![glb_indices[0], glb_indices[1], glb_indices[2]]);
                return Ok((area, wn, centroid_ecef));
            }
            if n == 4 {
                let (area, wn) = self.emit_triangles(vec![
                    glb_indices[0],
                    glb_indices[1],
                    glb_indices[2],
                    glb_indices[0],
                    glb_indices[2],
                    glb_indices[3],
                ]);
                return Ok((area, wn, centroid_ecef));
            }
        }

        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        for pos in &local_positions {
            for axis in 0..3 {
                if pos[axis] < min[axis] {
                    min[axis] = pos[axis];
                }
                if pos[axis] > max[axis] {
                    max[axis] = pos[axis];
                }
            }
        }

        let mut ranges = [0.0f32; 3];
        for axis in 0..3 {
            ranges[axis] = max[axis] - min[axis];
        }
        let drop_axis = ranges
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(2);

        let mut flat_coords: Vec<f64> = Vec::with_capacity(local_positions.len() * 2);
        for pos in &local_positions {
            match drop_axis {
                0 => {
                    flat_coords.push(pos[1] as f64);
                    flat_coords.push(pos[2] as f64);
                }
                1 => {
                    flat_coords.push(pos[0] as f64);
                    flat_coords.push(pos[2] as f64);
                }
                _ => {
                    flat_coords.push(pos[0] as f64);
                    flat_coords.push(pos[1] as f64);
                }
            }
        }

        let triangulated = earcut(&flat_coords, &hole_indices, 2);
        if triangulated.len() < 3 {
            return Ok((0.0, [0.0, 0.0, 0.0], centroid_ecef));
        }

        let mut face_indices = Vec::with_capacity(triangulated.len());
        for idx in triangulated {
            face_indices.push(glb_indices[idx]);
        }

        let (area, wn) = self.emit_triangles(face_indices);
        Ok((area, wn, centroid_ecef))
    }

    /// Emit triangles into the index buffer and accumulate per-vertex normals.
    /// Returns (total_area, [weighted_nx, weighted_ny, weighted_nz]) for the surface.
    fn emit_triangles(&mut self, face_indices: Vec<u32>) -> (f32, [f32; 3]) {
        let mut total_area: f32 = 0.0;
        let mut weighted_normal = [0.0f32; 3];
        for tri in face_indices.chunks_exact(3) {
            let i0 = tri[0] as usize;
            let i1 = tri[1] as usize;
            let i2 = tri[2] as usize;

            let v0 = self.positions[i0];
            let v1 = self.positions[i1];
            let v2 = self.positions[i2];

            let u = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
            let v = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
            let cross = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            // |cross| = 2 * triangle_area
            let cross_len = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
            total_area += cross_len * 0.5;
            weighted_normal[0] += cross[0];
            weighted_normal[1] += cross[1];
            weighted_normal[2] += cross[2];

            for &i in tri {
                let n = &mut self.normals[i as usize];
                n[0] += cross[0];
                n[1] += cross[1];
                n[2] += cross[2];
            }

            self.indices.extend_from_slice(tri);
        }
        (total_area, weighted_normal)
    }

    fn compute_local_position(
        &self,
        idx: usize,
        vertices_qc: &[[i64; 3]],
        transform: &Transform,
    ) -> Result<[f32; 3], anyhow::Error> {
        let [x_qc, y_qc, z_qc] = vertices_qc[idx];
        // 1. Dequantize coordinates to input CRS
        let x_input = (x_qc as f64 * transform.scale[0]) + transform.translate[0];
        let y_input = (y_qc as f64 * transform.scale[1]) + transform.translate[1];
        let z_input = (z_qc as f64 * transform.scale[2]) + transform.translate[2] + self.vertical_bias;

        // 2. Transform coordinates from input CRS to ECEF (EPSG:4978)
        // This ensures coordinate system consistency with root transform (which is in ECEF)
        let (x_ecef, y_ecef, z_ecef) = self.transformer_to_ecef
            .convert((x_input, y_input, z_input))
            .context("Transform vertex coordinates to ECEF")?;

        // 3. Make coordinates relative to root center in ECEF
        // Root transform will translate these relative coordinates to the correct ECEF position
        let x_local = (x_ecef - self.root_center_ecef.0) as f32;
        let y_local = (y_ecef - self.root_center_ecef.1) as f32;
        let z_local = (z_ecef - self.root_center_ecef.2) as f32;
        

        // Return ECEF coordinates relative to root center
        // Y-up transformation in glTF node will convert from ECEF Z-up to glTF Y-up standard
        Ok([x_local, y_local, z_local])
    }

    fn write_glb<P: AsRef<Path>>(&mut self, output_path: P, tiles_version: TilesVersion, splat_cloud: Option<&SplatCloud>, splat_indices: &[usize]) -> Result<()> {
        self.normalize_normals();

        let has_mesh = !self.positions.is_empty();

        if !has_mesh && splat_indices.is_empty() {
            // Truly empty — no buildings, no splats
            if let Some(parent) = output_path.as_ref().parent() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("Failed to create parent directory for {:?}", output_path.as_ref()))?;
            }
            File::create(output_path.as_ref()).context("Create empty tile file")?;
            return Ok(());
        }

        let mut bin_buffer: Vec<u8> = Vec::with_capacity(
            self.positions.len() * (12 + 12 + 16 + 4) + self.indices.len() * 4 + 4096
        );

        // ── Mesh binary data (only when building geometry exists) ──
        let mut positions_offset = 0;
        let mut normals_offset = 0;
        let mut colors_offset = 0;
        let mut batch_ids_offset = 0;
        let mut indices_offset = 0;

        if has_mesh {
            for p in &self.positions {
                for component in p {
                    bin_buffer.extend_from_slice(&component.to_le_bytes());
                }
            }

            normals_offset = bin_buffer.len();
            for n in &self.normals {
                for component in n {
                    bin_buffer.extend_from_slice(&component.to_le_bytes());
                }
            }

            // Per-vertex RGBA colors (COLOR_0)
            colors_offset = bin_buffer.len();
            for c in &self.colors {
                for component in c {
                    bin_buffer.extend_from_slice(&component.to_le_bytes());
                }
            }

            // Use U16 for batch/feature IDs so validators accept it (glTF 2.0 mesh attributes cannot use UNSIGNED_INT).
            let feature_count = self.next_batch_index as usize;
            if feature_count > 65535 {
                bail!(
                    "Tile has {} features; glTF mesh attribute uses UNSIGNED_SHORT (max 65535)",
                    feature_count
                );
            }
            batch_ids_offset = bin_buffer.len();
            for &bid in &self.batch_ids {
                bin_buffer.extend_from_slice(&(bid as u16).to_le_bytes());
                bin_buffer.extend_from_slice(&[0u8, 0u8]); // pad to 4-byte stride
            }

            // batch_ids now already 4-byte aligned (each element is 4 bytes)
            indices_offset = bin_buffer.len();
            for index in &self.indices {
                bin_buffer.extend_from_slice(&index.to_le_bytes());
            }
        }

        // EXT_structural_metadata string property tables (1.1 only, mesh only)
        let mut string_data_offset = 0;
        let mut string_offsets_offset = 0;
        let mut string_offsets_len = 0;
        let mut id_string_data_len = 0usize;
        let mut type_string_data_offset = 0;
        let mut type_string_offsets_offset = 0;
        let mut type_string_offsets_len = 0;
        let mut type_string_data_len = 0usize;
        let mut stype_string_data_offset = 0;
        let mut stype_string_offsets_offset = 0;
        let mut stype_string_offsets_len = 0;
        let mut stype_string_data_len = 0usize;
        let mut surface_area_offset = 0usize;
        let mut surface_area_len = 0usize;
        let mut azimuth_offset = 0usize;
        let mut azimuth_len = 0usize;
        let mut elevation_offset = 0usize;
        let mut elevation_len = 0usize;
        let mut attr_buffer_views: Vec<AttrBufferView> = Vec::new();

        if has_mesh && tiles_version == TilesVersion::V1_1 {
            string_data_offset = bin_buffer.len();
            let mut string_offsets: Vec<u32> = Vec::with_capacity(self.batch_id_to_cityobject_id.len() + 1);
            let mut offset_acc: u32 = 0;
            for id in &self.batch_id_to_cityobject_id {
                string_offsets.push(offset_acc);
                let bytes = id.as_bytes();
                bin_buffer.extend_from_slice(bytes);
                bin_buffer.push(0); // null terminator
                offset_acc += (bytes.len() + 1) as u32;
            }
            string_offsets.push(offset_acc);
            id_string_data_len = bin_buffer.len() - string_data_offset;
            // Align to 4 bytes before u32 string offset array
            let pad = (4 - (bin_buffer.len() % 4)) % 4;
            bin_buffer.extend(std::iter::repeat(0u8).take(pad));
            string_offsets_offset = bin_buffer.len();
            string_offsets_len = string_offsets.len();
            for &o in &string_offsets {
                bin_buffer.extend_from_slice(&o.to_le_bytes());
            }

            type_string_data_offset = bin_buffer.len();
            let mut type_string_offsets: Vec<u32> =
                Vec::with_capacity(self.batch_id_to_cityobject_type.len() + 1);
            let mut type_offset_acc: u32 = 0;
            for type_name in &self.batch_id_to_cityobject_type {
                type_string_offsets.push(type_offset_acc);
                let bytes = type_name.as_bytes();
                bin_buffer.extend_from_slice(bytes);
                bin_buffer.push(0); // null terminator
                type_offset_acc += (bytes.len() + 1) as u32;
            }
            type_string_offsets.push(type_offset_acc);
            type_string_data_len = bin_buffer.len() - type_string_data_offset;
            // Align to 4 bytes before u32 type string offset array
            let pad = (4 - (bin_buffer.len() % 4)) % 4;
            bin_buffer.extend(std::iter::repeat(0u8).take(pad));
            type_string_offsets_offset = bin_buffer.len();
            type_string_offsets_len = type_string_offsets.len();
            for &o in &type_string_offsets {
                bin_buffer.extend_from_slice(&o.to_le_bytes());
            }

            // SurfaceType string property (e.g. "RoofSurface", "WallSurface", "GroundSurface")
            stype_string_data_offset = bin_buffer.len();
            let mut stype_string_offsets: Vec<u32> =
                Vec::with_capacity(self.batch_id_to_surface_type.len() + 1);
            let mut stype_offset_acc: u32 = 0;
            for stype_name in &self.batch_id_to_surface_type {
                stype_string_offsets.push(stype_offset_acc);
                let bytes = stype_name.as_bytes();
                bin_buffer.extend_from_slice(bytes);
                bin_buffer.push(0); // null terminator
                stype_offset_acc += (bytes.len() + 1) as u32;
            }
            stype_string_offsets.push(stype_offset_acc);
            stype_string_data_len = bin_buffer.len() - stype_string_data_offset;
            // Align to 4 bytes before u32 surface type string offset array
            let pad = (4 - (bin_buffer.len() % 4)) % 4;
            bin_buffer.extend(std::iter::repeat(0u8).take(pad));
            stype_string_offsets_offset = bin_buffer.len();
            stype_string_offsets_len = stype_string_offsets.len();
            for &o in &stype_string_offsets {
                bin_buffer.extend_from_slice(&o.to_le_bytes());
            }

            // ── Computed surface properties: SurfaceArea, SurfaceAzimuth, SurfaceElevation ──
            // Always 4-byte aligned because preceding u32 offsets are already aligned.
            surface_area_offset = bin_buffer.len();
            for &a in &self.batch_id_to_area {
                bin_buffer.extend_from_slice(&a.to_le_bytes());
            }
            surface_area_len = self.batch_id_to_area.len() * 4;

            azimuth_offset = bin_buffer.len();
            for &az in &self.batch_id_to_azimuth {
                bin_buffer.extend_from_slice(&az.to_le_bytes());
            }
            azimuth_len = self.batch_id_to_azimuth.len() * 4;

            elevation_offset = bin_buffer.len();
            for &el in &self.batch_id_to_elevation {
                bin_buffer.extend_from_slice(&el.to_le_bytes());
            }
            elevation_len = self.batch_id_to_elevation.len() * 4;

            // ── Forwarded CityObject attribute columns ──
            // Phase 1: infer schema (column name → type) from all rows.
            // Supported types: String, Float32, Int32, Boolean. Null/Array/Object → skip.
            // Int→Float promotion: if any row has a float for a previously-integer column.
            let n_rows = self.batch_id_to_attributes.len();
            let attr_schema: Vec<(String, AttrType)> = {
                let mut schema_map: std::collections::BTreeMap<String, AttrType> = std::collections::BTreeMap::new();
                for attrs_opt in &self.batch_id_to_attributes {
                    if let Some(attrs) = attrs_opt {
                        for (key, val) in attrs.iter() {
                            let val_type = match val {
                                serde_json::Value::String(_) => Some(AttrType::String),
                                serde_json::Value::Bool(_) => Some(AttrType::Boolean),
                                serde_json::Value::Number(n) => {
                                    if n.is_f64() && n.as_i64().is_none() {
                                        Some(AttrType::Float32)
                                    } else {
                                        Some(AttrType::Int32)
                                    }
                                }
                                _ => None, // Null, Array, Object → skip
                            };
                            if let Some(vt) = val_type {
                                schema_map
                                    .entry(key.clone())
                                    .and_modify(|existing| {
                                        // Promote Int32 → Float32 if mixed
                                        if *existing == AttrType::Int32 && vt == AttrType::Float32 {
                                            *existing = AttrType::Float32;
                                        }
                                    })
                                    .or_insert(vt);
                            }
                        }
                    }
                }
                schema_map.into_iter().collect()
            };

            // Phase 2: serialize each column into buffer views.
            // Track (bv_data_offset, bv_data_len, bv_offsets_offset, bv_offsets_len) per column.
            for (col_name, col_type) in &attr_schema {
                match col_type {
                    AttrType::String => {
                        let data_offset = bin_buffer.len();
                        let mut offsets: Vec<u32> = Vec::with_capacity(n_rows + 1);
                        let mut acc: u32 = 0;
                        for attrs_opt in &self.batch_id_to_attributes {
                            offsets.push(acc);
                            let s = attrs_opt.as_ref()
                                .and_then(|a| a.get(col_name))
                                .and_then(|v| v.as_str())
                                .unwrap_or(""); // noData sentinel for STRING
                            let bytes = s.as_bytes();
                            bin_buffer.extend_from_slice(bytes);
                            bin_buffer.push(0);
                            acc += (bytes.len() + 1) as u32;
                        }
                        offsets.push(acc);
                        let data_len = bin_buffer.len() - data_offset;
                        let pad = (4 - (bin_buffer.len() % 4)) % 4;
                        bin_buffer.extend(std::iter::repeat(0u8).take(pad));
                        let offsets_offset = bin_buffer.len();
                        for &o in &offsets {
                            bin_buffer.extend_from_slice(&o.to_le_bytes());
                        }
                        attr_buffer_views.push(AttrBufferView {
                            col_name: col_name.clone(),
                            col_type: *col_type,
                            data_bv_offset: data_offset,
                            data_bv_len: data_len,
                            offsets_bv_offset: Some(offsets_offset),
                            offsets_bv_len: Some(offsets.len() * 4),
                        });
                    }
                    AttrType::Float32 => {
                        let pad = (4 - (bin_buffer.len() % 4)) % 4;
                        bin_buffer.extend(std::iter::repeat(0u8).take(pad));
                        let data_offset = bin_buffer.len();
                        for attrs_opt in &self.batch_id_to_attributes {
                            let val = attrs_opt.as_ref()
                                .and_then(|a| a.get(col_name))
                                .and_then(|v| v.as_f64())
                                .map(|f| f as f32)
                                .unwrap_or(f32::NAN); // noData sentinel for FLOAT32
                            bin_buffer.extend_from_slice(&val.to_le_bytes());
                        }
                        let data_len = n_rows * 4;
                        attr_buffer_views.push(AttrBufferView {
                            col_name: col_name.clone(),
                            col_type: *col_type,
                            data_bv_offset: data_offset,
                            data_bv_len: data_len,
                            offsets_bv_offset: None,
                            offsets_bv_len: None,
                        });
                    }
                    AttrType::Int32 => {
                        let pad = (4 - (bin_buffer.len() % 4)) % 4;
                        bin_buffer.extend(std::iter::repeat(0u8).take(pad));
                        let data_offset = bin_buffer.len();
                        for attrs_opt in &self.batch_id_to_attributes {
                            let val = attrs_opt.as_ref()
                                .and_then(|a| a.get(col_name))
                                .and_then(|v| v.as_i64())
                                .map(|i| i as i32)
                                .unwrap_or(i32::MIN); // noData sentinel for INT32
                            bin_buffer.extend_from_slice(&val.to_le_bytes());
                        }
                        let data_len = n_rows * 4;
                        attr_buffer_views.push(AttrBufferView {
                            col_name: col_name.clone(),
                            col_type: *col_type,
                            data_bv_offset: data_offset,
                            data_bv_len: data_len,
                            offsets_bv_offset: None,
                            offsets_bv_len: None,
                        });
                    }
                    AttrType::Boolean => {
                        let data_offset = bin_buffer.len();
                        for attrs_opt in &self.batch_id_to_attributes {
                            let val: u8 = attrs_opt.as_ref()
                                .and_then(|a| a.get(col_name))
                                .and_then(|v| v.as_bool())
                                .map(|b| if b { 1u8 } else { 0u8 })
                                .unwrap_or(255); // noData sentinel for BOOLEAN
                            bin_buffer.push(val);
                        }
                        let data_len = n_rows;
                        // Align to 4 bytes after boolean array
                        let pad = (4 - (bin_buffer.len() % 4)) % 4;
                        bin_buffer.extend(std::iter::repeat(0u8).take(pad));
                        attr_buffer_views.push(AttrBufferView {
                            col_name: col_name.clone(),
                            col_type: *col_type,
                            data_bv_offset: data_offset,
                            data_bv_len: data_len,
                            offsets_bv_offset: None,
                            offsets_bv_len: None,
                        });
                    }
                }
            }
        }

        // ── Gaussian splat binary data ──
        // Write splat attribute arrays into the binary buffer (Phase 4a).
        // SH degree-0 normalization constant.
        const SH_C0: f32 = 0.28209479177387814;

        let splat_buf = if !splat_indices.is_empty() {
            if let Some(cloud) = splat_cloud {
                // Pad to 4-byte alignment before splat data.
                let pad = (4 - (bin_buffer.len() % 4)) % 4;
                bin_buffer.extend(std::iter::repeat(0u8).take(pad));

                let n = splat_indices.len();
                let mut pos_min = [f32::MAX; 3];
                let mut pos_max = [f32::MIN; 3];

                // POSITION (VEC3 F32)
                let splat_position_offset = bin_buffer.len();
                for &idx in splat_indices.iter() {
                    let pt = &cloud.splats[idx];
                    let (x_ecef, y_ecef, z_ecef) = self.transformer_to_ecef
                        .convert((pt.position[0], pt.position[1], pt.position[2]))
                        .unwrap_or((0.0, 0.0, 0.0));
                    let pos = [
                        (x_ecef - self.root_center_ecef.0) as f32,
                        (y_ecef - self.root_center_ecef.1) as f32,
                        (z_ecef - self.root_center_ecef.2) as f32,
                    ];
                    for (i, &v) in pos.iter().enumerate() {
                        if v < pos_min[i] { pos_min[i] = v; }
                        if v > pos_max[i] { pos_max[i] = v; }
                    }
                    for &c in &pos { bin_buffer.extend_from_slice(&c.to_le_bytes()); }
                }

                // COLOR_0 (VEC4 F32) — fallback for renderers without GS support
                let splat_color_offset = bin_buffer.len();
                for &idx in splat_indices.iter() {
                    let rgb = cloud.splats[idx].rgb;
                    for &c in &[rgb[0], rgb[1], rgb[2], 1.0f32] {
                        bin_buffer.extend_from_slice(&c.to_le_bytes());
                    }
                }

                // ROTATION (VEC4 F32) — quaternion [x,y,z,w]
                let splat_rotation_offset = bin_buffer.len();
                for &idx in splat_indices.iter() {
                    let rot = cloud.splats[idx].rotation;
                    for &c in &rot { bin_buffer.extend_from_slice(&c.to_le_bytes()); }
                }

                // SCALE (VEC3 F32)
                let splat_scale_offset = bin_buffer.len();
                for &idx in splat_indices.iter() {
                    let scale = cloud.splats[idx].scale;
                    for &c in &scale { bin_buffer.extend_from_slice(&c.to_le_bytes()); }
                }

                // OPACITY (SCALAR F32)
                let splat_opacity_offset = bin_buffer.len();
                for &idx in splat_indices.iter() {
                    bin_buffer.extend_from_slice(&cloud.splats[idx].opacity.to_le_bytes());
                }

                // SH_DEGREE_0_COEF_0 (VEC3 F32) — from RGB
                let splat_sh0_offset = bin_buffer.len();
                for &idx in splat_indices.iter() {
                    let rgb = cloud.splats[idx].rgb;
                    let sh = [
                        (rgb[0] - 0.5) / SH_C0,
                        (rgb[1] - 0.5) / SH_C0,
                        (rgb[2] - 0.5) / SH_C0,
                    ];
                    for &c in &sh { bin_buffer.extend_from_slice(&c.to_le_bytes()); }
                }

                Some(SplatBufferData {
                    count: n,
                    position_offset: splat_position_offset,
                    color_offset: splat_color_offset,
                    rotation_offset: splat_rotation_offset,
                    scale_offset: splat_scale_offset,
                    opacity_offset: splat_opacity_offset,
                    sh0_offset: splat_sh0_offset,
                    pos_min,
                    pos_max,
                })
            } else {
                None
            }
        } else {
            None
        };

        let mut accessors: Vec<json::Accessor> = Vec::new();
        let mut buffer_views: Vec<json::buffer::View> = Vec::new();

        if has_mesh {
            // Mesh accessors (indices 0-4)
            accessors.push(json::Accessor {
                buffer_view: Some(json::Index::new(0)),
                byte_offset: Some(json::validation::USize64(0)),
                count: json::validation::USize64(self.positions.len() as u64),
                component_type: json::validation::Checked::Valid(json::accessor::GenericComponentType(
                    json::accessor::ComponentType::F32,
                )),
                normalized: false,
                min: Some(json::Value::Array(
                    (0..3)
                        .map(|axis| {
                            let min = self.positions.iter().map(|v| v[axis]).fold(f32::INFINITY, f32::min);
                            json::Value::from(min)
                        })
                        .collect(),
                )),
                max: Some(json::Value::Array(
                    (0..3)
                        .map(|axis| {
                            let max = self.positions.iter().map(|v| v[axis]).fold(f32::NEG_INFINITY, f32::max);
                            json::Value::from(max)
                        })
                        .collect(),
                )),
                type_: json::validation::Checked::Valid(json::accessor::Type::Vec3),
                extensions: Default::default(),
                extras: Default::default(),
                name: None,
                sparse: None,
            });
            accessors.push(json::Accessor {
                buffer_view: Some(json::Index::new(1)),
                byte_offset: Some(json::validation::USize64(0)),
                count: json::validation::USize64(self.normals.len() as u64),
                component_type: json::validation::Checked::Valid(json::accessor::GenericComponentType(
                    json::accessor::ComponentType::F32,
                )),
                normalized: false,
                type_: json::validation::Checked::Valid(json::accessor::Type::Vec3),
                extensions: Default::default(),
                extras: Default::default(),
                min: None,
                max: None,
                name: None,
                sparse: None,
            });
            accessors.push(json::Accessor {
                buffer_view: Some(json::Index::new(2)),
                byte_offset: Some(json::validation::USize64(0)),
                count: json::validation::USize64(self.colors.len() as u64),
                component_type: json::validation::Checked::Valid(json::accessor::GenericComponentType(
                    json::accessor::ComponentType::F32,
                )),
                normalized: false,
                type_: json::validation::Checked::Valid(json::accessor::Type::Vec4),
                extensions: Default::default(),
                extras: Default::default(),
                min: None,
                max: None,
                name: None,
                sparse: None,
            });
            accessors.push(json::Accessor {
                buffer_view: Some(json::Index::new(3)),
                byte_offset: Some(json::validation::USize64(0)),
                count: json::validation::USize64(self.batch_ids.len() as u64),
                component_type: json::validation::Checked::Valid(json::accessor::GenericComponentType(
                    json::accessor::ComponentType::U16,
                )),
                normalized: false,
                min: Some(json::Value::from(vec![*self.batch_ids.iter().min().unwrap_or(&0)])),
                max: Some(json::Value::from(vec![*self.batch_ids.iter().max().unwrap_or(&0)])),
                type_: json::validation::Checked::Valid(json::accessor::Type::Scalar),
                extensions: Default::default(),
                extras: Default::default(),
                name: None,
                sparse: None,
            });
            accessors.push(json::Accessor {
                buffer_view: Some(json::Index::new(4)),
                byte_offset: Some(json::validation::USize64(0)),
                count: json::validation::USize64(self.indices.len() as u64),
                component_type: json::validation::Checked::Valid(json::accessor::GenericComponentType(
                    json::accessor::ComponentType::U32,
                )),
                normalized: false,
                min: Some(json::Value::from(vec![0])),
                max: Some(json::Value::from(vec![self.indices.iter().copied().max().unwrap_or(0)])),
                type_: json::validation::Checked::Valid(json::accessor::Type::Scalar),
                extensions: Default::default(),
                extras: Default::default(),
                name: None,
                sparse: None,
            });

            // BV 0-4: mesh geometry buffer views
            buffer_views.extend([
                // BV 0: POSITION
                json::buffer::View {
                    buffer: json::Index::new(0),
                    byte_length: json::validation::USize64((self.positions.len() * 12) as u64),
                    byte_offset: Some(json::validation::USize64(positions_offset as u64)),
                    byte_stride: None,
                    target: Some(json::validation::Checked::Valid(json::buffer::Target::ArrayBuffer)),
                    extensions: Default::default(),
                    extras: Default::default(),
                    name: None,
                },
                // BV 1: NORMAL
                json::buffer::View {
                    buffer: json::Index::new(0),
                    byte_length: json::validation::USize64((self.normals.len() * 12) as u64),
                    byte_offset: Some(json::validation::USize64(normals_offset as u64)),
                    byte_stride: None,
                    target: Some(json::validation::Checked::Valid(json::buffer::Target::ArrayBuffer)),
                    extensions: Default::default(),
                    extras: Default::default(),
                    name: None,
                },
                // BV 2: COLOR_0 (Vec4 F32 = 16 bytes/vertex)
                json::buffer::View {
                    buffer: json::Index::new(0),
                    byte_length: json::validation::USize64((self.colors.len() * 16) as u64),
                    byte_offset: Some(json::validation::USize64(colors_offset as u64)),
                    byte_stride: None,
                    target: Some(json::validation::Checked::Valid(json::buffer::Target::ArrayBuffer)),
                    extensions: Default::default(),
                    extras: Default::default(),
                    name: None,
                },
                // BV 3: _BATCHID (1.0) or _FEATURE_ID_0 (1.1)
                json::buffer::View {
                    buffer: json::Index::new(0),
                    byte_length: json::validation::USize64((self.batch_ids.len() * 4) as u64),
                    byte_offset: Some(json::validation::USize64(batch_ids_offset as u64)),
                    byte_stride: Some(json::buffer::Stride(4)),
                    target: Some(json::validation::Checked::Valid(json::buffer::Target::ArrayBuffer)),
                    extensions: Default::default(),
                    extras: Default::default(),
                    name: None,
                },
                // BV 4: indices
                json::buffer::View {
                    buffer: json::Index::new(0),
                    byte_length: json::validation::USize64((self.indices.len() * 4) as u64),
                    byte_offset: Some(json::validation::USize64(indices_offset as u64)),
                    byte_stride: None,
                    target: Some(json::validation::Checked::Valid(json::buffer::Target::ElementArrayBuffer)),
                    extensions: Default::default(),
                    extras: Default::default(),
                    name: None,
                },
            ]);
        }

        // BV 5-8: EXT_structural_metadata string property tables (1.1 only, mesh only)
        if has_mesh && tiles_version == TilesVersion::V1_1 {
            buffer_views.extend([
                // BV 5: BuildingId string data
                json::buffer::View {
                    buffer: json::Index::new(0),
                    byte_length: json::validation::USize64(id_string_data_len as u64),
                    byte_offset: Some(json::validation::USize64(string_data_offset as u64)),
                    byte_stride: None,
                    target: None,
                    extensions: Default::default(),
                    extras: Default::default(),
                    name: None,
                },
                // BV 6: BuildingId string offsets
                json::buffer::View {
                    buffer: json::Index::new(0),
                    byte_length: json::validation::USize64((string_offsets_len * 4) as u64),
                    byte_offset: Some(json::validation::USize64(string_offsets_offset as u64)),
                    byte_stride: None,
                    target: None,
                    extensions: Default::default(),
                    extras: Default::default(),
                    name: None,
                },
                // BV 7: CityObjectType string data
                json::buffer::View {
                    buffer: json::Index::new(0),
                    byte_length: json::validation::USize64(type_string_data_len as u64),
                    byte_offset: Some(json::validation::USize64(type_string_data_offset as u64)),
                    byte_stride: None,
                    target: None,
                    extensions: Default::default(),
                    extras: Default::default(),
                    name: None,
                },
                // BV 8: CityObjectType string offsets
                json::buffer::View {
                    buffer: json::Index::new(0),
                    byte_length: json::validation::USize64((type_string_offsets_len * 4) as u64),
                    byte_offset: Some(json::validation::USize64(type_string_offsets_offset as u64)),
                    byte_stride: None,
                    target: None,
                    extensions: Default::default(),
                    extras: Default::default(),
                    name: None,
                },
                // BV 9: SurfaceType string data
                json::buffer::View {
                    buffer: json::Index::new(0),
                    byte_length: json::validation::USize64(stype_string_data_len as u64),
                    byte_offset: Some(json::validation::USize64(stype_string_data_offset as u64)),
                    byte_stride: None,
                    target: None,
                    extensions: Default::default(),
                    extras: Default::default(),
                    name: None,
                },
                // BV 10: SurfaceType string offsets
                json::buffer::View {
                    buffer: json::Index::new(0),
                    byte_length: json::validation::USize64((stype_string_offsets_len * 4) as u64),
                    byte_offset: Some(json::validation::USize64(stype_string_offsets_offset as u64)),
                    byte_stride: None,
                    target: None,
                    extensions: Default::default(),
                    extras: Default::default(),
                    name: None,
                },
                // BV 11: SurfaceArea (FLOAT32)
                json::buffer::View {
                    buffer: json::Index::new(0),
                    byte_length: json::validation::USize64(surface_area_len as u64),
                    byte_offset: Some(json::validation::USize64(surface_area_offset as u64)),
                    byte_stride: None,
                    target: None,
                    extensions: Default::default(),
                    extras: Default::default(),
                    name: None,
                },
                // BV 12: SurfaceAzimuth (FLOAT32)
                json::buffer::View {
                    buffer: json::Index::new(0),
                    byte_length: json::validation::USize64(azimuth_len as u64),
                    byte_offset: Some(json::validation::USize64(azimuth_offset as u64)),
                    byte_stride: None,
                    target: None,
                    extensions: Default::default(),
                    extras: Default::default(),
                    name: None,
                },
                // BV 13: SurfaceElevation (FLOAT32)
                json::buffer::View {
                    buffer: json::Index::new(0),
                    byte_length: json::validation::USize64(elevation_len as u64),
                    byte_offset: Some(json::validation::USize64(elevation_offset as u64)),
                    byte_stride: None,
                    target: None,
                    extensions: Default::default(),
                    extras: Default::default(),
                    name: None,
                },
            ]);

            // BV 15+: dynamic attribute columns
            for abv in &attr_buffer_views {
                // Data buffer view
                buffer_views.push(json::buffer::View {
                    buffer: json::Index::new(0),
                    byte_length: json::validation::USize64(abv.data_bv_len as u64),
                    byte_offset: Some(json::validation::USize64(abv.data_bv_offset as u64)),
                    byte_stride: None,
                    target: None,
                    extensions: Default::default(),
                    extras: Default::default(),
                    name: None,
                });
                // String offset buffer view (only for STRING columns)
                if let (Some(off), Some(len)) = (abv.offsets_bv_offset, abv.offsets_bv_len) {
                    buffer_views.push(json::buffer::View {
                        buffer: json::Index::new(0),
                        byte_length: json::validation::USize64(len as u64),
                        byte_offset: Some(json::validation::USize64(off as u64)),
                        byte_stride: None,
                        target: None,
                        extensions: Default::default(),
                        extras: Default::default(),
                        name: None,
                    });
                }
            }
        }

        // ── Splat buffer views (appended after all mesh/metadata BVs) ──
        let splat_bv_base = buffer_views.len() as u32;
        if let Some(ref sb) = splat_buf {
            let n = sb.count;
            // BV S+0: splat POSITION (VEC3 F32)
            buffer_views.push(json::buffer::View {
                buffer: json::Index::new(0),
                byte_length: json::validation::USize64((n * 12) as u64),
                byte_offset: Some(json::validation::USize64(sb.position_offset as u64)),
                byte_stride: None,
                target: None,
                extensions: Default::default(),
                extras: Default::default(),
                name: None,
            });
            // BV S+1: splat COLOR_0 (VEC4 F32)
            buffer_views.push(json::buffer::View {
                buffer: json::Index::new(0),
                byte_length: json::validation::USize64((n * 16) as u64),
                byte_offset: Some(json::validation::USize64(sb.color_offset as u64)),
                byte_stride: None,
                target: None,
                extensions: Default::default(),
                extras: Default::default(),
                name: None,
            });
            // BV S+2: splat ROTATION (VEC4 F32)
            buffer_views.push(json::buffer::View {
                buffer: json::Index::new(0),
                byte_length: json::validation::USize64((n * 16) as u64),
                byte_offset: Some(json::validation::USize64(sb.rotation_offset as u64)),
                byte_stride: None,
                target: None,
                extensions: Default::default(),
                extras: Default::default(),
                name: None,
            });
            // BV S+3: splat SCALE (VEC3 F32)
            buffer_views.push(json::buffer::View {
                buffer: json::Index::new(0),
                byte_length: json::validation::USize64((n * 12) as u64),
                byte_offset: Some(json::validation::USize64(sb.scale_offset as u64)),
                byte_stride: None,
                target: None,
                extensions: Default::default(),
                extras: Default::default(),
                name: None,
            });
            // BV S+4: splat OPACITY (SCALAR F32)
            buffer_views.push(json::buffer::View {
                buffer: json::Index::new(0),
                byte_length: json::validation::USize64((n * 4) as u64),
                byte_offset: Some(json::validation::USize64(sb.opacity_offset as u64)),
                byte_stride: None,
                target: None,
                extensions: Default::default(),
                extras: Default::default(),
                name: None,
            });
            // BV S+5: splat SH_DEGREE_0_COEF_0 (VEC3 F32)
            buffer_views.push(json::buffer::View {
                buffer: json::Index::new(0),
                byte_length: json::validation::USize64((n * 12) as u64),
                byte_offset: Some(json::validation::USize64(sb.sh0_offset as u64)),
                byte_stride: None,
                target: None,
                extensions: Default::default(),
                extras: Default::default(),
                name: None,
            });
        }

        let mut primitives: Vec<json::mesh::Primitive> = Vec::new();
        let mut materials: Vec<json::Material> = Vec::new();

        if has_mesh {
            let mut attributes = std::collections::BTreeMap::new();
            attributes.insert(
                json::validation::Checked::Valid(json::mesh::Semantic::Positions),
                json::Index::new(0),
            );
            attributes.insert(
                json::validation::Checked::Valid(json::mesh::Semantic::Normals),
                json::Index::new(1),
            );
            // Per-vertex RGBA color
            attributes.insert(
                json::validation::Checked::Valid(json::mesh::Semantic::Colors(0)),
                json::Index::new(2),
            );
            // 1.1: _FEATURE_ID_0 for EXT_mesh_features; 1.0: _BATCHID for batch table
            let batch_attr_name = match tiles_version {
                TilesVersion::V1_1 => "FEATURE_ID_0",
                TilesVersion::V1_0 => "BATCHID",
            };
            attributes.insert(
                json::validation::Checked::Valid(json::mesh::Semantic::Extras(batch_attr_name.into())),
                json::Index::new(3),
            );

            // White base material — vertex colors provide the actual per-type coloring
            let material = create_material("#FFFFFF", self.metallic_factor, self.roughness_factor)?;

            let feature_count = self.next_batch_index;

            // EXT_mesh_features primitive extension (1.1 only)
            let primitive_extensions = match tiles_version {
                TilesVersion::V1_1 => {
                    let mut ext_others = serde_json::Map::new();
                    ext_others.insert(
                        "EXT_mesh_features".to_string(),
                        serde_json::json!({
                            "featureIds": [{
                                "attribute": 0,
                                "featureCount": feature_count,
                                "nullFeatureId": 65535,
                                "propertyTable": 0
                            }]
                        }),
                    );
                    Some(json::extensions::mesh::Primitive {
                        others: ext_others,
                        ..Default::default()
                    })
                }
                TilesVersion::V1_0 => None,
            };

            let primitive = json::mesh::Primitive {
                attributes,
                indices: Some(json::Index::new(4)),
                material: Some(json::Index::new(0)),
                mode: json::validation::Checked::Valid(json::mesh::Mode::Triangles),
                targets: None,
                extensions: primitive_extensions,
                extras: Default::default(),
            };

            primitives.push(primitive);
            materials.push(material);
        }

        // ── Gaussian splat primitive (Phase 4a) ──
        if let Some(ref sb) = splat_buf {
            let acc_base = accessors.len() as u32;

            // Splat accessors: POSITION, COLOR_0, ROTATION, SCALE, OPACITY, SH0
            let make_accessor = |bv_idx: u32, count: usize, comp: json::accessor::ComponentType, ty: json::accessor::Type, min: Option<json::Value>, max: Option<json::Value>| {
                json::Accessor {
                    buffer_view: Some(json::Index::new(bv_idx)),
                    byte_offset: Some(json::validation::USize64(0)),
                    count: json::validation::USize64(count as u64),
                    component_type: json::validation::Checked::Valid(json::accessor::GenericComponentType(comp)),
                    normalized: false,
                    type_: json::validation::Checked::Valid(ty),
                    min,
                    max,
                    extensions: Default::default(),
                    extras: Default::default(),
                    name: None,
                    sparse: None,
                }
            };

            // Accessor S+0: splat POSITION with min/max
            accessors.push(make_accessor(
                splat_bv_base, sb.count,
                json::accessor::ComponentType::F32, json::accessor::Type::Vec3,
                Some(json::Value::Array(sb.pos_min.iter().map(|&v| json::Value::from(v)).collect())),
                Some(json::Value::Array(sb.pos_max.iter().map(|&v| json::Value::from(v)).collect())),
            ));
            // Accessor S+1: splat COLOR_0
            accessors.push(make_accessor(
                splat_bv_base + 1, sb.count,
                json::accessor::ComponentType::F32, json::accessor::Type::Vec4, None, None,
            ));
            // Accessor S+2: splat ROTATION
            accessors.push(make_accessor(
                splat_bv_base + 2, sb.count,
                json::accessor::ComponentType::F32, json::accessor::Type::Vec4, None, None,
            ));
            // Accessor S+3: splat SCALE
            accessors.push(make_accessor(
                splat_bv_base + 3, sb.count,
                json::accessor::ComponentType::F32, json::accessor::Type::Vec3, None, None,
            ));
            // Accessor S+4: splat OPACITY
            accessors.push(make_accessor(
                splat_bv_base + 4, sb.count,
                json::accessor::ComponentType::F32, json::accessor::Type::Scalar, None, None,
            ));
            // Accessor S+5: splat SH_DEGREE_0_COEF_0
            accessors.push(make_accessor(
                splat_bv_base + 5, sb.count,
                json::accessor::ComponentType::F32, json::accessor::Type::Vec3, None, None,
            ));

            // Splat primitive attributes (standard glTF semantics)
            let mut splat_attributes = std::collections::BTreeMap::new();
            splat_attributes.insert(
                json::validation::Checked::Valid(json::mesh::Semantic::Positions),
                json::Index::new(acc_base),
            );
            splat_attributes.insert(
                json::validation::Checked::Valid(json::mesh::Semantic::Colors(0)),
                json::Index::new(acc_base + 1),
            );

            // KHR_gaussian_splatting extension on the primitive
            let mut gs_ext = serde_json::Map::new();
            gs_ext.insert("KHR_gaussian_splatting".to_string(), serde_json::json!({
                "kernel": "ellipse",
                "colorSpace": "sh",
                "attributes": {
                    "ROTATION": acc_base + 2,
                    "SCALE": acc_base + 3,
                    "OPACITY": acc_base + 4,
                    "SH_DEGREE_0_COEF_0": acc_base + 5,
                }
            }));

            let splat_primitive = json::mesh::Primitive {
                attributes: splat_attributes,
                indices: None,
                material: Some(json::Index::new(materials.len() as u32)),
                mode: json::validation::Checked::Valid(json::mesh::Mode::Points),
                targets: None,
                extensions: Some(json::extensions::mesh::Primitive {
                    others: gs_ext,
                    ..Default::default()
                }),
                extras: Default::default(),
            };
            primitives.push(splat_primitive);

            // Unlit material for splat fallback (colored point cloud rendering)
            let mut unlit_ext = serde_json::Map::new();
            unlit_ext.insert("KHR_materials_unlit".into(), serde_json::json!({}));
            materials.push(json::Material {
                pbr_metallic_roughness: json::material::PbrMetallicRoughness {
                    base_color_factor: json::material::PbrBaseColorFactor([1.0, 1.0, 1.0, 1.0]),
                    metallic_factor: json::material::StrengthFactor(0.0),
                    roughness_factor: json::material::StrengthFactor(1.0),
                    base_color_texture: None,
                    metallic_roughness_texture: None,
                    extensions: Default::default(),
                    extras: Default::default(),
                },
                extensions: Some(json::extensions::material::Material {
                    others: unlit_ext,
                    ..Default::default()
                }),
                alpha_mode: json::validation::Checked::Valid(json::material::AlphaMode::Opaque),
                double_sided: true,
                ..Default::default()
            });
        }

        let mesh = json::Mesh {
            primitives,
            weights: None,
            extensions: Default::default(),
            extras: Default::default(),
            name: None,
        };

        // Apply Y-up transformation matrix to convert from ECEF (Z-up) to glTF standard (Y-up)
        // GLB content coordinates are in ECEF (Z-up), and this matrix converts them to glTF Y-up format
        // This matches pg2b3dm's approach - the Y-up matrix is needed in GLB node
        // Matrix format: [1,0,0,0, 0,0,-1,0, 0,1,0,0, 0,0,0,1] (column-major in glTF JSON)
        // Transformation: X'=X, Y'=-Z (ECEF Z becomes glTF -Y), Z'=Y (ECEF Y becomes glTF Z)
        let y_up_matrix = [
            1.0, 0.0, 0.0, 0.0,   // Column 0: [1, 0, 0, 0] - X axis
            0.0, 0.0, -1.0, 0.0,  // Column 1: [0, 0, -1, 0] - Y axis becomes -Z
            0.0, 1.0, 0.0, 0.0,   // Column 2: [0, 1, 0, 0] - Z axis becomes Y
            0.0, 0.0, 0.0, 1.0,   // Column 3: [0, 0, 0, 1] - Translation/scale
        ];

        let node = json::Node {
            mesh: Some(json::Index::new(0)),
            camera: None,
            children: None,
            skin: None,
            matrix: Some(y_up_matrix),
            rotation: None,
            scale: None,
            translation: None,
            weights: None,
            extensions: Default::default(),
            extras: Default::default(),
            name: None,
        };

        let scene = json::Scene {
            nodes: vec![json::Index::new(0)],
            extensions: Default::default(),
            extras: Default::default(),
            name: None,
        };

        // EXT_structural_metadata + extensions_used (1.1 only, mesh only)
        let (root_extensions, extensions_used) = match (tiles_version, has_mesh) {
            (TilesVersion::V1_1, true) => {
                let n_features = self.batch_id_to_cityobject_id.len();

                // Build schema properties and property table entries dynamically.
                let mut schema_props = serde_json::Map::new();
                let mut table_props = serde_json::Map::new();

                // Fixed properties: BuildingId, CityObjectType, SurfaceType
                schema_props.insert("BuildingId".into(), serde_json::json!({
                    "type": "STRING", "stringOffsetType": "UINT32"
                }));
                table_props.insert("BuildingId".into(), serde_json::json!({
                    "values": 5, "stringOffsets": 6
                }));

                schema_props.insert("CityObjectType".into(), serde_json::json!({
                    "type": "STRING", "stringOffsetType": "UINT32"
                }));
                table_props.insert("CityObjectType".into(), serde_json::json!({
                    "values": 7, "stringOffsets": 8
                }));

                schema_props.insert("SurfaceType".into(), serde_json::json!({
                    "type": "STRING", "stringOffsetType": "UINT32"
                }));
                table_props.insert("SurfaceType".into(), serde_json::json!({
                    "values": 9, "stringOffsets": 10
                }));

                // Computed per-surface properties (always emitted, BV 11-14)
                schema_props.insert("SurfaceArea".into(), serde_json::json!({
                    "type": "SCALAR", "componentType": "FLOAT32"
                }));
                table_props.insert("SurfaceArea".into(), serde_json::json!({
                    "values": 11
                }));

                schema_props.insert("SurfaceAzimuth".into(), serde_json::json!({
                    "type": "SCALAR", "componentType": "FLOAT32"
                }));
                table_props.insert("SurfaceAzimuth".into(), serde_json::json!({
                    "values": 12
                }));

                schema_props.insert("SurfaceElevation".into(), serde_json::json!({
                    "type": "SCALAR", "componentType": "FLOAT32"
                }));
                table_props.insert("SurfaceElevation".into(), serde_json::json!({
                    "values": 13
                }));

                // Dynamic attribute columns: BV indices start at 14
                let mut next_bv: u32 = 14;
                for abv in &attr_buffer_views {
                    match abv.col_type {
                        AttrType::String => {
                            schema_props.insert(abv.col_name.clone(), serde_json::json!({
                                "type": "STRING",
                                "stringOffsetType": "UINT32",
                                "noData": ""
                            }));
                            table_props.insert(abv.col_name.clone(), serde_json::json!({
                                "values": next_bv,
                                "stringOffsets": next_bv + 1
                            }));
                            next_bv += 2;
                        }
                        AttrType::Float32 => {
                            schema_props.insert(abv.col_name.clone(), serde_json::json!({
                                "type": "SCALAR",
                                "componentType": "FLOAT32",
                                "noData": f32::NAN
                            }));
                            table_props.insert(abv.col_name.clone(), serde_json::json!({
                                "values": next_bv
                            }));
                            next_bv += 1;
                        }
                        AttrType::Int32 => {
                            schema_props.insert(abv.col_name.clone(), serde_json::json!({
                                "type": "SCALAR",
                                "componentType": "INT32",
                                "noData": i32::MIN
                            }));
                            table_props.insert(abv.col_name.clone(), serde_json::json!({
                                "values": next_bv
                            }));
                            next_bv += 1;
                        }
                        AttrType::Boolean => {
                            schema_props.insert(abv.col_name.clone(), serde_json::json!({
                                "type": "BOOLEAN",
                                "noData": 255
                            }));
                            table_props.insert(abv.col_name.clone(), serde_json::json!({
                                "values": next_bv
                            }));
                            next_bv += 1;
                        }
                    }
                }

                let structural_metadata_ext = serde_json::json!({
                    "schema": {
                        "classes": {
                            "Feature": {
                                "properties": schema_props
                            }
                        }
                    },
                    "propertyTables": [{
                        "class": "Feature",
                        "count": n_features,
                        "properties": table_props
                    }]
                });
                let mut root_ext_others = serde_json::Map::new();
                root_ext_others.insert("EXT_structural_metadata".to_string(), structural_metadata_ext);
                (
                    Some(json::extensions::root::Root {
                        others: root_ext_others,
                        ..Default::default()
                    }),
                    vec!["EXT_mesh_features".into(), "EXT_structural_metadata".into()],
                )
            }
            _ => (None, vec![]),
        };

        // Add Gaussian splat extensions if splat primitive is present.
        let mut extensions_used = extensions_used;
        if splat_buf.is_some() {
            extensions_used.push("KHR_gaussian_splatting".into());
            extensions_used.push("KHR_materials_unlit".into());
        }

        let root = json::Root {
            accessors,
            extensions_used,
            extensions: root_extensions,
            buffers: vec![json::Buffer {
                byte_length: json::validation::USize64(bin_buffer.len() as u64),
                uri: None,
                name: Some("buffer0".into()),
                extensions: Default::default(),
                extras: Default::default(),
            }],
            buffer_views,
            materials,
            meshes: vec![mesh],
            nodes: vec![node],
            scenes: vec![scene],
            scene: Some(json::Index::new(0)),
            asset: json::Asset {
                version: GLTF_VERSION.into(),
                generator: Some("tyler".into()),
                copyright: None,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut json_bytes = json::serialize::to_string(&root)?.into_bytes();
        let json_padding = (4 - (json_bytes.len() % 4)) % 4;
        json_bytes.extend(std::iter::repeat(b' ').take(json_padding));

        let bin_padding = (4 - (bin_buffer.len() % 4)) % 4;
        bin_buffer.extend(std::iter::repeat(0).take(bin_padding));

        let total_length = 12 + 8 + json_bytes.len() + 8 + bin_buffer.len();
        let mut glb_bytes = Vec::with_capacity(total_length);
        glb_bytes.extend_from_slice(b"glTF");
        glb_bytes.extend_from_slice(&2u32.to_le_bytes());
        glb_bytes.extend_from_slice(&(total_length as u32).to_le_bytes());

        glb_bytes.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
        glb_bytes.extend_from_slice(b"JSON");
        glb_bytes.extend_from_slice(&json_bytes);

        glb_bytes.extend_from_slice(&(bin_buffer.len() as u32).to_le_bytes());
        glb_bytes.extend_from_slice(b"BIN\0");
        glb_bytes.extend_from_slice(&bin_buffer);

        // For 1.0 with mesh: wrap GLB in B3DM container with feature table + batch table
        // Splat-only tiles are always plain GLB (no B3DM wrapping)
        let final_bytes = match (tiles_version, has_mesh) {
            (TilesVersion::V1_0, true) => wrap_b3dm(
                &glb_bytes,
                &self.batch_id_to_cityobject_id,
                &self.batch_id_to_cityobject_type,
                &self.batch_id_to_surface_type,
            ),
            _ => glb_bytes,
        };

        // Create parent directories if they don't exist
        if let Some(parent) = output_path.as_ref().parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create parent directory for {:?}", output_path.as_ref()))?;
        }

        let mut file = BufWriter::with_capacity(1 << 20, File::create(output_path)?);
        file.write_all(&final_bytes)?;

        Ok(())
    }

    fn normalize_normals(&mut self) {
        for normal in self.normals.iter_mut() {
            let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            if length > f32::EPSILON {
                normal[0] /= length;
                normal[1] /= length;
                normal[2] /= length;
            } else {
                // Zero-length normal (degenerate triangles). Use default up vector [0, 1, 0]
                // which is appropriate for glTF Y-up coordinate system
                normal[0] = 0.0;
                normal[1] = 1.0;
                normal[2] = 0.0;
            }
        }
    }
}

/// Wrap a GLB binary in a B3DM container with feature table and batch table.
/// B3DM format: 28-byte header + feature table JSON + batch table JSON + GLB body.
fn wrap_b3dm(glb_bytes: &[u8], building_ids: &[Rc<String>], city_object_types: &[Rc<String>], surface_types: &[Rc<String>]) -> Vec<u8> {
    let batch_length = building_ids.len();

    // Feature table JSON: {"BATCH_LENGTH": n}
    let mut ft_json_bytes = serde_json::to_vec(&serde_json::json!({ "BATCH_LENGTH": batch_length })).unwrap();
    // Pad feature table JSON to 8-byte alignment (header is 28 bytes)
    let ft_padding = (8 - ((28 + ft_json_bytes.len()) % 8)) % 8;
    ft_json_bytes.extend(std::iter::repeat(b' ').take(ft_padding));

    // Batch table JSON: {"BuildingId": [...], "CityObjectType": [...], "SurfaceType": [...]}
    // Collect &str slices for serialization (avoids cloning Rc<String> into owned Strings)
    let ids: Vec<&str> = building_ids.iter().map(|s| s.as_str()).collect();
    let types: Vec<&str> = city_object_types.iter().map(|s| s.as_str()).collect();
    let stypes: Vec<&str> = surface_types.iter().map(|s| s.as_str()).collect();
    let mut bt_json_bytes = serde_json::to_vec(&serde_json::json!({
        "BuildingId": ids,
        "CityObjectType": types,
        "SurfaceType": stypes,
    })).unwrap();
    // Pad batch table JSON to 8-byte alignment
    let bt_padding = (8 - ((28 + ft_json_bytes.len() + bt_json_bytes.len()) % 8)) % 8;
    bt_json_bytes.extend(std::iter::repeat(b' ').take(bt_padding));

    let total = 28 + ft_json_bytes.len() + bt_json_bytes.len() + glb_bytes.len();
    let mut b3dm = Vec::with_capacity(total);

    // 28-byte header
    b3dm.extend_from_slice(b"b3dm");                                       // magic
    b3dm.extend_from_slice(&1u32.to_le_bytes());                           // version
    b3dm.extend_from_slice(&(total as u32).to_le_bytes());                 // byteLength
    b3dm.extend_from_slice(&(ft_json_bytes.len() as u32).to_le_bytes());   // featureTableJSONByteLength
    b3dm.extend_from_slice(&0u32.to_le_bytes());                           // featureTableBinaryByteLength
    b3dm.extend_from_slice(&(bt_json_bytes.len() as u32).to_le_bytes());   // batchTableJSONByteLength
    b3dm.extend_from_slice(&0u32.to_le_bytes());                           // batchTableBinaryByteLength

    // Body
    b3dm.extend_from_slice(&ft_json_bytes);
    b3dm.extend_from_slice(&bt_json_bytes);
    b3dm.extend_from_slice(glb_bytes);

    b3dm
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a MeshBuilder with synthetic data and write to a GLB file.
    /// Returns (raw_glb_bytes, raw_file_bytes) — for V1_0, raw_file_bytes is
    /// the B3DM wrapper and raw_glb_bytes is the embedded GLB extracted from it.
    /// For V1_1, both are identical.
    fn build_test_glb(
        n_vertices: usize,
        n_features: usize,
        tiles_version: TilesVersion,
    ) -> (Vec<u8>, Vec<u8>) {
        assert!(n_vertices % 3 == 0, "n_vertices must be a multiple of 3");
        assert!(n_features >= 1, "need at least 1 feature");

        let transformer = Proj::new_known_crs("EPSG:7415", "EPSG:4978", None)
            .expect("PROJ transformer");
        let mut builder = MeshBuilder::new(&transformer, (0.0, 0.0, 0.0), 0.0, 0.0, 1.0);

        // Populate synthetic vertex data
        for i in 0..n_vertices {
            let f = i as f32;
            builder.positions.push([f, f + 1.0, f + 2.0]);
            builder.normals.push([0.0, 1.0, 0.0]);
            builder.colors.push([1.0, 0.0, 0.0, 1.0]);
            builder.batch_ids.push((i % n_features) as u32);
        }
        for i in 0..n_vertices as u32 {
            builder.indices.push(i);
        }
        builder.next_batch_index = n_features as u32;

        // Varying string lengths to exercise alignment edge cases
        let id_patterns = ["A", "AB", "ABC"];
        let type_patterns = ["Building", "WaterBody", "Bridge"];
        let surface_type_patterns = ["RoofSurface", "WallSurface", "GroundSurface"];
        for i in 0..n_features {
            builder
                .batch_id_to_cityobject_id
                .push(Rc::new(id_patterns[i % id_patterns.len()].to_string()));
            builder
                .batch_id_to_cityobject_type
                .push(Rc::new(type_patterns[i % type_patterns.len()].to_string()));
            builder
                .batch_id_to_surface_type
                .push(Rc::new(surface_type_patterns[i % surface_type_patterns.len()].to_string()));
            // Synthetic CityJSON attributes for testing typed column serialization
            let mut attrs = serde_json::Map::new();
            attrs.insert("slope".into(), serde_json::json!(30.0 + i as f64));
            attrs.insert("azimuth".into(), serde_json::json!(180 + i as i64));
            attrs.insert("material".into(), serde_json::json!(format!("mat_{i}")));
            if i % 2 == 0 {
                attrs.insert("is_roof".into(), serde_json::json!(true));
            }
            builder.batch_id_to_attributes.push(Some(Rc::new(attrs)));
            // Synthetic computed surface properties
            builder.batch_id_to_area.push(10.0 + i as f32);
            builder.batch_id_to_azimuth.push(0.0);
            builder.batch_id_to_elevation.push(90.0);
        }

        let tmp = tempfile::NamedTempFile::new().expect("tempfile");
        let path = tmp.path().to_path_buf();
        builder
            .write_glb(&path, tiles_version, None, &[])
            .expect("write_glb failed");

        let file_bytes = std::fs::read(&path).expect("read GLB file");

        let glb_bytes = match tiles_version {
            TilesVersion::V1_1 => file_bytes.clone(),
            TilesVersion::V1_0 => {
                // Extract embedded GLB from B3DM: skip 28-byte header + feature/batch table JSON
                let ft_json_len =
                    u32::from_le_bytes(file_bytes[12..16].try_into().unwrap()) as usize;
                let bt_json_len =
                    u32::from_le_bytes(file_bytes[20..24].try_into().unwrap()) as usize;
                let glb_start = 28 + ft_json_len + bt_json_len;
                file_bytes[glb_start..].to_vec()
            }
        };

        (glb_bytes, file_bytes)
    }

    #[test]
    fn glb_no_invalid_byte_stride() {
        for &n in &[3, 6, 9, 12, 99, 102] {
            for &version in &[TilesVersion::V1_0, TilesVersion::V1_1] {
                let (glb_bytes, _) = build_test_glb(n, 2, version);
                let gltf = gltf::Gltf::from_slice(&glb_bytes)
                    .unwrap_or_else(|e| panic!("Failed to parse GLB (n={n}, v={version:?}): {e}"));

                for bv in gltf.document.views() {
                    if let Some(stride) = bv.stride() {
                        assert!(
                            stride >= 4,
                            "byteStride {} < 4 on BV {} (n={n}, v={version:?})",
                            stride,
                            bv.index()
                        );
                        assert!(
                            stride % 4 == 0,
                            "byteStride {} not multiple of 4 on BV {} (n={n}, v={version:?})",
                            stride,
                            bv.index()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn glb_buffer_view_offsets_aligned() {
        for &n in &[3, 6, 9, 12, 102] {
            for &version in &[TilesVersion::V1_0, TilesVersion::V1_1] {
                let (glb_bytes, _) = build_test_glb(n, 2, version);
                let gltf = gltf::Gltf::from_slice(&glb_bytes)
                    .unwrap_or_else(|e| panic!("Failed to parse GLB (n={n}, v={version:?}): {e}"));

                for accessor in gltf.document.accessors() {
                    let bv = accessor.view().unwrap();
                    let comp_size = match accessor.data_type() {
                        gltf::accessor::DataType::U8 | gltf::accessor::DataType::I8 => 1,
                        gltf::accessor::DataType::U16 | gltf::accessor::DataType::I16 => 2,
                        gltf::accessor::DataType::U32 | gltf::accessor::DataType::F32 => 4,
                    };
                    let total_offset = accessor.offset() + bv.offset();
                    assert!(
                        total_offset % comp_size == 0,
                        "Accessor {} (BV {}): offset {} not aligned to component size {} (n={n}, v={version:?})",
                        accessor.index(),
                        bv.index(),
                        total_offset,
                        comp_size
                    );
                }
            }
        }
    }

    #[test]
    fn glb_buffer_view_bounds_valid() {
        for &n in &[3, 6, 99, 102] {
            for &version in &[TilesVersion::V1_0, TilesVersion::V1_1] {
                let n_features = if n >= 6 { 3 } else { 1 };
                let (glb_bytes, _) = build_test_glb(n, n_features, version);
                let gltf = gltf::Gltf::from_slice(&glb_bytes)
                    .unwrap_or_else(|e| panic!("Failed to parse GLB (n={n}, v={version:?}): {e}"));

                let buffer_len = gltf.document.buffers().next().unwrap().length();
                for bv in gltf.document.views() {
                    assert!(
                        bv.offset() + bv.length() <= buffer_len,
                        "BV {} exceeds buffer: offset {} + length {} > {} (n={n}, v={version:?})",
                        bv.index(),
                        bv.offset(),
                        bv.length(),
                        buffer_len
                    );
                }
            }
        }
    }

    #[test]
    fn glb_valid_structure() {
        for &version in &[TilesVersion::V1_0, TilesVersion::V1_1] {
            let (glb_bytes, _) = build_test_glb(6, 2, version);
            let gltf = gltf::Gltf::from_slice(&glb_bytes)
                .unwrap_or_else(|e| panic!("Failed to parse GLB (v={version:?}): {e}"));

            assert_eq!(gltf.document.meshes().count(), 1);
            assert_eq!(gltf.document.nodes().count(), 1);
            assert_eq!(gltf.document.scenes().count(), 1);
            assert_eq!(gltf.document.buffers().count(), 1);
            assert_eq!(gltf.document.accessors().count(), 5);

            let expected_bvs = match version {
                TilesVersion::V1_1 => 19, // 5 geometry + 6 string props + 3 computed + 5 attribute columns
                TilesVersion::V1_0 => 5,
            };
            assert_eq!(
                gltf.document.views().count(),
                expected_bvs,
                "Wrong BV count for {version:?}"
            );

            // Verify asset info by parsing JSON chunk directly
            let json_chunk_len =
                u32::from_le_bytes(glb_bytes[12..16].try_into().unwrap()) as usize;
            let json_bytes = &glb_bytes[20..20 + json_chunk_len];
            let root: serde_json::Value = serde_json::from_slice(json_bytes).unwrap();
            assert_eq!(root["asset"]["version"], "2.0");
            assert_eq!(root["asset"]["generator"], "tyler");

            // Verify EXT_mesh_features attribute index is 0 (set index, not accessor index)
            if version == TilesVersion::V1_1 {
                let prim = &root["meshes"][0]["primitives"][0];
                let ext_mf = &prim["extensions"]["EXT_mesh_features"];
                assert_eq!(
                    ext_mf["featureIds"][0]["attribute"], 0,
                    "EXT_mesh_features attribute must be 0 (the set index in _FEATURE_ID_0)"
                );
                assert_eq!(ext_mf["featureIds"][0]["propertyTable"], 0);
            }

            // Verify accessor types and counts
            let accessors: Vec<_> = gltf.document.accessors().collect();
            // Accessor 0: POSITION (Vec3/F32, count=6)
            assert_eq!(accessors[0].count(), 6);
            assert_eq!(accessors[0].data_type(), gltf::accessor::DataType::F32);
            assert_eq!(accessors[0].dimensions(), gltf::accessor::Dimensions::Vec3);
            // Accessor 1: NORMAL (Vec3/F32)
            assert_eq!(accessors[1].data_type(), gltf::accessor::DataType::F32);
            assert_eq!(accessors[1].dimensions(), gltf::accessor::Dimensions::Vec3);
            // Accessor 2: COLOR_0 (Vec4/F32)
            assert_eq!(accessors[2].data_type(), gltf::accessor::DataType::F32);
            assert_eq!(accessors[2].dimensions(), gltf::accessor::Dimensions::Vec4);
            // Accessor 3: BATCHID/FEATURE_ID (Scalar/U16)
            assert_eq!(accessors[3].data_type(), gltf::accessor::DataType::U16);
            assert_eq!(
                accessors[3].dimensions(),
                gltf::accessor::Dimensions::Scalar
            );
            // Accessor 4: indices (Scalar/U32)
            assert_eq!(accessors[4].data_type(), gltf::accessor::DataType::U32);
            assert_eq!(
                accessors[4].dimensions(),
                gltf::accessor::Dimensions::Scalar
            );
            assert_eq!(accessors[4].count(), 6);
        }
    }

    #[test]
    fn glb_chunk_alignment() {
        for &n in &[3, 6] {
            let (glb_bytes, _) = build_test_glb(n, 1, TilesVersion::V1_1);

            // GLB header
            assert_eq!(&glb_bytes[0..4], b"glTF");
            let version = u32::from_le_bytes(glb_bytes[4..8].try_into().unwrap());
            assert_eq!(version, 2);
            let total_len = u32::from_le_bytes(glb_bytes[8..12].try_into().unwrap()) as usize;
            assert_eq!(total_len, glb_bytes.len());

            // JSON chunk alignment
            let json_chunk_len =
                u32::from_le_bytes(glb_bytes[12..16].try_into().unwrap()) as usize;
            assert!(
                json_chunk_len % 4 == 0,
                "JSON chunk length {} not 4-byte aligned (n={n})",
                json_chunk_len
            );

            // BIN chunk alignment
            let bin_chunk_start = 12 + 8 + json_chunk_len;
            let bin_chunk_len =
                u32::from_le_bytes(glb_bytes[bin_chunk_start..bin_chunk_start + 4].try_into().unwrap())
                    as usize;
            assert!(
                bin_chunk_len % 4 == 0,
                "BIN chunk length {} not 4-byte aligned (n={n})",
                bin_chunk_len
            );
        }
    }

    #[test]
    fn b3dm_wrapping_preserves_valid_glb() {
        let (glb_bytes, file_bytes) = build_test_glb(6, 2, TilesVersion::V1_0);

        // Verify B3DM header
        assert_eq!(&file_bytes[0..4], b"b3dm");
        let b3dm_version = u32::from_le_bytes(file_bytes[4..8].try_into().unwrap());
        assert_eq!(b3dm_version, 1);
        let b3dm_total = u32::from_le_bytes(file_bytes[8..12].try_into().unwrap()) as usize;
        assert_eq!(b3dm_total, file_bytes.len());

        // Verify the embedded GLB is valid and passes alignment checks
        let gltf = gltf::Gltf::from_slice(&glb_bytes)
            .expect("B3DM embedded GLB should be valid glTF 2.0");

        for bv in gltf.document.views() {
            if let Some(stride) = bv.stride() {
                assert!(stride >= 4, "byteStride {} < 4 in B3DM GLB", stride);
                assert!(stride % 4 == 0, "byteStride {} not multiple of 4 in B3DM GLB", stride);
            }
        }

        for accessor in gltf.document.accessors() {
            let bv = accessor.view().unwrap();
            let comp_size = match accessor.data_type() {
                gltf::accessor::DataType::U8 | gltf::accessor::DataType::I8 => 1,
                gltf::accessor::DataType::U16 | gltf::accessor::DataType::I16 => 2,
                gltf::accessor::DataType::U32 | gltf::accessor::DataType::F32 => 4,
            };
            let total_offset = accessor.offset() + bv.offset();
            assert!(
                total_offset % comp_size == 0,
                "B3DM GLB: accessor {} offset {} not aligned to {}",
                accessor.index(),
                total_offset,
                comp_size
            );
        }
    }
}
