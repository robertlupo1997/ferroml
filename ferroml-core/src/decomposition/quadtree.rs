//! Quad-tree for Barnes-Hut force approximation in 2D t-SNE.
//!
//! The quad-tree recursively partitions 2D space into quadrants,
//! storing center-of-mass and total mass for each cell. During gradient
//! computation, distant cells are approximated as single points when
//! `cell_size / distance < theta` (the Barnes-Hut criterion).

#![allow(clippy::doc_markdown)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

use ndarray::Array2;

/// A 2D bounding box.
#[derive(Debug, Clone)]
struct BoundingBox {
    cx: f64,
    cy: f64,
    half_size: f64,
}

impl BoundingBox {
    /// Return the quadrant index (0..4) for point (x, y).
    fn quadrant(&self, x: f64, y: f64) -> usize {
        let right = x > self.cx;
        let top = y > self.cy;
        match (right, top) {
            (false, false) => 0, // SW
            (true, false) => 1,  // SE
            (false, true) => 2,  // NW
            (true, true) => 3,   // NE
        }
    }

    fn child_box(&self, quadrant: usize) -> Self {
        let q = self.half_size / 2.0;
        let (dx, dy) = match quadrant {
            0 => (-q, -q), // SW
            1 => (q, -q),  // SE
            2 => (-q, q),  // NW
            3 => (q, q),   // NE
            _ => unreachable!(),
        };
        Self {
            cx: self.cx + dx,
            cy: self.cy + dy,
            half_size: q,
        }
    }
}

/// A node in the quad-tree.
#[derive(Debug)]
enum QTNode {
    Empty,
    Leaf {
        x: f64,
        y: f64,
        mass: f64,
    },
    Internal {
        center_of_mass_x: f64,
        center_of_mass_y: f64,
        total_mass: f64,
        bbox: BoundingBox,
        children: [Box<QTNode>; 4],
    },
}

/// Quad-tree for Barnes-Hut approximation in 2D.
///
/// Stores the embedding points and supports computing approximate
/// repulsive forces via the Barnes-Hut criterion.
#[derive(Debug)]
pub struct QuadTree {
    root: QTNode,
    bbox: BoundingBox,
}

impl QuadTree {
    /// Build a quad-tree from a 2D embedding (n x 2 array).
    ///
    /// # Panics
    /// Panics if the embedding does not have exactly 2 columns.
    pub fn new(embedding: &Array2<f64>) -> Self {
        assert_eq!(
            embedding.ncols(),
            2,
            "QuadTree requires 2D embedding (n x 2)"
        );

        let n = embedding.nrows();
        if n == 0 {
            return Self {
                root: QTNode::Empty,
                bbox: BoundingBox {
                    cx: 0.0,
                    cy: 0.0,
                    half_size: 1.0,
                },
            };
        }

        // Compute bounding box
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for i in 0..n {
            let x = embedding[[i, 0]];
            let y = embedding[[i, 1]];
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }

        let cx = (min_x + max_x) / 2.0;
        let cy = (min_y + max_y) / 2.0;
        let half_size = ((max_x - min_x).max(max_y - min_y) / 2.0) + 1e-10;

        let bbox = BoundingBox { cx, cy, half_size };
        let mut root = QTNode::Empty;

        for i in 0..n {
            let x = embedding[[i, 0]];
            let y = embedding[[i, 1]];
            Self::insert(&mut root, x, y, 1.0, &bbox, 0);
        }

        Self { root, bbox }
    }

    /// Insert a point into the tree.
    fn insert(node: &mut QTNode, x: f64, y: f64, mass: f64, bbox: &BoundingBox, depth: usize) {
        // Prevent infinite recursion for coincident points
        if depth > 50 {
            // Just accumulate mass at this node
            match node {
                QTNode::Leaf {
                    mass: ref mut m, ..
                } => {
                    *m += mass;
                }
                _ => {
                    *node = QTNode::Leaf { x, y, mass };
                }
            }
            return;
        }

        match std::mem::replace(node, QTNode::Empty) {
            QTNode::Empty => {
                *node = QTNode::Leaf { x, y, mass };
            }
            QTNode::Leaf {
                x: lx,
                y: ly,
                mass: lm,
            } => {
                // Split: create internal node with 4 empty children
                let total_mass = lm + mass;
                let com_x = (lx * lm + x * mass) / total_mass;
                let com_y = (ly * lm + y * mass) / total_mass;

                let mut children: [Box<QTNode>; 4] = [
                    Box::new(QTNode::Empty),
                    Box::new(QTNode::Empty),
                    Box::new(QTNode::Empty),
                    Box::new(QTNode::Empty),
                ];

                // Re-insert the old leaf point
                let q1 = bbox.quadrant(lx, ly);
                let child_bbox1 = bbox.child_box(q1);
                Self::insert(&mut children[q1], lx, ly, lm, &child_bbox1, depth + 1);

                // Insert the new point
                let q2 = bbox.quadrant(x, y);
                let child_bbox2 = bbox.child_box(q2);
                Self::insert(&mut children[q2], x, y, mass, &child_bbox2, depth + 1);

                *node = QTNode::Internal {
                    center_of_mass_x: com_x,
                    center_of_mass_y: com_y,
                    total_mass,
                    bbox: bbox.clone(),
                    children,
                };
            }
            QTNode::Internal {
                center_of_mass_x: com_x,
                center_of_mass_y: com_y,
                total_mass,
                bbox: ref node_bbox,
                mut children,
            } => {
                // Update center of mass
                let new_total = total_mass + mass;
                let new_com_x = (com_x * total_mass + x * mass) / new_total;
                let new_com_y = (com_y * total_mass + y * mass) / new_total;

                // Insert into appropriate child
                let q = node_bbox.quadrant(x, y);
                let child_bbox = node_bbox.child_box(q);
                Self::insert(&mut children[q], x, y, mass, &child_bbox, depth + 1);

                *node = QTNode::Internal {
                    center_of_mass_x: new_com_x,
                    center_of_mass_y: new_com_y,
                    total_mass: new_total,
                    bbox: node_bbox.clone(),
                    children,
                };
            }
        }
    }

    /// Compute repulsive (non-edge) forces on a point using Barnes-Hut approximation.
    ///
    /// Returns `(force_x, force_y, sum_Q)` where:
    /// - `(force_x, force_y)` is the repulsive force contribution
    /// - `sum_Q` is the sum of q_{ij} values (Student-t kernel denominators)
    ///
    /// `theta` controls the accuracy/speed tradeoff:
    /// - theta = 0: exact (no approximation)
    /// - theta = 0.5: good balance (default)
    /// - theta = 1.0: fast but less accurate
    pub fn compute_non_edge_forces(
        &self,
        point_x: f64,
        point_y: f64,
        theta: f64,
    ) -> (f64, f64, f64) {
        let mut force_x = 0.0;
        let mut force_y = 0.0;
        let mut sum_q = 0.0;

        Self::compute_forces_recursive(
            &self.root,
            &self.bbox,
            point_x,
            point_y,
            theta,
            &mut force_x,
            &mut force_y,
            &mut sum_q,
        );

        (force_x, force_y, sum_q)
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_forces_recursive(
        node: &QTNode,
        _bbox: &BoundingBox,
        px: f64,
        py: f64,
        theta: f64,
        force_x: &mut f64,
        force_y: &mut f64,
        sum_q: &mut f64,
    ) {
        match node {
            QTNode::Empty => {}
            QTNode::Leaf { x, y, mass } => {
                let dx = px - x;
                let dy = py - y;
                let dist_sq = dx * dx + dy * dy;

                if dist_sq < 1e-30 {
                    return; // Skip self-interaction
                }

                let q_ij = 1.0 / (1.0 + dist_sq); // Student-t kernel
                let mult = *mass * q_ij * q_ij; // mass * q_ij^2

                *force_x += mult * dx;
                *force_y += mult * dy;
                *sum_q += *mass * q_ij;
            }
            QTNode::Internal {
                center_of_mass_x: com_x,
                center_of_mass_y: com_y,
                total_mass,
                bbox: node_bbox,
                children,
            } => {
                let dx = px - com_x;
                let dy = py - com_y;
                let dist_sq = dx * dx + dy * dy;
                let cell_size = node_bbox.half_size * 2.0;

                // Barnes-Hut criterion: cell_size / sqrt(dist_sq) < theta
                if cell_size * cell_size < theta * theta * dist_sq {
                    // Treat entire cell as a single point at center of mass
                    if dist_sq < 1e-30 {
                        return;
                    }
                    let q_ij = 1.0 / (1.0 + dist_sq);
                    let mult = *total_mass * q_ij * q_ij;

                    *force_x += mult * dx;
                    *force_y += mult * dy;
                    *sum_q += *total_mass * q_ij;
                } else {
                    // Recurse into children
                    for (i, child) in children.iter().enumerate() {
                        let child_bbox = node_bbox.child_box(i);
                        Self::compute_forces_recursive(
                            child,
                            &child_bbox,
                            px,
                            py,
                            theta,
                            force_x,
                            force_y,
                            sum_q,
                        );
                    }
                }
            }
        }
    }

    /// Get the total mass (number of points) in the tree.
    pub fn total_mass(&self) -> f64 {
        Self::node_mass(&self.root)
    }

    fn node_mass(node: &QTNode) -> f64 {
        match node {
            QTNode::Empty => 0.0,
            QTNode::Leaf { mass, .. } => *mass,
            QTNode::Internal { total_mass, .. } => *total_mass,
        }
    }

    /// Get the center of mass of the entire tree.
    pub fn center_of_mass(&self) -> Option<(f64, f64)> {
        match &self.root {
            QTNode::Empty => None,
            QTNode::Leaf { x, y, .. } => Some((*x, *y)),
            QTNode::Internal {
                center_of_mass_x,
                center_of_mass_y,
                ..
            } => Some((*center_of_mass_x, *center_of_mass_y)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_quadtree_correct_center_of_mass() {
        let embedding = array![[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0],];
        let tree = QuadTree::new(&embedding);

        let (cx, cy) = tree.center_of_mass().unwrap();
        assert!(
            (cx - 1.0).abs() < 1e-10,
            "Center of mass x should be 1.0, got {}",
            cx
        );
        assert!(
            (cy - 1.0).abs() < 1e-10,
            "Center of mass y should be 1.0, got {}",
            cy
        );
        assert!((tree.total_mass() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_quadtree_single_point() {
        let embedding = array![[3.0, 4.0]];
        let tree = QuadTree::new(&embedding);

        let (cx, cy) = tree.center_of_mass().unwrap();
        assert!((cx - 3.0).abs() < 1e-10);
        assert!((cy - 4.0).abs() < 1e-10);
        assert!((tree.total_mass() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_quadtree_empty() {
        let embedding = Array2::<f64>::zeros((0, 2));
        let tree = QuadTree::new(&embedding);
        assert!(tree.center_of_mass().is_none());
        assert!((tree.total_mass()).abs() < 1e-10);
    }

    #[test]
    fn test_quadtree_forces_repulsive() {
        // Two points should repel each other
        let embedding = array![[0.0, 0.0], [1.0, 0.0]];
        let tree = QuadTree::new(&embedding);

        // Force on point at (0, 0)
        let (fx, fy, sum_q) = tree.compute_non_edge_forces(0.0, 0.0, 0.0);

        // Force should push point 0 to the left (negative x)
        assert!(
            fx < 0.0,
            "Force should be repulsive (negative x), got {}",
            fx
        );
        assert!(
            fy.abs() < 1e-10,
            "Force in y should be near zero, got {}",
            fy
        );
        assert!(sum_q > 0.0, "sum_Q should be positive");
    }

    #[test]
    fn test_quadtree_barnes_hut_converges_to_exact() {
        // As theta -> 0, Barnes-Hut should converge to exact forces
        use rand::SeedableRng;
        use rand_distr::{Distribution, StandardNormal};

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let n = 50;
        let mut embedding = Array2::zeros((n, 2));
        for i in 0..n {
            embedding[[i, 0]] = StandardNormal.sample(&mut rng);
            embedding[[i, 1]] = StandardNormal.sample(&mut rng);
        }

        let tree = QuadTree::new(&embedding);

        // Query point
        let px = embedding[[0, 0]];
        let py = embedding[[0, 1]];

        // Exact forces (theta = 0)
        let (exact_fx, exact_fy, exact_sq) = tree.compute_non_edge_forces(px, py, 0.0);

        // Approximate forces (theta = 0.1 — very accurate)
        let (approx_fx, approx_fy, approx_sq) = tree.compute_non_edge_forces(px, py, 0.1);

        // Should be very close
        let fx_err = ((exact_fx - approx_fx) / (exact_fx.abs() + 1e-12)).abs();
        let fy_err = ((exact_fy - approx_fy) / (exact_fy.abs() + 1e-12)).abs();
        let sq_err = ((exact_sq - approx_sq) / (exact_sq.abs() + 1e-12)).abs();

        assert!(
            fx_err < 0.05,
            "Force x relative error {} should be < 5%",
            fx_err
        );
        assert!(
            fy_err < 0.05,
            "Force y relative error {} should be < 5%",
            fy_err
        );
        assert!(
            sq_err < 0.05,
            "sum_Q relative error {} should be < 5%",
            sq_err
        );
    }

    #[test]
    fn test_quadtree_coincident_points() {
        // All points at the same location
        let embedding = array![[1.0, 1.0], [1.0, 1.0], [1.0, 1.0],];
        let tree = QuadTree::new(&embedding);
        assert!((tree.total_mass() - 3.0).abs() < 1e-10);
        let (cx, cy) = tree.center_of_mass().unwrap();
        assert!((cx - 1.0).abs() < 1e-10);
        assert!((cy - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_quadtree_asymmetric_layout() {
        // Test center of mass with asymmetric point distribution
        let embedding = array![[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [3.0, 0.0],];
        let tree = QuadTree::new(&embedding);
        let (cx, cy) = tree.center_of_mass().unwrap();
        // COM = (0+0+0+3)/4 = 0.75, (0+0+0+0)/4 = 0.0
        assert!(
            (cx - 0.75).abs() < 1e-10,
            "Expected COM x = 0.75, got {}",
            cx
        );
        assert!((cy).abs() < 1e-10, "Expected COM y = 0.0, got {}", cy);
    }
}
