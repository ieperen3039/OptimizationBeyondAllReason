// Two dimensions
//
// For points in two dimensions, this problem can be solved in time O(n log n) by an algorithm that performs the following steps:[1][2]
//
//     Sort the points in one of the coordinate dimensions (the x-coordinate, say)
//     For each point, in decreasing order by x-coordinate, test whether its y-coordinate is greater
// than the maximum y-coordinate of any previously processed point. (For the first point, this is vacuously true).
// If it is, output the point as one of the maximal points, and remember its y-coordinate as the greatest seen so far.
//
// If the coordinates of the points are assumed to be integers, this can be sped up using integer sorting algorithms,
// to have the same asymptotic running time as the sorting algorithms.[3]
// Three dimensions
//
// For points in three dimensions, it is again possible to find the maximal points in time O(n log n)
// using an algorithm similar to the two-dimensional one that performs the following steps:
//
//     Sort the points in one of the coordinate dimensions (the x-coordinate, say)
//     For each point, in decreasing order by x-coordinate, test whether its projection onto the yz
//          plane is maximal among the set of projections of the set of points processed so far.
//          If it is, output the point as one of the maximal points, and remember its y-coordinate as the greatest seen so far.
//
// This method reduces the problem of computing the maximal points of a static three-dimensional
// point set to one of maintaining the maximal points of a dynamic two-dimensional point set.
// The two-dimensional subproblem can be solved efficiently by using a balanced binary search tree
// to maintain the set of maxima of a dynamic point set. Using this data structure, it is possible
// to test whether a new point is dominated by the existing points, to find and remove the
// previously-undominated points that are dominated by a new point, and to add a new point to the
// set of maximal points, in logarithmic time per point. The number of search tree operations is
// linear over the course of the algorithm, so the total time is O(n log n).[1][2]
//
// For points with integer coordinates the first part of the algorithm, sorting the points, can
// again be sped up by integer sorting. If the points are sorted separately by all three of their
// dimensions, the range of values of their coordinates can be reduced to the range from 1 to n
// without changing the relative order of any two coordinates and without changing the identities
// of the maximal points. After this reduction in the coordinate space, the problem of maintaining
// a dynamic two-dimensional set of maximal points may be solved by using a van Emde Boas tree in
// place of the balanced binary search tree. These changes to the algorithm speed up its running
// time to O(n log log n).[3]

struct Element {
    pub time: f32,
    pub metal: f32,
    pub energy: f32,
}

pub struct ParetoOptimumTracker {
    front: Vec<Element>,
}

#[derive(PartialEq)]
pub enum CheckResult {
    Dominant,
    Inferior,
    Incomparable,
}

impl ParetoOptimumTracker {
    pub fn check(&mut self, time: f32, e_prod: f32, m_prod: f32) -> CheckResult {
        let result = self.front.binary_search_by(|e| e.time.total_cmp(&time));
        match result {
            Ok(idx) => {
                todo!()
            }
            Err(idx) => {}
        }

        todo!()
    }
}
