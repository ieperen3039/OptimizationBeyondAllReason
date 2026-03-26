use crate::pareto::CheckResult::{Dominant, Incomparable, Inferior};

pub type Time = f32;
pub type EProd = f32;
pub type MProd = f32;

pub struct ParetoOptimumTracker {
    tree: TreeNode,
}

#[derive(PartialEq)]
pub enum CheckResult {
    Dominant,
    Inferior,
    Incomparable,
}

impl ParetoOptimumTracker {
    pub fn new() -> Self {
        Self {
            tree: TreeNode {
                time: 0_f32,
                e_prod: f32::MAX,
                m_prod: f32::MAX,
                subnodes: [const { None }; 6],
            },
        }
    }

    pub fn check(&mut self, time: Time, e_prod: EProd, m_prod: MProd) -> CheckResult {
        ParetoOptimumTracker::check_inner(&mut self.tree, time, e_prod, m_prod)
    }

    fn check_inner(node: &mut TreeNode, time: Time, e_prod: EProd, m_prod: MProd) -> CheckResult {
        match (node.time < time, node.e_prod > e_prod, node.m_prod > m_prod) {
            (true, true, true) => {
                *node = TreeNode {
                    time,
                    e_prod,
                    m_prod,
                    subnodes: [const { None }; 6],
                };
                Dominant
            }
            (false, false, false) => {
                Inferior
            }
            (better_t, better_e, better_m) => {
                let mut state = Incomparable;
                if better_t && better_e {
                    state = Self::check_idx(node, 1, time, e_prod, m_prod)
                }
                if state == Incomparable && better_t && better_m {
                    state = Self::check_idx(node, 2, time, e_prod, m_prod)
                }
                if state == Incomparable && better_e && better_m {
                    state = Self::check_idx(node, 5, time, e_prod, m_prod)
                }
                if state == Incomparable && better_t {
                    state = Self::check_idx(node, 0, time, e_prod, m_prod)
                }
                if state == Incomparable && better_e {
                    state = Self::check_idx(node, 3, time, e_prod, m_prod)
                }
                if state == Incomparable && better_m {
                    state = Self::check_idx(node, 4, time, e_prod, m_prod)
                }
                state
            }
        }
    }

    fn check_idx(
        node: &mut TreeNode,
        sub_node_idx: usize,
        time: Time,
        e_prod: EProd,
        m_prod: MProd,
    ) -> CheckResult {
        if let Some(subtree) = &mut node.subnodes[sub_node_idx] {
            Self::check_inner(subtree, time, e_prod, m_prod)
        } else {
            node.subnodes[sub_node_idx] = Some(Box::new(TreeNode {
                time,
                e_prod,
                m_prod,
                subnodes: [const { None }; 6],
            }));
            Incomparable
        }
    }
}

struct TreeNode {
    // dominant point of this node
    time: Time,
    e_prod: EProd,
    m_prod: MProd,
    // 0: dominated in time
    // 1: dominated in time and e
    // 2: dominated in time and m
    // 3: dominated in e
    // 4: dominated in m
    // 5: dominated in e and m
    subnodes: [Option<Box<TreeNode>>; 6],
}
