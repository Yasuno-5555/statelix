#ifndef STATELIX_BRANCH_BOUND_H
#define STATELIX_BRANCH_BOUND_H

#include <vector>
#include <queue>
#include <functional>
#include <limits>
#include <algorithm>

namespace statelix {

// Branch & Bound for Integer Linear Programming (ILP)
// Minimize c^T x subject to Ax <= b, x in {0, 1}^n (Binary ILP)

struct BnBNode {
    std::vector<int> fixed;  // -1 = unfixed, 0 = fixed to 0, 1 = fixed to 1
    double lower_bound;
    int level;
    
    bool operator>(const BnBNode& other) const {
        return lower_bound > other.lower_bound; // Min-heap
    }
};

struct BnBResult {
    std::vector<int> solution;
    double objective;
    int nodes_explored;
    bool feasible;
};

class BranchAndBound {
public:
    int max_nodes = 10000;
    
    // Solve Binary ILP: min c'x s.t. Ax <= b, x in {0,1}^n
    // c: objective coefficients (n x 1)
    // A: constraint matrix (m x n)
    // b: constraint RHS (m x 1)
    BnBResult solve(const std::vector<double>& c,
                    const std::vector<std::vector<double>>& A,
                    const std::vector<double>& b) {
        
        int n = c.size();
        int m = A.size();
        
        double best_obj = std::numeric_limits<double>::infinity();
        std::vector<int> best_solution(n, 0);
        bool found_feasible = false;
        int nodes_explored = 0;
        
        // Priority queue (min-heap by lower bound)
        std::priority_queue<BnBNode, std::vector<BnBNode>, std::greater<BnBNode>> pq;
        
        // Root node
        BnBNode root;
        root.fixed.resize(n, -1);
        root.lower_bound = compute_relaxation_bound(c, A, b, root.fixed);
        root.level = 0;
        pq.push(root);
        
        while (!pq.empty() && nodes_explored < max_nodes) {
            BnBNode node = pq.top();
            pq.pop();
            nodes_explored++;
            
            // Prune: if lower bound >= best, skip
            if (node.lower_bound >= best_obj) continue;
            
            // Check if all variables are fixed (leaf node)
            int branch_var = -1;
            for(int i = 0; i < n; ++i) {
                if (node.fixed[i] == -1) {
                    branch_var = i;
                    break;
                }
            }
            
            if (branch_var == -1) {
                // All fixed - check feasibility and update best
                std::vector<int> sol(n);
                for(int i = 0; i < n; ++i) sol[i] = node.fixed[i];
                
                if (is_feasible(A, b, sol)) {
                    double obj = 0;
                    for(int i = 0; i < n; ++i) obj += c[i] * sol[i];
                    if (obj < best_obj) {
                        best_obj = obj;
                        best_solution = sol;
                        found_feasible = true;
                    }
                }
                continue;
            }
            
            // Branch on branch_var
            for(int val = 0; val <= 1; ++val) {
                BnBNode child;
                child.fixed = node.fixed;
                child.fixed[branch_var] = val;
                child.level = node.level + 1;
                child.lower_bound = compute_relaxation_bound(c, A, b, child.fixed);
                
                if (child.lower_bound < best_obj) {
                    pq.push(child);
                }
            }
        }
        
        return {best_solution, best_obj, nodes_explored, found_feasible};
    }

private:
    // Simple LP relaxation bound (greedy for now)
    double compute_relaxation_bound(const std::vector<double>& c,
                                    const std::vector<std::vector<double>>& A,
                                    const std::vector<double>& b,
                                    const std::vector<int>& fixed) {
        int n = c.size();
        double bound = 0;
        
        // Sum fixed variables' contributions
        for(int i = 0; i < n; ++i) {
            if (fixed[i] == 1) {
                bound += c[i];
            } else if (fixed[i] == -1) {
                // For unfixed: take min(0, c[i]) as optimistic bound
                if (c[i] < 0) bound += c[i];
            }
        }
        
        return bound;
    }
    
    bool is_feasible(const std::vector<std::vector<double>>& A,
                     const std::vector<double>& b,
                     const std::vector<int>& x) {
        int m = A.size();
        int n = x.size();
        
        for(int i = 0; i < m; ++i) {
            double lhs = 0;
            for(int j = 0; j < n; ++j) {
                lhs += A[i][j] * x[j];
            }
            if (lhs > b[i] + 1e-9) return false;
        }
        return true;
    }
};

} // namespace statelix

#endif // STATELIX_BRANCH_BOUND_H
