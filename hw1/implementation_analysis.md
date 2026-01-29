# Report

r12944041

## 1. Implementation Description

### Core Algorithm
The implementation uses a **parallel A* search algorithm** to solve Sokoban puzzles efficiently. The solver employs macro moves (pushing boxes) as fundamental actions rather than individual player movements.

### Key Components

#### State Representation
- **GameState**: Encapsulates box positions, player position, and path reconstruction data
- **Zobrist Hashing**: Fast state hashing using pre-computed random values for efficient duplicate detection
- **Parent Pointers**: Enable complete solution path reconstruction

#### Heuristic Function
- **Adaptive Complexity-Based Approach**:
  - Simple Manhattan distance for small puzzles (≤9x9)
  - Enhanced greedy assignment for complex puzzles (>9x9)
- **Greedy Assignment**: Matches boxes to targets using constraint-based ordering (most constrained boxes first)
- **Congestion Penalty**: Additional cost for puzzles with many boxes

#### Deadlock Detection
- **Corner Deadlock**: Boxes trapped in corners without targets
- **Wall Deadlock**: Adjacent boxes against walls that cannot be separated
- **Grouping Deadlock**: 2x2 box formations that cannot be rearranged
- **Adaptive Detection**: Adjusts based on board openness to avoid false positives in tight layouts

---

## 2. Difficulties Encountered and Solutions

### Primary Challenge: Adaptive Algorithm Design

The most significant difficulty was **recognizing and adapting to different puzzle patterns** across test cases, requiring different optimization strategies for varying puzzle characteristics.

#### Key Challenges and Solutions:

**Pattern Recognition**:
- **Problem**: Test cases range from small 6x6 grids to complex layouts with different target distributions
- **Solution**: Implemented adaptive heuristics switching between simple Manhattan distance (≤9x9 puzzles) and enhanced greedy assignment (complex puzzles)

**Technical Optimization**:
- **Macro Move Implementation**: Replaced individual player movements with atomic box pushes, reducing exponential search complexity
- **Reachability Optimization**: Single BFS pass computing all reachable positions and paths simultaneously for O(1) path reconstruction

#### Remaining Challenges:

**Complex Clustered Patterns (Test cases 22, 23, 25)**:
- **Issue**: Clustered target arrangements require specific box ordering strategies not yet developed
- **Current limitation**: Algorithm recognizes pattern similarity but lacks effective strategies for tight target groupings
- **Potential solutions**: Box sequence dependency analysis, cluster-aware heuristics, specialized move ordering for clustered targets

### Performance Optimization Hotspots:

1. **Deadlock Detection**: Initially caused significant overhead
   - **Solution**: Implemented hierarchical deadlock checks with early termination and caching

2. **State Hash Computation**: Frequent hash calculations for duplicate detection
   - **Solution**: Zobrist hashing with lazy evaluation and caching

3. **Memory Allocation**: Frequent GameState object creation
   - **Solution**: Optimized state copying and reduced object allocations

---

## 3. Strengths and Weaknesses of pthread and OpenMP

### pthread (POSIX Threads)

#### Strengths
1. **Fine-Grained Control**: Direct management of thread creation, synchronization, and termination
2. **Portability**: Available across Unix-like systems with consistent API
3. **Low-Level Access**: Direct access to thread attributes, scheduling policies, and synchronization primitives
4. **Memory Model**: Explicit control over shared memory and thread-local storage
5. **Debugging Support**: Well-established tools and techniques for pthread debugging
6. **Deterministic Behavior**: Predictable thread behavior when properly synchronized

#### Weaknesses
1. **Complexity**: Requires explicit management of thread lifecycle and synchronization
2. **Error-Prone**: Manual mutex management leads to deadlocks, race conditions, and memory leaks
3. **Verbose Code**: Significant boilerplate for thread creation and cleanup
4. **Platform Limitations**: Not available on Windows without additional libraries
5. **Scalability Issues**: No automatic load balancing or work distribution
6. **Resource Management**: Manual handling of thread pools and resource allocation

### OpenMP (Open Multi-Processing)

#### Strengths
1. **Simplicity**: Pragma-based approach requires minimal code changes
2. **Compiler Integration**: Automatic parallelization with compiler support
3. **Load Balancing**: Built-in work distribution strategies (static, dynamic, guided)
4. **Scalability**: Automatic thread management based on available cores
5. **Loop Parallelization**: Excellent for data-parallel operations
6. **Incremental Adoption**: Easy to add parallelism to existing sequential code
7. **Cross-Platform**: Supported by major compilers (GCC, Clang, Intel, MSVC)

#### Weaknesses
1. **Limited Control**: Less fine-grained control over thread behavior
2. **Fork-Join Model**: Restricted to structured parallelism patterns
3. **Debugging Complexity**: Race conditions can be harder to debug due to compiler transformations
4. **Compiler Dependency**: Performance depends heavily on compiler optimization quality
5. **Memory Model**: Implicit memory consistency can lead to subtle bugs
6. **Task Granularity**: Less suitable for irregular or dynamic task distributions
7. **Limited Synchronization**: Fewer synchronization primitives compared to pthread
