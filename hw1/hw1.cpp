#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <algorithm>
#include <climits>
#include <random>
#include <pthread.h>
#include <memory>
#include <tbb/concurrent_priority_queue.h>
#include <tbb/concurrent_unordered_set.h>
#include <tbb/task_group.h>
#include <atomic>
#include <boost/functional/hash.hpp>

using namespace std;

// Zobrist hashing tables
class ZobristHash {
private:
    static const int MAX_POSITIONS = 256;
    static vector<vector<uint64_t>> boxKeys;
    static bool initialized;

public:
    static void initialize(int totalPositions) {
        if (initialized) return;

        random_device rd;
        mt19937_64 gen(rd());
        uniform_int_distribution<uint64_t> dis;

        boxKeys.resize(totalPositions, vector<uint64_t>(1));

        for (int pos = 0; pos < totalPositions; pos++) {
            boxKeys[pos][0] = dis(gen);
        }

        initialized = true;
    }

    static uint64_t getBoxKey(int position) {
        return boxKeys[position][0];
    }
};

vector<vector<uint64_t>> ZobristHash::boxKeys;
bool ZobristHash::initialized = false;

// Forward declaration
struct GameState;

struct GameState {
    vector<int> boxes;
    int playerPos;
    shared_ptr<GameState> parent;       // Parent state for path reconstruction
    string lastMoveSequence;            // Move sequence from parent to this state
    int gCost;
    int hCost;
    mutable uint64_t boxHash;
    mutable bool hashComputed;

    GameState() : gCost(0), hCost(0), boxHash(0), hashComputed(false) {}

    int fCost() const { return gCost + hCost; }

    uint64_t getBoxHash() const {
        if (!hashComputed) {
            boxHash = 0;
            for (const int& box : boxes) {
                boxHash ^= ZobristHash::getBoxKey(box);
            }
            hashComputed = true;
        }
        return boxHash;
    }


    void invalidateHash() {
        hashComputed = false;
    }

    bool operator==(const GameState& other) const {
        if (playerPos != other.playerPos) return false;
        if (getBoxHash() != other.getBoxHash()) return false;
        return boxes == other.boxes;
    }

    bool operator>(const GameState& other) const {
        if (fCost() != other.fCost()) {
            return fCost() > other.fCost();
        }
        return gCost < other.gCost;
    }
};

struct GameStateHash {
    size_t operator()(const GameState& state) const {
        std::size_t seed = 0;
        boost::hash_combine(seed, state.playerPos);
        boost::hash_range(seed, state.boxes.begin(), state.boxes.end());
        return seed;
    }
};

// Shared data structure for parallel processing
struct SharedData {
    tbb::concurrent_priority_queue<GameState, greater<GameState>> openSet;
    tbb::concurrent_unordered_set<GameState, GameStateHash> closedSet;
    tbb::concurrent_unordered_set<uint64_t> deadlockCache;

    atomic<bool> solutionFound{false};
    string solution;

    atomic<size_t> statesExplored{0};
    atomic<size_t> deadlockRejections{0};
};

class OptimizedSokobanSolver {
private:
    vector<string> originalBoard;
    vector<int> targets;
    unordered_set<int> targetSet;  // Pre-computed target set for efficiency
    int rows, cols;
    GameState initialState;
    SharedData* sharedData;

    const int dx[4] = {-1, 0, 1, 0};
    const int dy[4] = {0, -1, 0, 1};
    const char dirChars[4] = {'W', 'A', 'S', 'D'};

    // Structure to hold reachability results with path reconstruction data
    struct ReachabilityResult {
        unordered_set<int> reachable;
        unordered_map<int, int> parent;     // For path reconstruction
        unordered_map<int, char> moveChar;  // Direction to reach each position

        string getPath(int start, int end) const {
            if (start == end) return "";
            if (reachable.find(end) == reachable.end()) return "";

            string path;
            int pos = end;
            while (parent.find(pos) != parent.end() && parent.at(pos) != start) {
                path = moveChar.at(pos) + path;
                pos = parent.at(pos);
            }
            if (parent.find(pos) != parent.end()) {
                path = moveChar.at(pos) + path;
            }
            return path;
        }
    };

public:
    bool parseInput(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            return false;
        }

        string line;
        while (getline(file, line)) {
            originalBoard.push_back(line);
        }
        file.close();

        if (originalBoard.empty()) {
            return false;
        }

        rows = originalBoard.size();
        cols = originalBoard[0].size();

        ZobristHash::initialize(rows * cols);

        int playerPos = -1;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                char cell = originalBoard[i][j];
                if (cell == 'o' || cell == 'O' || cell == '!') {
                    playerPos = coordToInt(i, j);
                }
                if (cell == 'x' || cell == 'X') {
                    initialState.boxes.push_back(coordToInt(i, j));
                }
                if (cell == '.' || cell == 'O' || cell == 'X') {
                    targets.push_back(coordToInt(i, j));
                }
            }
        }

        sort(initialState.boxes.begin(), initialState.boxes.end());
        initialState.playerPos = playerPos;
        initialState.parent = nullptr;
        initialState.lastMoveSequence = "";
        initialState.gCost = 0;

        // Initialize target set for efficient lookups
        targetSet = unordered_set<int>(targets.begin(), targets.end());

        initialState.hCost = calculateHeuristic(initialState);
        initialState.invalidateHash();

        return true;
    }

private:
    int coordToInt(int row, int col) const {
        return row * cols + col;
    }

    pair<int, int> intToCoord(int pos) const {
        return make_pair(pos / cols, pos % cols);
    }

    bool isValid(int row, int col) const {
        return row >= 0 && row < rows && col >= 0 && col < cols;
    }

    bool isWalkable(int row, int col) const {
        if (!isValid(row, col)) return false;
        return originalBoard[row][col] != '#';
    }

    bool isBoxPlaceable(int row, int col) const {
        if (!isValid(row, col)) return false;
        char cell = originalBoard[row][col];
        return cell != '#' && cell != '@';
    }

    uint64_t calculateBoxHash(const vector<int>& boxes) const {
        uint64_t hash = 0;
        for (const int& box : boxes) {
            hash ^= ZobristHash::getBoxKey(box);
        }
        return hash;
    }

    // Calculate Manhattan distance with obstacles
    int calculatePathCost(int from, int to, const vector<int>& boxes) const {
        // Use vector search for small box counts - faster than unordered_set
        auto isBoxAt = [&](int pos) {
            return find(boxes.begin(), boxes.end(), pos) != boxes.end();
        };

        pair<int, int> fromCoord = intToCoord(from);
        pair<int, int> toCoord = intToCoord(to);

        int baseCost = abs(fromCoord.first - toCoord.first) + abs(fromCoord.second - toCoord.second);

        // Add penalty for obstacles in the path
        int obstaclePenalty = 0;
        int dx = (toCoord.first > fromCoord.first) ? 1 : ((toCoord.first < fromCoord.first) ? -1 : 0);
        int dy = (toCoord.second > fromCoord.second) ? 1 : ((toCoord.second < fromCoord.second) ? -1 : 0);

        int x = fromCoord.first, y = fromCoord.second;
        while (x != toCoord.first || y != toCoord.second) {
            if (x != toCoord.first) x += dx;
            if (y != toCoord.second) y += dy;

            int pos = coordToInt(x, y);
            if (isBoxAt(pos)) {
                obstaclePenalty += 3; // Penalty for having to move other boxes
            }
        }

        return baseCost + obstaclePenalty;
    }

    // ENHANCED HEURISTIC: Adaptive complexity-based approach
    int calculateHeuristic(const GameState& state) const {
        // Use simple heuristic for small puzzles (fast)
        if (rows * cols <= 81) { // 9x9 or smaller
            return calculateSimpleHeuristic(state);
        } else {
            // Use enhanced heuristic for complex puzzles (better guidance)
            return calculateEnhancedHeuristic(state);
        }
    }

    // Original simple heuristic for small puzzles
    int calculateSimpleHeuristic(const GameState& state) const {
        int totalDistance = 0;
        int boxesNotOnTarget = 0;

        for (const int& boxPos : state.boxes) {
            if (targetSet.find(boxPos) != targetSet.end()) {
                continue;
            }

            boxesNotOnTarget++;
            pair<int, int> boxCoord = intToCoord(boxPos);
            int minDistance = INT_MAX;

            for (const int& targetPos : targets) {
                pair<int, int> targetCoord = intToCoord(targetPos);
                int distance = abs(boxCoord.first - targetCoord.first) +
                              abs(boxCoord.second - targetCoord.second);
                minDistance = min(minDistance, distance);
            }

            totalDistance += minDistance;
        }

        return totalDistance + (boxesNotOnTarget * 2);
    }

    // Enhanced greedy heuristic for complex puzzles
    int calculateEnhancedHeuristic(const GameState& state) const {
        // Separate boxes into solved and unsolved
        vector<int> unsolvedBoxes;
        vector<int> availableTargets;

        for (const int& boxPos : state.boxes) {
            if (targetSet.find(boxPos) == targetSet.end()) {
                unsolvedBoxes.push_back(boxPos);
            }
        }

        for (const int& targetPos : targets) {
            bool occupied = false;
            for (const int& boxPos : state.boxes) {
                if (boxPos == targetPos) {
                    occupied = true;
                    break;
                }
            }
            if (!occupied) {
                availableTargets.push_back(targetPos);
            }
        }

        if (unsolvedBoxes.empty()) return 0;

        // GREEDY ASSIGNMENT: Much faster than Hungarian O(n²) vs O(n³)
        vector<bool> targetUsed(availableTargets.size(), false);
        int totalCost = 0;

        // Sort boxes by how constrained they are (fewer good target options)
        vector<pair<int, int>> boxConstraints; // (constraint_score, box_index)
        for (int i = 0; i < unsolvedBoxes.size(); i++) {
            int goodTargets = 0;
            for (int j = 0; j < availableTargets.size(); j++) {
                int cost = calculatePathCost(unsolvedBoxes[i], availableTargets[j], state.boxes);
                if (cost <= 5) goodTargets++; // Within reasonable distance
            }
            boxConstraints.push_back({goodTargets, i});
        }
        sort(boxConstraints.begin(), boxConstraints.end()); // Most constrained first

        // Assign each box to its best available target
        for (auto& constraint : boxConstraints) {
            int boxIdx = constraint.second;
            int boxPos = unsolvedBoxes[boxIdx];

            int bestCost = INT_MAX;
            int bestTarget = -1;

            for (int j = 0; j < availableTargets.size(); j++) {
                if (targetUsed[j]) continue;

                int cost = calculatePathCost(boxPos, availableTargets[j], state.boxes);
                if (cost < bestCost) {
                    bestCost = cost;
                    bestTarget = j;
                }
            }

            if (bestTarget != -1) {
                targetUsed[bestTarget] = true;
                totalCost += bestCost;
            }
        }

        // Light congestion penalty (simplified)
        int congestionPenalty = 0;
        if (unsolvedBoxes.size() > 4) {
            congestionPenalty = unsolvedBoxes.size(); // Small penalty for complexity
        }

        return totalCost + congestionPenalty;
    }

    // Compute reachable positions and path data in single BFS pass
    ReachabilityResult computeReachabilityWithPaths(int start, const vector<int>& boxes) const {
        ReachabilityResult result;

        // Use vector search for small box counts - faster than unordered_set
        auto isBoxAt = [&](int pos) {
            return find(boxes.begin(), boxes.end(), pos) != boxes.end();
        };
        queue<int> q;

        q.push(start);
        result.reachable.insert(start);
        result.parent[start] = -1;

        while (!q.empty()) {
            int current = q.front();
            q.pop();

            pair<int, int> coord = intToCoord(current);
            int row = coord.first, col = coord.second;

            for (int dir = 0; dir < 4; dir++) {
                int newRow = row + dx[dir];
                int newCol = col + dy[dir];
                int newPos = coordToInt(newRow, newCol);

                if (isWalkable(newRow, newCol) &&
                    !isBoxAt(newPos) &&
                    result.reachable.find(newPos) == result.reachable.end()) {

                    result.reachable.insert(newPos);
                    result.parent[newPos] = current;
                    result.moveChar[newPos] = dirChars[dir];
                    q.push(newPos);
                }
            }
        }

        return result;
    }

    // Generate all possible macro moves (box pushes) from current state - OPTIMIZED
    vector<GameState> generateMacroMoves(const GameState& state) const {
        vector<GameState> successors;

        // Use vector search for small box counts (typically < 10 boxes) - faster than unordered_set
        auto isBoxAt = [&](int pos) {
            return find(state.boxes.begin(), state.boxes.end(), pos) != state.boxes.end();
        };

        // Single BFS call to get reachability and path data
        ReachabilityResult reachability = computeReachabilityWithPaths(state.playerPos, state.boxes);

        // Try pushing each box in each direction
        for (size_t boxIdx = 0; boxIdx < state.boxes.size(); boxIdx++) {
            int boxPos = state.boxes[boxIdx];
            pair<int, int> boxCoord = intToCoord(boxPos);
            int boxRow = boxCoord.first, boxCol = boxCoord.second;

            for (int dir = 0; dir < 4; dir++) {
                // Where box would move to
                int newBoxRow = boxRow + dx[dir];
                int newBoxCol = boxCol + dy[dir];
                int newBoxPos = coordToInt(newBoxRow, newBoxCol);

                // Early deadlock filtering - check basic constraints first
                if (!isBoxPlaceable(newBoxRow, newBoxCol) ||
                    isBoxAt(newBoxPos)) {
                    continue;
                }

                // Where player needs to be to push
                int pushFromRow = boxRow - dx[dir];
                int pushFromCol = boxCol - dy[dir];
                int pushFromPos = coordToInt(pushFromRow, pushFromCol);

                // Check if player can reach push position using precomputed reachability
                if (!isWalkable(pushFromRow, pushFromCol) ||
                    isBoxAt(pushFromPos) ||
                    reachability.reachable.find(pushFromPos) == reachability.reachable.end()) {
                    continue;
                }

                // Create new state
                auto newState = make_shared<GameState>();
                newState->boxes = state.boxes;
                newState->boxes[boxIdx] = newBoxPos;
                sort(newState->boxes.begin(), newState->boxes.end());
                newState->playerPos = boxPos;  // Player ends up where box was
                newState->gCost = state.gCost + 1;  // One more push
                newState->hCost = calculateHeuristic(*newState);
                newState->parent = make_shared<GameState>(state);
                newState->invalidateHash();

                // Use precomputed path data instead of calling getMovePath
                string playerMovement = reachability.getPath(state.playerPos, pushFromPos);
                newState->lastMoveSequence = playerMovement + dirChars[dir];

                successors.push_back(*newState);
            }
        }

        return successors;
    }

    bool isSolved(const GameState& state) const {
        for (const int& box : state.boxes) {
            if (targetSet.find(box) == targetSet.end()) {
                return false;
            }
        }
        return true;
    }

    bool isWallOrBoundary(int row, int col) const {
        if (!isValid(row, col)) return true;
        return originalBoard[row][col] == '#';
    }

    bool isSimplifiedDeadlock(const vector<int>& boxes) const {
        for (const int& boxPos : boxes) {
            // Skip if box is already on a target
            if (targetSet.find(boxPos) != targetSet.end()) continue;

            pair<int, int> coord = intToCoord(boxPos);
            int row = coord.first, col = coord.second;

            // Check for L-shaped corner deadlock only (most reliable)
            bool northWall = isWallOrBoundary(row - 1, col);
            bool southWall = isWallOrBoundary(row + 1, col);
            bool westWall = isWallOrBoundary(row, col - 1);
            bool eastWall = isWallOrBoundary(row, col + 1);

            // Corner deadlock: box in corner formed by two perpendicular walls
            if ((northWall && westWall) || (northWall && eastWall) ||
                (southWall && westWall) || (southWall && eastWall)) {
                return true;
            }
        }
        return false;
    }

    bool isGroupingDeadlock(const vector<int>& boxes) const {
        // Use vector search for small box counts - faster than unordered_set
        auto isBoxAt = [&](int pos) {
            return find(boxes.begin(), boxes.end(), pos) != boxes.end();
        };

        for (const int& box : boxes) {
            if (targetSet.find(box) != targetSet.end()) continue;

            pair<int, int> coord = intToCoord(box);
            int row = coord.first, col = coord.second;

            int topRight = coordToInt(row, col + 1);
            int bottomLeft = coordToInt(row + 1, col);
            int bottomRight = coordToInt(row + 1, col + 1);

            if (isBoxAt(topRight) &&
                isBoxAt(bottomLeft) &&
                isBoxAt(bottomRight)) {

                bool allOnTarget = (targetSet.find(box) != targetSet.end() &&
                                   targetSet.find(topRight) != targetSet.end() &&
                                   targetSet.find(bottomLeft) != targetSet.end() &&
                                   targetSet.find(bottomRight) != targetSet.end());

                if (!allOnTarget) {
                    return true;
                }
            }
        }
        return false;
    }

    bool isDeadlock(const GameState& state) const {
        uint64_t boxHash = calculateBoxHash(state.boxes);

        if (sharedData->deadlockCache.find(boxHash) != sharedData->deadlockCache.end()) {
            return true;
        }

        if (isSimplifiedDeadlock(state.boxes)) {
            sharedData->deadlockCache.insert(boxHash);
            return true;
        }

        // Only check grouping deadlock for complex puzzles (large board or many boxes)
        if (rows * cols > 81 || state.boxes.size() > 6) {
            if (isGroupingDeadlock(state.boxes)) {
                sharedData->deadlockCache.insert(boxHash);
                return true;
            }
        }

        return false;
    }

    // Reconstruct full solution path from goal state
    string reconstructSolution(const GameState& goalState) const {
        string solution;
        shared_ptr<GameState> current = goalState.parent;
        vector<string> moveSequences;

        // Collect move sequences in reverse order
        moveSequences.push_back(goalState.lastMoveSequence);

        while (current && current->parent) {
            moveSequences.push_back(current->lastMoveSequence);
            current = current->parent;
        }

        // Build solution string in correct order
        for (int i = moveSequences.size() - 1; i >= 0; i--) {
            solution += moveSequences[i];
        }

        return solution;
    }

    static void* workerThread(void* arg) {
        OptimizedSokobanSolver* solver = static_cast<OptimizedSokobanSolver*>(arg);
        solver->threadWork();
        return nullptr;
    }

    void threadWork() {
        GameState current;

        while (!sharedData->solutionFound) {
            bool hasState = sharedData->openSet.try_pop(current);
            if (!hasState) {
                break;
            }

            // Check closed set
            if (sharedData->closedSet.find(current) != sharedData->closedSet.end()) {
                continue;
            }

            sharedData->closedSet.insert(current);
            sharedData->statesExplored++;

            if (isSolved(current)) {
                if (!sharedData->solutionFound.exchange(true)) {
                    sharedData->solution = reconstructSolution(current);
                }
                break;
            }

            // Expand successors
            vector<GameState> successors = generateMacroMoves(current);

            for (const GameState& nextState : successors) {
                if (isDeadlock(nextState)) {
                    sharedData->deadlockRejections++;
                    continue;
                }

                if (sharedData->closedSet.find(nextState) == sharedData->closedSet.end()) {
                    sharedData->openSet.push(nextState);
                }
            }
        }
    }

public:
    string solve() {
        sharedData = new SharedData();
        sharedData->openSet.push(initialState);

        const int NUM_THREADS = 6;
        tbb::task_group tg;

        for (int i = 0; i < NUM_THREADS; i++) {
            tg.run([this]() { this->threadWork(); });
        }

        tg.wait(); // wait for all threads

        string result = sharedData->solution;
        delete sharedData;
        return result;
    }
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        return 1;
    }

    OptimizedSokobanSolver solver;
    if (!solver.parseInput(argv[1])) {
        return 1;
    }

    string solution = solver.solve();

    if (solution.empty()) {
        return 1;
    }

    cout << solution << endl;
    return 0;
}