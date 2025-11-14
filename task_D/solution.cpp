#include <bits/stdc++.h>
using namespace std;

struct Point {
    double x, y;
};

double getAngle(const Point& from, const Point& to) {
    return atan2(to.y - from.y, to.x - from.x);
}

Point getCentroid(const vector<Point>& pts, const vector<int>& ids) {
    Point center = {0, 0};
    for (int id : ids) {
        center.x += pts[id].x;
        center.y += pts[id].y;
    }
    center.x /= ids.size();
    center.y /= ids.size();
    return center;
}

vector<vector<int>> adj;
vector<Point> points;
vector<int> result;

int getSubtreeSize(int v, int parent) {
    int size = 1;
    for (int u : adj[v]) {
        if (u != parent) {
            size += getSubtreeSize(u, v);
        }
    }
    return size;
}

void dfs(int v, int parent, vector<int> availPts) {
    if (availPts.empty()) return;

    // Pick point for current vertex (first point)
    result[v] = availPts[0];
    Point myPoint = points[availPts[0]];

    // Remove chosen point
    availPts.erase(availPts.begin());

    if (availPts.empty()) return;

    // Get children
    vector<int> children;
    for (int u : adj[v]) {
        if (u != parent) {
            children.push_back(u);
        }
    }

    if (children.empty()) return;

    // Calculate angle to parent (direction to avoid)
    double parentAngle = -M_PI / 2;  // default: down
    if (parent != -1) {
        Point parentPt = points[result[parent]];
        parentAngle = getAngle(myPoint, parentPt);
    }

    // Sort available points by angle relative to parent direction
    vector<pair<double, int>> ptsByAngle;
    for (int pid : availPts) {
        double ang = getAngle(myPoint, points[pid]);
        double relAng = ang - parentAngle;
        // Normalize to [0, 2*PI)
        while (relAng < 0) relAng += 2 * M_PI;
        while (relAng >= 2 * M_PI) relAng -= 2 * M_PI;
        ptsByAngle.push_back({relAng, pid});
    }
    sort(ptsByAngle.begin(), ptsByAngle.end());

    // Get subtree sizes for children
    vector<pair<int, int>> childSizes;  // {size, child_id}
    for (int ch : children) {
        int sz = getSubtreeSize(ch, v);
        childSizes.push_back({sz, ch});
    }

    // First pass: distribute points to compute centroids
    vector<vector<int>> tempChildPts(children.size());
    int idx = 0;
    for (int i = 0; i < children.size(); i++) {
        int need = childSizes[i].first;
        for (int j = 0; j < need && idx < ptsByAngle.size(); j++, idx++) {
            tempChildPts[i].push_back(ptsByAngle[idx].second);
        }
    }

    // Compute centroid for each child and sort children by centroid angle
    vector<tuple<double, int, int>> childrenByAngle;  // {angle, child_id, index}
    for (int i = 0; i < children.size(); i++) {
        if (!tempChildPts[i].empty()) {
            Point centroid = getCentroid(points, tempChildPts[i]);
            double ang = getAngle(myPoint, centroid);
            double relAng = ang - parentAngle;
            while (relAng < 0) relAng += 2 * M_PI;
            while (relAng >= 2 * M_PI) relAng -= 2 * M_PI;
            childrenByAngle.push_back({relAng, childSizes[i].second, childSizes[i].first});
        }
    }
    sort(childrenByAngle.begin(), childrenByAngle.end());

    // Second pass: redistribute points to children in sorted order
    vector<vector<int>> childPts(children.size());
    idx = 0;
    for (int i = 0; i < childrenByAngle.size(); i++) {
        int need = get<2>(childrenByAngle[i]);
        for (int j = 0; j < need && idx < ptsByAngle.size(); j++, idx++) {
            childPts[i].push_back(ptsByAngle[idx].second);
        }
    }

    // Recursively process each child
    for (int i = 0; i < childrenByAngle.size(); i++) {
        int childId = get<1>(childrenByAngle[i]);
        dfs(childId, v, childPts[i]);
    }
}

void solve() {
    int n;
    cin >> n;

    adj.assign(n, vector<int>());
    points.resize(n);
    result.assign(n, -1);

    // Read tree
    for (int i = 0; i < n - 1; i++) {
        int u, v;
        cin >> u >> v;
        u--; v--;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // Read points
    for (int i = 0; i < n; i++) {
        cin >> points[i].x >> points[i].y;
    }

    // Use node 0 as root
    int root = 0;

    // Prepare initial points - put extremal point first for root
    vector<int> allPts(n);
    iota(allPts.begin(), allPts.end(), 0);

    // Find point with highest y-coordinate
    int maxYIdx = 0;
    for (int i = 1; i < n; i++) {
        if (points[i].y > points[maxYIdx].y ||
            (points[i].y == points[maxYIdx].y && points[i].x < points[maxYIdx].x)) {
            maxYIdx = i;
        }
    }

    // Swap to make it first
    swap(allPts[0], allPts[maxYIdx]);

    // Run DFS
    dfs(root, -1, allPts);

    // Output result (1-indexed)
    for (int i = 0; i < n; i++) {
        cout << result[i] + 1;
        if (i + 1 < n) cout << " ";
    }
    cout << "\n";
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int t;
    cin >> t;
    while (t--) {
        solve();
    }

    return 0;
}
